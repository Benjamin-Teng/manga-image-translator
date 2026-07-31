# Animated Image Support (GIF / animated WebP)

**Date:** 2026-07-31
**Status:** Approved
**Scope:** Translate animated images by running the pipeline once and compositing the
rendered text layer onto every frame.

## Problem

The translator has no animation support:

- `manga_translator/mode/local.py:252` carries a `# TODO: Add .gif handler` that was never
  implemented. `.gif` falls through to the "treat as image" branch.
- `save.py` registers `png`, `webp`, `jpg`/`jpeg`, `xcf`/`psd`/`pdf`. `gif` is absent, so
  `save_result()` raises `FormatNotSupportedException` at save time — *after* the whole
  translation has already run.
- `ImageFormat._save()` calls `result.save(dest)` with no `save_all=True`, so an animated
  WebP input is silently reduced to a single frame. This is worse than the GIF case: it
  fails quietly and the user loses the animation without any error.

Both input paths read frames via `Image.open()`, which yields only frame 0.

## Goal

Given an animated input, produce an animated output in which every frame has the original
text removed and the translated text drawn on it.

**Non-goal:** per-frame detection/OCR/translation. That costs N times a single image and
produces text that jitters between frames because OCR and translation results differ per
frame.

## Core approach

Run the existing pipeline exactly once on frame 0, derive a text overlay from what the
pipeline changed, and paste that overlay onto every frame.

This works because the target use case is *動態漫畫* — animated manga where the artwork
moves but the speech bubbles are static. Where the bubbles are static, freezing frame 0's
inpainted background inside the bubble is invisible.

### Deriving the overlay

The pipeline (`manga_translator.py:596-632`) leaves these on the context:

| Field | Contents |
|---|---|
| `ctx.img_rgb` | source RGB |
| `ctx.mask` | mask of the original text |
| `ctx.img_inpainted` | background with original text removed |
| `ctx.img_rendered` | inpainted background + translated text |

The overlay is the **pixel difference** between source and rendered:

```python
changed = (img_rendered != img_rgb).any(axis=2)
overlay = RGBA(img_rendered, alpha=changed * 255)
```

This captures both halves of the edit — the erased original text *and* the drawn
translation — without needing to know where glyphs were placed.

Alternatives rejected:

- **Mask-based** (`ctx.mask` dilated): translated text is frequently longer than the
  source and overflows the original mask, so the overflow would be clipped.
- **Bounding-box patches** (`ctx.text_regions`): coarsest option; opaque rectangles show
  visible seams whenever anything near the bubble edge moves.

The difference approach also does not depend on rendering internals, so it survives future
changes to how text is drawn.

### Edge cases in the overlay approach

1. **Colorizer enabled** — every pixel changes, so the difference covers the whole frame
   and the output animation freezes into frame 0. Detect a change ratio above 50% and log
   a warning. Do not silently produce a frozen animation.
2. **Upscaling enabled** — `img_rendered` is at the upscaled resolution while the frames
   are at source resolution. The overlay is always resized to the frame size before
   compositing.

## Architecture

The existing pipeline is not modified. `translate()` receives a single PIL image (frame 0)
exactly as it does today. All new logic sits around it.

### New module: `manga_translator/animation.py`

```python
@dataclass
class Animation:
    frames: list[Image.Image]   # RGBA, fully composited canvases
    durations: list[int]        # milliseconds per frame
    loop: int                   # 0 = infinite

def load_animation(path) -> Animation | None   # None when the input is not animated
def build_overlay(src_rgb, rendered_rgb, size) -> Image.Image   # RGBA
def apply_overlay(anim, overlay) -> Animation
```

Self-contained, depends only on PIL and numpy, testable in isolation.

### Changes to existing files

| File | Change | Size |
|---|---|---|
| `save.py` | add `GifFormat`; make `ImageFormat` animation-aware | ~25 lines |
| `mode/local.py` | animation branch in `_translate_file`; `_output_ext()` for the extension decision; route animated files out of the batch collector | ~35 lines |
| `args.py` | add `--anim-format`; `--format` choices derive from `OUTPUT_FORMATS`, so registering `GifFormat` exposes `gif` automatically | 1 line |

### `--anim-format`

`--format` applies to every file, so forcing GIF to get phone-playable output
would also turn static manga pages into 256-colour GIFs. `--anim-format` applies
only to inputs that actually hold more than one frame, leaving static pages at
their own extension. `is_animated_path()` answers that without decoding every
frame, so it is cheap enough to run while the output paths are still being
planned.

The bat launchers default to `--anim-format gif`, since playing back on a phone
is the reason this feature exists; `MT_ANIM_FORMAT=webp` keeps full colour and
`MT_ANIM_FORMAT=none` restores the source extension.

### Data flow

```
input.webp (56 frames)
  ├─ load_animation() ──────→ frames[], durations[], loop
  ├─ frames[0] ─→ translate() ─→ ctx.img_rgb, ctx.img_rendered    (unchanged pipeline)
  │                                    ↓
  │                             build_overlay()
  │                                    ↓
  │                       overlay (RGBA, opaque only where changed)
  └──────────────────────→ apply_overlay() onto every frame
                                       ↓
                    ctx.anim_frames / anim_durations / anim_loop
                                       ↓
                            save_result()      (signature unchanged)
                              ↙          ↘
                        output.gif    output.webp
```

Animation data travels on `ctx`, so `save_result(result, dest, ctx)` keeps its signature
and the format-registration mechanism is untouched.

## GIF encoding

A shared palette across all frames is required. PIL quantizes each frame independently by
default, giving each frame a different palette and producing visible colour flicker during
playback — worse than the banding from 256 colours.

1. Sample frames at a stride chosen so at most 32 frames contribute (`stride =
   ceil(n_frames / 32)`), merge them into one tall image, and
   `quantize(colors=255, method=MEDIANCUT)` to obtain a single palette.
2. Reserve one index for transparency (GIF transparency is 1-bit).
3. Quantize every frame against that single palette with Floyd–Steinberg dithering.

**Alpha:** GIF has no partial transparency. Pixels that are not fully opaque are composited
onto white, matching the existing `load_image()` behaviour at
`manga_translator/utils/generic.py:227`.

**File size:** 600×900×56 frames is roughly 8–15 MB. Log the written size; suggest
`--format webp` above 20 MB. Do not silently downscale or drop frames — that is the user's
decision.

## Error handling

| Condition | Behaviour |
|---|---|
| Input not animated | Unchanged from today; `load_animation()` returns `None` |
| Animated input, `--format jpg`/`png` | Write frame 0 and warn that the animation was dropped |
| Animated input, colorizer on | Warn about the frozen-animation result, still produce output |
| Frame decode failure mid-sequence | Fail the file with a clear error; do not write a truncated animation |
| Missing/`None` per-frame duration | Fall back to 100 ms (PIL reports `None` for frame 0) |

## Testing

Fixtures are generated synthetically so the suite has no binary dependencies and can cover
both cases precisely:

- **static-bubble fixture** — moving gradient background, stationary white bubble with
  source text. The primary supported case.
- **moving-bubble fixture** — bubble translates across frames. Documents the known
  limitation rather than asserting correctness.

Unit tests:

1. `load_animation()` returns `None` for a static PNG.
2. `load_animation()` recovers frame count, per-frame durations, and loop from an animated
   WebP and from a GIF.
3. `build_overlay()` produces alpha 0 where source and rendered agree, 255 where they
   differ.
4. `build_overlay()` resizes to the requested frame size when the rendered image is larger
   (upscaling case).
5. `apply_overlay()` leaves pixels outside the overlay untouched on every frame.
6. GIF round-trip preserves frame count, durations, and loop.
7. Animated WebP round-trip preserves frame count, durations, and loop.
8. A single shared palette is used across GIF frames.

End-to-end QA runs the CLI against a synthetic animated fixture for `--format gif` and
`--format webp`, then reopens both outputs to verify frame count, timing, loop, and that
the text overlay is present on the last frame as well as the first.
