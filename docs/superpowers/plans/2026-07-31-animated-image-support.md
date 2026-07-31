# Animated Image Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Translate animated GIF/WebP by running the pipeline once on frame 0 and compositing the resulting text overlay onto every frame.

**Architecture:** A new self-contained `manga_translator/animation.py` loads frames, derives an RGBA overlay from the pixel difference between `ctx.img_rgb` and `ctx.img_rendered`, and composites it onto each frame. The existing translation pipeline is not modified. Animation data rides on `ctx` so `save_result()` keeps its signature.

**Tech Stack:** Python 3.x, Pillow 12.2.0, NumPy, pytest.

## Global Constraints

- Static-image behaviour must not change. `load_animation()` returns `None` for non-animated input and every existing code path stays as-is.
- The translation pipeline (`manga_translator/manga_translator.py`) must not be edited.
- `save_result(result, dest, ctx)` keeps its current signature.
- `args.py` must not be edited — `--format` choices derive from `OUTPUT_FORMATS`.
- GIF frames must share one palette (per-frame palettes cause playback flicker).
- Default frame duration when PIL reports `None` is 100 ms.
- Palette sampling uses at most 32 frames: `stride = ceil(n_frames / 32)`.
- Tests live in `test/`, run with `pytest` from the repo root (`pytest.ini` sets `pythonpath = .`).

---

### Task 1: Animation loading

**Files:**
- Create: `manga_translator/animation.py`
- Test: `test/test_animation.py`

**Interfaces:**
- Consumes: nothing
- Produces: `Animation` dataclass with fields `frames: list[Image.Image]` (RGBA), `durations: list[int]` (ms), `loop: int`; `load_animation(path: str) -> Animation | None`; module constant `DEFAULT_FRAME_DURATION_MS = 100`

- [ ] **Step 1: Write the failing tests**

```python
# test/test_animation.py
import numpy as np
import pytest
from PIL import Image

from manga_translator.animation import (
    DEFAULT_FRAME_DURATION_MS,
    Animation,
    apply_overlay,
    build_overlay,
    load_animation,
    overlay_coverage,
)


def _make_frames(n=4, size=(40, 30)):
    """n frames whose background shifts every frame."""
    frames = []
    for i in range(n):
        arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        arr[:, :, 0] = (i * 60) % 256          # moving background
        frames.append(Image.fromarray(arr, 'RGB').convert('RGBA'))
    return frames


def _write_anim(path, frames, duration=40, loop=0, fmt='WEBP'):
    frames[0].save(
        path, format=fmt, save_all=True, append_images=frames[1:],
        duration=duration, loop=loop, lossless=True,
    )
    return str(path)


def test_load_animation_returns_none_for_static_png(tmp_path):
    p = tmp_path / 'static.png'
    Image.new('RGB', (10, 10), 'white').save(p)
    assert load_animation(str(p)) is None


def test_load_animation_reads_webp_frames_durations_and_loop(tmp_path):
    path = _write_anim(tmp_path / 'a.webp', _make_frames(5), duration=40, loop=0)
    anim = load_animation(path)
    assert isinstance(anim, Animation)
    assert len(anim.frames) == 5
    assert anim.durations == [40] * 5
    assert anim.loop == 0
    assert all(f.mode == 'RGBA' for f in anim.frames)


def test_load_animation_reads_gif(tmp_path):
    path = _write_anim(tmp_path / 'a.gif', _make_frames(3), duration=80, fmt='GIF')
    anim = load_animation(path)
    assert len(anim.frames) == 3
    assert anim.durations == [80] * 3


def test_load_animation_defaults_missing_duration(tmp_path):
    frames = _make_frames(3)
    p = tmp_path / 'nodur.webp'
    frames[0].save(p, format='WEBP', save_all=True, append_images=frames[1:], lossless=True)
    anim = load_animation(str(p))
    assert all(d == DEFAULT_FRAME_DURATION_MS for d in anim.durations)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest test/test_animation.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'manga_translator.animation'`

- [ ] **Step 3: Write the implementation**

```python
# manga_translator/animation.py
"""Frame-level handling for animated inputs (GIF / animated WebP).

The translation pipeline runs once, on the first frame. The pixels it changed
become an RGBA overlay that is composited onto every frame. This keeps the cost
at one image regardless of frame count and keeps the translated text from
jittering, at the price of freezing frame 0's background wherever the overlay is
opaque -- invisible as long as the speech bubbles do not move.
"""
import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from PIL import Image, ImageSequence

from .utils import get_logger

logger = get_logger('animation')

# PIL reports no duration for the first frame of some encoders.
DEFAULT_FRAME_DURATION_MS = 100


@dataclass
class Animation:
    frames: List[Image.Image]   # RGBA, fully composited canvases
    durations: List[int]        # milliseconds per frame
    loop: int                   # 0 == infinite


def load_animation(path: str) -> Optional[Animation]:
    """Return an Animation, or None when `path` is not an animated image."""
    with Image.open(path) as img:
        if not getattr(img, 'is_animated', False):
            return None
        frames, durations = [], []
        for frame in ImageSequence.Iterator(img):
            frames.append(frame.convert('RGBA'))
            duration = frame.info.get('duration')
            durations.append(int(duration) if duration else DEFAULT_FRAME_DURATION_MS)
        loop = img.info.get('loop', 0)
    logger.info(f'Animated input: {len(frames)} frames, loop={loop}')
    return Animation(frames=frames, durations=durations, loop=int(loop))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest test/test_animation.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add manga_translator/animation.py test/test_animation.py
git commit -m "Add animation frame loading for GIF/WebP"
```

---

### Task 2: Overlay construction

**Files:**
- Modify: `manga_translator/animation.py`
- Test: `test/test_animation.py`

**Interfaces:**
- Consumes: `Animation` from Task 1
- Produces: `build_overlay(src_rgb: np.ndarray, rendered_rgb: np.ndarray, size: tuple[int, int]) -> Image.Image` returning RGBA; `overlay_coverage(overlay: Image.Image) -> float` returning the opaque fraction in `[0, 1]`

- [ ] **Step 1: Write the failing tests**

```python
# append to test/test_animation.py

def test_build_overlay_is_transparent_where_unchanged():
    src = np.zeros((20, 20, 3), dtype=np.uint8)
    rendered = src.copy()
    rendered[5:10, 5:10] = 255           # pipeline changed this block only
    overlay = build_overlay(src, rendered, (20, 20))
    alpha = np.array(overlay)[:, :, 3]
    assert alpha[0, 0] == 0
    assert alpha[7, 7] == 255
    assert (alpha == 255).sum() == 25


def test_build_overlay_carries_rendered_pixels():
    src = np.zeros((10, 10, 3), dtype=np.uint8)
    rendered = src.copy()
    rendered[2:4, 2:4] = (10, 200, 30)
    overlay = build_overlay(src, rendered, (10, 10))
    rgb = np.array(overlay)[:, :, :3]
    assert tuple(rgb[3, 3]) == (10, 200, 30)


def test_build_overlay_resizes_to_requested_size():
    src = np.zeros((40, 40, 3), dtype=np.uint8)
    rendered = src.copy()
    rendered[10:20, 10:20] = 255
    overlay = build_overlay(src, rendered, (20, 20))
    assert overlay.size == (20, 20)


def test_build_overlay_handles_rendered_larger_than_source():
    """Upscaling leaves img_rendered bigger than the frames."""
    src = np.zeros((20, 20, 3), dtype=np.uint8)
    rendered = np.zeros((40, 40, 3), dtype=np.uint8)
    rendered[10:30, 10:30] = 255
    overlay = build_overlay(src, rendered, (20, 20))
    assert overlay.size == (20, 20)
    assert np.array(overlay)[:, :, 3].max() == 255


def test_overlay_coverage_reports_opaque_fraction():
    src = np.zeros((10, 10, 3), dtype=np.uint8)
    rendered = src.copy()
    rendered[0:5, :] = 255               # half the pixels
    overlay = build_overlay(src, rendered, (10, 10))
    assert overlay_coverage(overlay) == pytest.approx(0.5, abs=0.01)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest test/test_animation.py -k overlay -v`
Expected: FAIL with `ImportError: cannot import name 'build_overlay'`

- [ ] **Step 3: Write the implementation**

```python
# append to manga_translator/animation.py

def build_overlay(src_rgb: np.ndarray, rendered_rgb: np.ndarray, size) -> Image.Image:
    """RGBA layer holding every pixel the pipeline changed.

    Opaque where `rendered_rgb` differs from `src_rgb`, transparent elsewhere.
    Captures both halves of the edit -- the erased source text and the drawn
    translation -- without needing to know where glyphs were placed.
    """
    if rendered_rgb.shape[:2] != src_rgb.shape[:2]:
        # Upscaling was enabled; bring the render back to the source grid.
        rendered_rgb = np.array(
            Image.fromarray(rendered_rgb.astype(np.uint8)).resize(
                (src_rgb.shape[1], src_rgb.shape[0]), Image.LANCZOS
            )
        )

    src = src_rgb.astype(np.int16)
    rendered = rendered_rgb.astype(np.int16)
    changed = np.abs(src - rendered).max(axis=2) > 0
    alpha = (changed * 255).astype(np.uint8)

    overlay = Image.fromarray(
        np.dstack([rendered_rgb.astype(np.uint8), alpha]), 'RGBA'
    )
    if overlay.size != tuple(size):
        overlay = overlay.resize(tuple(size), Image.LANCZOS)
    return overlay


def overlay_coverage(overlay: Image.Image) -> float:
    """Fraction of the overlay that is opaque, in [0, 1]."""
    alpha = np.array(overlay)[:, :, 3]
    return float((alpha > 0).mean())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest test/test_animation.py -v`
Expected: PASS (9 tests)

- [ ] **Step 5: Commit**

```bash
git add manga_translator/animation.py test/test_animation.py
git commit -m "Add difference-based text overlay construction"
```

---

### Task 3: Overlay compositing

**Files:**
- Modify: `manga_translator/animation.py`
- Test: `test/test_animation.py`

**Interfaces:**
- Consumes: `Animation`, `build_overlay` from Tasks 1-2
- Produces: `apply_overlay(anim: Animation, overlay: Image.Image) -> Animation`

- [ ] **Step 1: Write the failing tests**

```python
# append to test/test_animation.py

def test_apply_overlay_preserves_timing_metadata(tmp_path):
    path = _write_anim(tmp_path / 'a.webp', _make_frames(4), duration=40, loop=0)
    anim = load_animation(path)
    src = np.zeros((30, 40, 3), dtype=np.uint8)
    rendered = src.copy()
    rendered[0:5, 0:5] = 255
    overlay = build_overlay(src, rendered, anim.frames[0].size)

    out = apply_overlay(anim, overlay)
    assert len(out.frames) == 4
    assert out.durations == anim.durations
    assert out.loop == anim.loop


def test_apply_overlay_writes_overlay_onto_every_frame(tmp_path):
    path = _write_anim(tmp_path / 'a.webp', _make_frames(4), duration=40)
    anim = load_animation(path)
    src = np.zeros((30, 40, 3), dtype=np.uint8)
    rendered = src.copy()
    rendered[0:5, 0:5] = (255, 0, 255)
    overlay = build_overlay(src, rendered, anim.frames[0].size)

    out = apply_overlay(anim, overlay)
    for frame in out.frames:
        assert tuple(np.array(frame)[2, 2][:3]) == (255, 0, 255)


def test_apply_overlay_leaves_untouched_pixels_animating(tmp_path):
    """Outside the overlay the frames must still differ from each other."""
    path = _write_anim(tmp_path / 'a.webp', _make_frames(4), duration=40)
    anim = load_animation(path)
    src = np.zeros((30, 40, 3), dtype=np.uint8)
    rendered = src.copy()
    rendered[0:5, 0:5] = 255
    overlay = build_overlay(src, rendered, anim.frames[0].size)

    out = apply_overlay(anim, overlay)
    corners = [tuple(np.array(f)[20, 30][:3]) for f in out.frames]
    assert len(set(corners)) > 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest test/test_animation.py -k apply_overlay -v`
Expected: FAIL with `ImportError: cannot import name 'apply_overlay'`

- [ ] **Step 3: Write the implementation**

```python
# append to manga_translator/animation.py

def apply_overlay(anim: Animation, overlay: Image.Image) -> Animation:
    """Composite `overlay` onto every frame, returning a new Animation."""
    frames = []
    for frame in anim.frames:
        base = frame.convert('RGBA')
        layer = overlay if overlay.size == base.size else overlay.resize(base.size, Image.LANCZOS)
        base.alpha_composite(layer)
        frames.append(base)
    return Animation(frames=frames, durations=list(anim.durations), loop=anim.loop)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest test/test_animation.py -v`
Expected: PASS (12 tests)

- [ ] **Step 5: Commit**

```bash
git add manga_translator/animation.py test/test_animation.py
git commit -m "Composite text overlay onto every animation frame"
```

---

### Task 4: GIF and animated WebP encoders

**Files:**
- Modify: `manga_translator/save.py`
- Test: `test/test_save_animation.py`

**Interfaces:**
- Consumes: `DEFAULT_FRAME_DURATION_MS` from Task 1
- Produces: `GifFormat` registered for `gif`; `ImageFormat` writing animated WebP when `ctx.anim_frames` holds more than one frame

Animation data is read off `ctx`: `ctx.anim_frames`, `ctx.anim_durations`, `ctx.anim_loop`. `Context.__getattr__` returns `None` for unset keys, so no `getattr` default is needed.

- [ ] **Step 1: Write the failing tests**

```python
# test/test_save_animation.py
import numpy as np
from PIL import Image

from manga_translator.save import save_result, OUTPUT_FORMATS
from manga_translator.utils import Context


def _frames(n=5, size=(40, 30)):
    out = []
    for i in range(n):
        arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        arr[:, :, 1] = (i * 50) % 256
        out.append(Image.fromarray(arr, 'RGB').convert('RGBA'))
    return out


def _anim_ctx(frames, duration=40, loop=0):
    ctx = Context()
    ctx.anim_frames = frames
    ctx.anim_durations = [duration] * len(frames)
    ctx.anim_loop = loop
    ctx.save_quality = 100
    return ctx


def test_gif_is_registered():
    assert 'gif' in OUTPUT_FORMATS


def test_gif_roundtrip_preserves_frames_and_timing(tmp_path):
    frames = _frames(5)
    dest = str(tmp_path / 'out.gif')
    save_result(frames[0], dest, _anim_ctx(frames, duration=40, loop=0))

    with Image.open(dest) as im:
        assert im.n_frames == 5
        assert im.info.get('loop') == 0
        durations = []
        for i in range(im.n_frames):
            im.seek(i)
            durations.append(im.info.get('duration'))
    assert durations == [40] * 5


def test_gif_uses_one_shared_palette(tmp_path):
    frames = _frames(5)
    dest = str(tmp_path / 'shared.gif')
    save_result(frames[0], dest, _anim_ctx(frames))

    palettes = []
    with Image.open(dest) as im:
        for i in range(im.n_frames):
            im.seek(i)
            palettes.append(tuple(im.getpalette() or ()))
    assert len(set(palettes)) == 1


def test_static_gif_still_saves(tmp_path):
    dest = str(tmp_path / 'static.gif')
    ctx = Context()
    ctx.save_quality = 100
    save_result(Image.new('RGB', (10, 10), 'red'), dest, ctx)
    with Image.open(dest) as im:
        assert im.size == (10, 10)


def test_animated_webp_roundtrip(tmp_path):
    frames = _frames(6)
    dest = str(tmp_path / 'out.webp')
    save_result(frames[0], dest, _anim_ctx(frames, duration=40, loop=0))

    with Image.open(dest) as im:
        assert im.is_animated
        assert im.n_frames == 6


def test_png_with_animation_writes_single_frame(tmp_path):
    frames = _frames(4)
    dest = str(tmp_path / 'out.png')
    save_result(frames[0], dest, _anim_ctx(frames))
    with Image.open(dest) as im:
        assert getattr(im, 'n_frames', 1) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest test/test_save_animation.py -v`
Expected: FAIL — `test_gif_is_registered` fails because `gif` is not in `OUTPUT_FORMATS`

- [ ] **Step 3: Write the implementation**

Replace the `ImageFormat` class in `manga_translator/save.py` and add `GifFormat` after it. Add the imports at the top of the file.

```python
# manga_translator/save.py -- imports
import math
import os

import numpy as np
from PIL import Image
from abc import abstractmethod
from .rendering.gimp_render import gimp_render

from .animation import DEFAULT_FRAME_DURATION_MS
from .utils import Context, get_logger

logger = get_logger('save')

# Palette selection does not need every frame or full resolution.
GIF_PALETTE_SAMPLE_FRAMES = 32
GIF_PALETTE_SAMPLE_WIDTH = 200
GIF_SIZE_WARN_BYTES = 20 * 1024 * 1024
```

```python
# manga_translator/save.py -- replaces the existing ImageFormat
class ImageFormat(ExportFormat):
    SUPPORTED_FORMATS = ['png', 'webp']

    def _save(self, result: Image.Image, dest: str, ctx: Context):
        frames = ctx.anim_frames
        if frames and len(frames) > 1:
            if dest.lower().endswith('.webp'):
                frames[0].save(
                    dest, save_all=True, append_images=frames[1:],
                    duration=ctx.anim_durations, loop=ctx.anim_loop, lossless=False,
                )
                logger.info(f'Saved {len(frames)} animated frames to "{dest}"')
                return
            logger.warning(
                f'"{os.path.basename(dest)}" is not an animated format; '
                f'wrote the first of {len(frames)} frames. Use --format gif or webp '
                f'to keep the animation.'
            )
        result.save(dest)


class GifFormat(ExportFormat):
    SUPPORTED_FORMATS = ['gif']

    def _save(self, result: Image.Image, dest: str, ctx: Context):
        frames = ctx.anim_frames
        if frames and len(frames) > 1:
            durations = ctx.anim_durations
            loop = ctx.anim_loop if ctx.anim_loop is not None else 0
        else:
            frames = [result]
            durations = [DEFAULT_FRAME_DURATION_MS]
            loop = 0

        flat = [_flatten_to_white(f) for f in frames]
        palette = _shared_palette(flat)
        quantized = [
            f.quantize(palette=palette, dither=Image.Dither.FLOYDSTEINBERG)
            for f in flat
        ]
        quantized[0].save(
            dest, save_all=True, append_images=quantized[1:],
            duration=durations, loop=loop, disposal=2, optimize=False,
        )

        size = os.path.getsize(dest)
        logger.info(f'Saved {len(frames)} GIF frames to "{dest}" ({size / 1e6:.1f} MB)')
        if size > GIF_SIZE_WARN_BYTES:
            logger.warning(
                f'GIF is {size / 1e6:.1f} MB. --format webp produces a much smaller '
                f'file at full colour depth, if your viewer supports animated WebP.'
            )


Both non-animated raster formats warn the same way, so factor the message out and
call it from `JPGFormat._save` too:

```python
def _warn_animation_dropped(dest: str, frames):
    if frames and len(frames) > 1:
        logger.warning(
            f'"{os.path.basename(dest)}" is not an animated format; '
            f'wrote the first of {len(frames)} frames. Use --format gif or webp '
            f'to keep the animation.'
        )


class JPGFormat(ExportFormat):
    SUPPORTED_FORMATS = ['jpg', 'jpeg']

    def _save(self, result: Image.Image, dest: str, ctx: Context):
        _warn_animation_dropped(dest, ctx.anim_frames)
        result = result.convert('RGB')
        # Certain versions of PIL only support JPEG but not JPG
        result.save(dest, quality=ctx.save_quality, format='JPEG')
```

`ImageFormat`'s inline warning above is replaced by a `_warn_animation_dropped(dest, frames)`
call so the two paths cannot drift apart.

```python
def _flatten_to_white(frame: Image.Image) -> Image.Image:
    """GIF has no partial transparency; match load_image()'s white background."""
    if frame.mode != 'RGBA':
        return frame.convert('RGB')
    background = Image.new('RGB', frame.size, (255, 255, 255))
    background.paste(frame, mask=frame.split()[3])
    return background


def _shared_palette(frames) -> Image.Image:
    """One palette for the whole animation.

    Per-frame quantisation gives each frame its own palette, which shows up as
    colour flicker during playback -- more objectionable than 256-colour banding.
    """
    stride = max(1, math.ceil(len(frames) / GIF_PALETTE_SAMPLE_FRAMES))
    sample = frames[::stride]

    scale = min(1.0, GIF_PALETTE_SAMPLE_WIDTH / sample[0].width)
    tile_size = (max(1, int(sample[0].width * scale)), max(1, int(sample[0].height * scale)))

    strip = Image.new('RGB', (tile_size[0], tile_size[1] * len(sample)))
    for i, frame in enumerate(sample):
        strip.paste(frame.resize(tile_size, Image.BILINEAR), (0, i * tile_size[1]))

    # 255 colours, leaving one palette index free for transparency.
    return strip.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest test/test_save_animation.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add manga_translator/save.py test/test_save_animation.py
git commit -m "Add GIF encoder and animated WebP output"
```

---

### Task 5: Wire animation into the local translation path

**Files:**
- Modify: `manga_translator/mode/local.py` (imports; `_translate_file` around line 254; batch collector around line 344-372)
- Test: `test/test_animation_pipeline.py`

**Interfaces:**
- Consumes: `load_animation`, `build_overlay`, `apply_overlay`, `overlay_coverage` from Tasks 1-3
- Produces: `MangaTranslatorLocal._attach_animation(ctx, path) -> bool` — sets `ctx.anim_frames` / `ctx.anim_durations` / `ctx.anim_loop` and returns whether an animation was attached

Split out as its own method so it can be tested without running the translation pipeline.

- [ ] **Step 1: Write the failing test**

```python
# test/test_animation_pipeline.py
import numpy as np
from PIL import Image

from manga_translator.animation import load_animation
from manga_translator.mode.local import MangaTranslatorLocal
from manga_translator.utils import Context


def _write_anim(path, n=5, size=(40, 30)):
    frames = []
    for i in range(n):
        arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        arr[:, :, 2] = (i * 50) % 256
        frames.append(Image.fromarray(arr, 'RGB').convert('RGBA'))
    frames[0].save(
        path, format='WEBP', save_all=True, append_images=frames[1:],
        duration=40, loop=0, lossless=True,
    )
    return str(path)


def test_attach_animation_returns_false_for_static(tmp_path):
    p = tmp_path / 's.png'
    Image.new('RGB', (10, 10), 'white').save(p)
    translator = MangaTranslatorLocal({})
    ctx = Context()
    assert translator._attach_animation(ctx, str(p)) is False
    assert ctx.anim_frames is None


def test_attach_animation_populates_ctx(tmp_path):
    path = _write_anim(tmp_path / 'a.webp', n=5)
    anim = load_animation(path)

    src = np.zeros((30, 40, 3), dtype=np.uint8)
    rendered = src.copy()
    rendered[0:6, 0:6] = (255, 0, 255)

    translator = MangaTranslatorLocal({})
    ctx = Context()
    ctx.img_rgb = src
    ctx.img_rendered = rendered

    assert translator._attach_animation(ctx, path) is True
    assert len(ctx.anim_frames) == 5
    assert ctx.anim_durations == [40] * 5
    assert ctx.anim_loop == 0
    for frame in ctx.anim_frames:
        assert tuple(np.array(frame)[2, 2][:3]) == (255, 0, 255)


def test_attach_animation_without_render_keeps_frames_unchanged(tmp_path):
    """No text found -> pipeline never set img_rendered; still emit the animation."""
    path = _write_anim(tmp_path / 'a.webp', n=4)
    translator = MangaTranslatorLocal({})
    ctx = Context()
    assert translator._attach_animation(ctx, path) is True
    assert len(ctx.anim_frames) == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest test/test_animation_pipeline.py -v`
Expected: FAIL with `AttributeError: 'MangaTranslatorLocal' object has no attribute '_attach_animation'`

- [ ] **Step 3: Write the implementation**

Add the import near the other `manga_translator` imports at the top of `local.py`:

```python
from ..animation import load_animation, build_overlay, apply_overlay, overlay_coverage
```

Add the method to `MangaTranslatorLocal`:

```python
    # Fraction of changed pixels above which the overlay effectively freezes the
    # animation -- happens when the colorizer rewrites the whole frame.
    ANIM_FREEZE_WARN_COVERAGE = 0.5

    def _attach_animation(self, ctx: Context, path: str) -> bool:
        """Attach translated animation frames to `ctx`. False if `path` is static."""
        anim = load_animation(path)
        if anim is None:
            return False

        if ctx.img_rgb is not None and ctx.img_rendered is not None:
            overlay = build_overlay(ctx.img_rgb, ctx.img_rendered, anim.frames[0].size)
            coverage = overlay_coverage(overlay)
            if coverage > self.ANIM_FREEZE_WARN_COVERAGE:
                logger.warning(
                    f'The translation changed {coverage:.0%} of the frame, so most of '
                    f'the animation will be frozen to the first frame. This usually '
                    f'means the colorizer or another whole-image filter is enabled.'
                )
            anim = apply_overlay(anim, overlay)
        else:
            logger.info('No rendered text for this animation; re-encoding frames as-is.')

        ctx.anim_frames = anim.frames
        ctx.anim_durations = anim.durations
        ctx.anim_loop = anim.loop
        return True
```

Then call it in `_translate_file`, immediately after `ctx = await self.translate(img, config)` and before the `if result:` save block:

```python
            ctx = await self.translate(img, config)
            result = ctx.result

            self._attach_animation(ctx, path)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest test/test_animation_pipeline.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Route animated files out of the batch collector**

In `_translate_folder_batch`, the collector at `local.py:344-372` builds `image_tasks`. Batch mode holds many decoded images in memory and has no per-file animation hook, so animated files are handled individually afterwards. Collect them into a separate list:

```python
                # 尝试加载图片
                try:
                    img = Image.open(file_path)
                    img.verify()
                    img = Image.open(file_path)  # 重新打开因为verify会关闭文件
                    if getattr(img, 'is_animated', False):
                        # Animated inputs need per-file frame handling; batching
                        # them would hold every frame of every file in memory.
                        animated_tasks.append((file_path, output_dest))
                        continue
                    image_tasks.append((img, config, file_path, output_dest))
                except Exception as e:
                    logger.warning(f'Failed to open image: {file_path}, error: {e}')
                    continue
```

Declare `animated_tasks = []` next to `image_tasks = []`, and after the batch loop finishes, process them:

```python
        for file_path, output_dest in animated_tasks:
            logger.info(f'Translating animated file separately: "{file_path}"')
            try:
                if await self.translate_file(file_path, output_dest, params, config):
                    translated_count += 1
            except Exception as e:
                logger.error(e)
                if not self.ignore_errors:
                    raise
```

- [ ] **Step 6: Verify the whole suite still passes**

Run: `.venv/Scripts/python.exe -m pytest test/test_animation.py test/test_save_animation.py test/test_animation_pipeline.py -v`
Expected: PASS (21 tests)

- [ ] **Step 7: Commit**

```bash
git add manga_translator/mode/local.py test/test_animation_pipeline.py
git commit -m "Wire animation handling into the local translation path"
```

---

### Task 6: End-to-end QA against a synthetic animated fixture

**Files:**
- Create: `test/fixtures/make_animated_fixture.py`
- Test: manual CLI run

The fixture is generated rather than committed as a binary, and gives precise control over the static-bubble case the design targets.

- [ ] **Step 1: Write the fixture generator**

```python
# test/fixtures/make_animated_fixture.py
"""Generate synthetic animated fixtures for manual QA.

Usage: python test/fixtures/make_animated_fixture.py <output-dir>
"""
import sys
from pathlib import Path

from PIL import Image, ImageDraw

W, H, N = 400, 560, 24


def _frame(i, bubble_xy):
    """Moving gradient background plus a white speech bubble with source text."""
    img = Image.new('RGB', (W, H))
    draw = ImageDraw.Draw(img)
    for y in range(H):
        shift = (i * 10) % 256
        draw.line([(0, y), (W, y)], fill=((y + shift) % 256, (y * 2 + shift) % 256, 200))

    bx, by = bubble_xy
    draw.ellipse([bx, by, bx + 240, by + 120], fill='white', outline='black', width=3)
    draw.text((bx + 40, by + 40), 'こんにちは\nげんきですか', fill='black')
    return img


def build(path: Path, moving_bubble: bool):
    frames = []
    for i in range(N):
        xy = (30 + (i * 4 if moving_bubble else 0), 30)
        frames.append(_frame(i, xy))
    frames[0].save(
        path, format='WEBP', save_all=True, append_images=frames[1:],
        duration=40, loop=0, lossless=True,
    )
    print(f'wrote {path} ({N} frames)')


if __name__ == '__main__':
    out = Path(sys.argv[1] if len(sys.argv) > 1 else '.')
    out.mkdir(parents=True, exist_ok=True)
    build(out / 'anim_static_bubble.webp', moving_bubble=False)
    build(out / 'anim_moving_bubble.webp', moving_bubble=True)
```

- [ ] **Step 2: Generate the fixtures**

```bash
.venv/Scripts/python.exe test/fixtures/make_animated_fixture.py <scratchdir>/qa_in
```

- [ ] **Step 3: Run the CLI for GIF output**

```bash
.venv/Scripts/python.exe -m manga_translator local \
  -i <scratchdir>/qa_in/anim_static_bubble.webp \
  -o <scratchdir>/qa_out --format gif --overwrite -v
```

Expected: completes; a `.gif` is written.

- [ ] **Step 4: Run the CLI for animated WebP output**

```bash
.venv/Scripts/python.exe -m manga_translator local \
  -i <scratchdir>/qa_in/anim_static_bubble.webp \
  -o <scratchdir>/qa_out --format webp --overwrite -v
```

Expected: completes; a `.webp` is written.

- [ ] **Step 5: Verify both outputs**

```python
from PIL import Image
for p in ['out.gif', 'out.webp']:
    with Image.open(p) as im:
        print(p, im.format, im.n_frames, im.info.get('loop'))
        assert im.n_frames == 24
```

Also confirm the translated text is present on the last frame, not just the first, and that the background outside the bubble still differs between frames.

- [ ] **Step 6: Confirm static images are unaffected**

Translate any single PNG and confirm the output matches pre-change behaviour.

- [ ] **Step 7: Commit**

```bash
git add test/fixtures/make_animated_fixture.py
git commit -m "Add synthetic animated fixtures for QA"
```
