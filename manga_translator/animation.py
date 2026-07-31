"""Frame-level handling for animated inputs (GIF / animated WebP).

The translation pipeline runs once, on the first frame. The pixels it changed
become an RGBA overlay that is composited onto every frame. This keeps the cost
at one image regardless of frame count and keeps the translated text from
jittering, at the price of freezing frame 0's background wherever the overlay is
opaque -- invisible as long as the speech bubbles do not move.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple

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


def is_animated_path(path: str) -> bool:
    """Whether `path` holds more than one frame, without decoding them all.

    Used to pick the output extension before translation starts, so a mixed
    folder can send animations to GIF while static pages keep their format.
    """
    try:
        with Image.open(path) as img:
            return bool(getattr(img, 'is_animated', False))
    except Exception:
        return False


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


def build_overlay(src_rgb: np.ndarray, rendered_rgb: np.ndarray,
                  size: Tuple[int, int]) -> Image.Image:
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


def apply_overlay(anim: Animation, overlay: Image.Image) -> Animation:
    """Composite `overlay` onto every frame, returning a new Animation."""
    frames = []
    for frame in anim.frames:
        base = frame.convert('RGBA')
        layer = overlay if overlay.size == base.size else overlay.resize(base.size, Image.LANCZOS)
        base.alpha_composite(layer)
        frames.append(base)
    return Animation(frames=frames, durations=list(anim.durations), loop=anim.loop)
