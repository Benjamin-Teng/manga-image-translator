"""Generate synthetic animated fixtures for manual QA.

Usage: python test/fixtures/make_animated_fixture.py <output-dir>

Two fixtures, matching the two cases the design distinguishes:
  anim_static_bubble.webp -- moving artwork, stationary bubble (the supported case)
  anim_moving_bubble.webp -- bubble travels across frames (documents the limitation)
"""
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H, N = 400, 560, 24
TEXT = 'こんにちは\nげんきですか'


def _font():
    """A CJK-capable font if one is installed, else PIL's default."""
    for candidate in (
        r'C:\Windows\Fonts\msgothic.ttc',
        r'C:\Windows\Fonts\meiryo.ttc',
        r'C:\Windows\Fonts\YuGothM.ttc',
        r'C:\Windows\Fonts\simsun.ttc',
    ):
        if Path(candidate).exists():
            try:
                return ImageFont.truetype(candidate, 22)
            except OSError:
                continue
    return ImageFont.load_default()


def _frame(i, bubble_xy, font):
    """Moving gradient background plus a white speech bubble with source text."""
    img = Image.new('RGB', (W, H))
    draw = ImageDraw.Draw(img)
    for y in range(H):
        shift = (i * 10) % 256
        draw.line([(0, y), (W, y)], fill=((y + shift) % 256, (y * 2 + shift) % 256, 200))

    bx, by = bubble_xy
    draw.ellipse([bx, by, bx + 260, by + 140], fill='white', outline='black', width=3)
    draw.multiline_text((bx + 45, by + 45), TEXT, fill='black', font=font, spacing=8)
    return img


def build(path: Path, moving_bubble: bool):
    font = _font()
    frames = [
        _frame(i, (30 + (i * 4 if moving_bubble else 0), 30), font)
        for i in range(N)
    ]
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
