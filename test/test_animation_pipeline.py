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
    translator = MangaTranslatorLocal({'kernel_size': 3})
    ctx = Context()
    assert translator._attach_animation(ctx, str(p)) is False
    assert ctx.anim_frames is None


def test_attach_animation_populates_ctx(tmp_path):
    path = _write_anim(tmp_path / 'a.webp', n=5)

    src = np.zeros((30, 40, 3), dtype=np.uint8)
    rendered = src.copy()
    rendered[0:6, 0:6] = (255, 0, 255)

    translator = MangaTranslatorLocal({'kernel_size': 3})
    ctx = Context()
    ctx.img_rgb = src
    ctx.img_rendered = rendered

    assert translator._attach_animation(ctx, path) is True
    assert len(ctx.anim_frames) == 5
    assert ctx.anim_durations == [40] * 5
    assert ctx.anim_loop == 0
    for frame in ctx.anim_frames:
        assert tuple(np.array(frame)[2, 2][:3]) == (255, 0, 255)


def test_attach_animation_leaves_background_animating(tmp_path):
    path = _write_anim(tmp_path / 'a.webp', n=5)

    src = np.zeros((30, 40, 3), dtype=np.uint8)
    rendered = src.copy()
    rendered[0:6, 0:6] = (255, 0, 255)

    translator = MangaTranslatorLocal({'kernel_size': 3})
    ctx = Context()
    ctx.img_rgb = src
    ctx.img_rendered = rendered
    translator._attach_animation(ctx, path)

    corners = [tuple(np.array(f)[25, 35][:3]) for f in ctx.anim_frames]
    assert len(set(corners)) > 1


def test_attach_animation_without_render_keeps_frames_unchanged(tmp_path):
    """No text found -> pipeline never set img_rendered; still emit the animation."""
    path = _write_anim(tmp_path / 'a.webp', n=4)
    translator = MangaTranslatorLocal({'kernel_size': 3})
    ctx = Context()
    assert translator._attach_animation(ctx, path) is True
    assert len(ctx.anim_frames) == 4


def test_output_ext_prefers_anim_format_for_animated_input(tmp_path):
    path = _write_anim(tmp_path / 'a.webp', n=3)
    translator = MangaTranslatorLocal({'kernel_size': 3, 'anim_format': 'gif'})
    assert translator._output_ext(path, None, 'webp') == 'gif'


def test_output_ext_leaves_static_input_alone(tmp_path):
    static = tmp_path / 's.png'
    Image.new('RGB', (10, 10), 'white').save(static)
    translator = MangaTranslatorLocal({'kernel_size': 3, 'anim_format': 'gif'})
    assert translator._output_ext(str(static), None, 'png') == 'png'


def test_output_ext_without_anim_format_keeps_source_ext(tmp_path):
    path = _write_anim(tmp_path / 'a.webp', n=3)
    translator = MangaTranslatorLocal({'kernel_size': 3})
    assert translator._output_ext(path, None, 'webp') == 'webp'


def test_explicit_format_still_wins_for_static(tmp_path):
    static = tmp_path / 's.png'
    Image.new('RGB', (10, 10), 'white').save(static)
    translator = MangaTranslatorLocal({'kernel_size': 3, 'anim_format': 'gif'})
    assert translator._output_ext(str(static), 'jpg', 'png') == 'jpg'
