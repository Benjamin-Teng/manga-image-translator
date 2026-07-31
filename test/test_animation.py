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


def test_is_animated_path_detects_animation(tmp_path):
    from manga_translator.animation import is_animated_path
    anim = _write_anim(tmp_path / 'a.webp', _make_frames(3))
    static = tmp_path / 's.png'
    Image.new('RGB', (10, 10), 'white').save(static)
    assert is_animated_path(anim) is True
    assert is_animated_path(str(static)) is False


def test_is_animated_path_is_false_for_unreadable_file(tmp_path):
    bad = tmp_path / 'notanimage.txt'
    bad.write_text('hello')
    from manga_translator.animation import is_animated_path
    assert is_animated_path(str(bad)) is False
