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


def _count_colour_tables(path):
    """Walk the GIF blocks, returning (image descriptors, ones with a local table).

    Checked at the byte level because Pillow reports a frame's palette
    inconsistently after seek(), while the property that actually matters -- one
    global colour table and no per-frame ones -- is unambiguous in the file.
    """
    data = open(path, 'rb').read()
    global_table = 2 ** ((data[10] & 0x07) + 1) if data[10] & 0x80 else 0
    i = 13 + global_table * 3
    descriptors = local_tables = 0
    while i < len(data):
        block = data[i]
        if block == 0x2C:                       # image descriptor
            descriptors += 1
            packed = data[i + 9]
            i += 10
            if packed & 0x80:
                local_tables += 1
                i += 3 * 2 ** ((packed & 0x07) + 1)
            i += 1                              # LZW minimum code size
            while i < len(data) and data[i] != 0:
                i += data[i] + 1
            i += 1
        elif block == 0x21:                     # extension
            i += 2
            while i < len(data) and data[i] != 0:
                i += data[i] + 1
            i += 1
        elif block == 0x3B:                     # trailer
            break
        else:
            i += 1
    return descriptors, local_tables


def test_gif_uses_one_shared_palette(tmp_path):
    frames = _frames(5)
    dest = str(tmp_path / 'shared.gif')
    save_result(frames[0], dest, _anim_ctx(frames))

    descriptors, local_tables = _count_colour_tables(dest)
    assert descriptors == 5
    assert local_tables == 0


def test_gif_frames_decode_back_to_their_source(tmp_path):
    """A static block must survive on every frame, not just the first."""
    frames = _frames(6)
    for frame in frames:
        pixels = np.array(frame)
        pixels[5:15, 5:20] = (255, 0, 255, 255)      # stands in for the text overlay
        frame.paste(Image.fromarray(pixels, 'RGBA'))

    dest = str(tmp_path / 'fidelity.gif')
    save_result(frames[0], dest, _anim_ctx(frames))

    with Image.open(dest) as im:
        assert im.n_frames == 6
        for i in range(im.n_frames):
            im.seek(i)
            im.load()
            assert tuple(np.array(im.convert('RGB'))[10, 10]) == (255, 0, 255)


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


def test_jpg_with_animation_writes_single_frame(tmp_path):
    frames = _frames(4)
    dest = str(tmp_path / 'out.jpg')
    save_result(frames[0], dest, _anim_ctx(frames))
    with Image.open(dest) as im:
        assert im.format == 'JPEG'
        assert getattr(im, 'n_frames', 1) == 1
