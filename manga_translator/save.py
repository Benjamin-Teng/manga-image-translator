import math
import os
from PIL import Image
from abc import abstractmethod
from .rendering.gimp_render import gimp_render

from .animation import DEFAULT_FRAME_DURATION_MS
from .utils import Context, get_logger

logger = get_logger('save')

# Palette selection needs neither every frame nor full resolution.
GIF_PALETTE_SAMPLE_FRAMES = 32
GIF_PALETTE_SAMPLE_WIDTH = 200
GIF_SIZE_WARN_BYTES = 20 * 1024 * 1024


class FormatNotSupportedException(Exception):
    def __init__(self, fmt: str):
        super().__init__(f'Format {fmt} is not supported.')

OUTPUT_FORMATS = {}
def register_format(format_cls):
    for fmt in format_cls.SUPPORTED_FORMATS:
        if fmt in OUTPUT_FORMATS:
            raise Exception(f'Tried to register multiple ExportFormats for "{fmt}"')
        OUTPUT_FORMATS[fmt] = format_cls()
    return format_cls

class ExportFormat():
    SUPPORTED_FORMATS = []

    # Subclasses will be auto registered
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        register_format(cls)

    def save(self, result: Image.Image, dest: str, ctx: Context):
        self._save(result, dest, ctx)

    @abstractmethod
    def _save(self, result: Image.Image, dest: str, ctx: Context):
        pass

def save_result(result: Image.Image, dest: str, ctx: Context):
    _, ext = os.path.splitext(dest)
    ext = ext[1:]
    if ext not in OUTPUT_FORMATS:
        raise FormatNotSupportedException(ext)

    format_handler: ExportFormat = OUTPUT_FORMATS[ext]
    format_handler.save(result, dest, ctx)


def _warn_animation_dropped(dest: str, frames):
    """Tell the user the animation was lost rather than dropping it silently."""
    if frames and len(frames) > 1:
        logger.warning(
            f'"{os.path.basename(dest)}" is not an animated format; '
            f'wrote the first of {len(frames)} frames. Use --format gif or webp '
            f'to keep the animation.'
        )


# -- Format Implementations

class ImageFormat(ExportFormat):
    SUPPORTED_FORMATS = ['png', 'webp']

    def _save(self, result: Image.Image, dest: str, ctx: Context):
        frames = ctx.anim_frames
        if frames and len(frames) > 1 and dest.lower().endswith('.webp'):
            frames[0].save(
                dest, save_all=True, append_images=frames[1:],
                duration=ctx.anim_durations, loop=ctx.anim_loop, lossless=False,
            )
            logger.info(f'Saved {len(frames)} animated frames to "{dest}"')
            return
        _warn_animation_dropped(dest, frames)
        result.save(dest)

class JPGFormat(ExportFormat):
    SUPPORTED_FORMATS = ['jpg', 'jpeg']

    def _save(self, result: Image.Image, dest: str, ctx: Context):
        _warn_animation_dropped(dest, ctx.anim_frames)
        result = result.convert('RGB')
        # Certain versions of PIL only support JPEG but not JPG
        result.save(dest, quality=ctx.save_quality, format='JPEG')

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
            duration=durations, loop=loop, disposal=1, optimize=False,
            # Without an explicit palette PIL emits a local colour table per
            # frame even when every frame was quantised to the same palette,
            # which is exactly the flicker the shared palette exists to avoid.
            palette=bytes(palette.getpalette()),
        )

        size = os.path.getsize(dest)
        logger.info(f'Saved {len(frames)} GIF frame(s) to "{dest}" ({size / 1e6:.1f} MB)')
        if size > GIF_SIZE_WARN_BYTES:
            logger.warning(
                f'GIF is {size / 1e6:.1f} MB. --format webp produces a much smaller '
                f'file at full colour depth, if your viewer supports animated WebP.'
            )

class GIMPFormat(ExportFormat):
    SUPPORTED_FORMATS = ['xcf', 'psd', 'pdf']

    def _save(self, result: Image.Image, dest: str, ctx: Context):
        gimp_render(dest, ctx)


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
    tile_size = (max(1, int(sample[0].width * scale)),
                 max(1, int(sample[0].height * scale)))

    strip = Image.new('RGB', (tile_size[0], tile_size[1] * len(sample)))
    for i, frame in enumerate(sample):
        strip.paste(frame.resize(tile_size, Image.BILINEAR), (0, i * tile_size[1]))

    # 255 colours, leaving one palette index free for transparency.
    return strip.quantize(colors=255, method=Image.Quantize.MEDIANCUT)

# class KraFormat(ExportFormat):
#     SUPPORTED_FORMATS = ['kra']

#     def _save(self, result: Image.Image, dest: str, ctx: Context):
#         ...

# class SvgFormat(TranslationExportFormat):
#     SUPPORTED_FORMATS = ['svg']
