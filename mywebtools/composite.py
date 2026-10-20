"""Picture-in-picture compositing of a talking avatar onto a main video.

All geometry is computed in Python from probed sizes, so the ffmpeg graph only
contains constants (fast, and works on Ubuntu 20.04's ffmpeg 4.2). The avatar
mask, border and shadow are drawn once with Pillow and fed to ffmpeg as looped
still images; the reference code's per-pixel ``geq`` mask was far too slow on
an ARM board.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter

SHAPES = ["circle", "rounded", "rect", "ellipse", "squircle", "hexagon", "diamond", "heart"]
ANCHORS = ["top_left", "top_center", "top_right", "middle_left", "center", "middle_right",
           "bottom_left", "bottom_center", "bottom_right"]
CANVASES = {
    "shorts": (1080, 1920),
    "shorts_720": (720, 1280),
    "portrait_45": (1080, 1350),
    "square": (1080, 1080),
    "landscape": (1920, 1080),
    "landscape_720": (1280, 720),
}
FITS = ["auto", "blur", "cover", "color"]

# Areas covered by the YouTube Shorts player UI on a 9:16 canvas (fractions).
SHORTS_SAFE = {"top": 0.08, "bottom": 0.21, "left": 0.03, "right": 0.14}


def even(value: float) -> int:
    return max(2, int(round(value / 2.0)) * 2)


def hex_color(value: str, default: str = "000000") -> str:
    value = (value or "").lstrip("#")
    return value.lower() if len(value) == 6 and all(c in "0123456789abcdefABCDEF" for c in value) else default


@dataclass
class Layout:
    """User-facing PiP settings (mirrors the avatar tool form)."""

    canvas: str = "shorts"
    canvas_w: int = 1080
    canvas_h: int = 1920
    fit: str = "auto"
    bg_color: str = "#000000"
    blur: int = 20
    main_align: str = "center"  # center | top | bottom (where letterboxed video sits)
    size: int = 360
    size_h: int = 0
    anchor: str = "bottom_left"
    margin_x: int = 40
    margin_y: int = 40
    safe_area: bool = True
    crop_mode: str = "auto"  # auto | full | custom
    crop_cx: float = 0.5
    crop_cy: float = 0.39
    crop_zoom: float = 1.0
    mirror: bool = False
    shape: str = "circle"
    radius: int = 40
    feather: int = 2
    border_w: int = 6
    border_color: str = "#ffffff"
    shadow: bool = True
    shadow_blur: int = 18
    shadow_opacity: int = 45
    opacity: int = 100
    chroma: bool = False
    key_color: str = "#00ff00"
    key_similarity: float = 0.30
    key_blend: float = 0.08
    fade_in: float = 0.3
    fade_out: float = 0.3


@dataclass
class Geometry:
    canvas_w: int
    canvas_h: int
    fit: str
    main_scale: tuple[int, int]
    main_pos: tuple[int, int]
    crop: tuple[int, int, int, int]  # w, h, x, y in avatar source pixels
    box: tuple[int, int]  # avatar size on canvas
    pos: tuple[int, int]  # avatar top-left on canvas
    shadow_pad: int = 0
    notes: list[str] = field(default_factory=list)


def canvas_size(layout: Layout, main_w: int, main_h: int) -> tuple[int, int]:
    if layout.canvas in CANVASES:
        return CANVASES[layout.canvas]
    if layout.canvas == "custom":
        return even(min(max(layout.canvas_w, 240), 3840)), even(min(max(layout.canvas_h, 240), 3840))
    return even(main_w or 1080), even(main_h or 1920)  # "source"


def resolve_fit(fit: str, main_w: int, main_h: int, cw: int, ch: int) -> str:
    if fit != "auto":
        return fit if fit in FITS else "blur"
    if not main_w or not main_h:
        return "cover"
    ratio = (main_w / main_h) / (cw / ch)
    # close enough (e.g. 9:16 iPhone clip on a 9:16 canvas) -> fill; else letterbox on blur
    return "cover" if 0.9 <= ratio <= 1.12 else "blur"


def _main_geometry(fit: str, align: str, mw: int, mh: int, cw: int, ch: int) -> tuple[tuple[int, int], tuple[int, int]]:
    mw, mh = mw or cw, mh or ch
    if fit == "cover":
        scale = max(cw / mw, ch / mh)
        sw, sh = even(mw * scale), even(mh * scale)
        sw, sh = max(sw, cw), max(sh, ch)
        return (sw, sh), ((sw - cw) // 2, (sh - ch) // 2)  # crop offset
    scale = min(cw / mw, ch / mh)
    sw, sh = min(even(mw * scale), cw), min(even(mh * scale), ch)
    x = (cw - sw) // 2
    y = {"top": 0, "bottom": ch - sh}.get(align, (ch - sh) // 2)
    return (sw, sh), (x, y)  # overlay position


def crop_rect(src_w: int, src_h: int, box_w: int, box_h: int, mode: str, cx: float, cy: float,
              zoom: float) -> tuple[int, int, int, int]:
    if mode == "full" or not src_w or not src_h:
        return src_w, src_h, 0, 0
    if mode == "auto":
        cx, zoom = 0.5, 1.0
        cy = 0.39 if src_h > src_w else 0.45  # faces sit in the upper middle of a portrait shot
    aspect = box_w / box_h
    if src_w / src_h > aspect:
        ch = src_h
        cw = ch * aspect
    else:
        cw = src_w
        ch = cw / aspect
    zoom = min(max(zoom, 1.0), 8.0)
    cw, ch = even(min(cw / zoom, src_w)), even(min(ch / zoom, src_h))
    cw, ch = min(cw, src_w - src_w % 2), min(ch, src_h - src_h % 2)
    x = int(round(min(max(cx * src_w - cw / 2, 0), src_w - cw)))
    y = int(round(min(max(cy * src_h - ch / 2, 0), src_h - ch)))
    return cw, ch, x, y


def avatar_box(layout: Layout, src_w: int, src_h: int, canvas_w: int, canvas_h: int) -> tuple[int, int]:
    w = even(min(max(layout.size, 32), canvas_w))
    if layout.size_h > 0 and layout.shape not in ("circle",):
        h = even(layout.size_h)
    elif layout.crop_mode == "full" and src_w and src_h:
        h = even(w * src_h / src_w)
    else:
        h = w
    return w, min(h, canvas_h)


def avatar_position(layout: Layout, cw: int, ch: int, w: int, h: int) -> tuple[int, int]:
    left = top = right = bottom = 0.0
    if layout.safe_area and ch > cw:
        left, right = SHORTS_SAFE["left"] * cw, SHORTS_SAFE["right"] * cw
        top, bottom = SHORTS_SAFE["top"] * ch, SHORTS_SAFE["bottom"] * ch
    vertical, _, horizontal = layout.anchor.partition("_") if "_" in layout.anchor else ("middle", "", "center")
    if layout.anchor == "center":
        vertical, horizontal = "middle", "center"
    mx, my = layout.margin_x, layout.margin_y
    x = {"left": left + mx, "right": cw - right - mx - w}.get(horizontal, (cw - w) / 2)
    y = {"top": top + my, "bottom": ch - bottom - my - h}.get(vertical, (ch - h) / 2)
    x = min(max(x, 0), cw - w)
    y = min(max(y, 0), ch - h)
    return int(round(x)), int(round(y))


def compute_geometry(layout: Layout, main_w: int, main_h: int, src_w: int, src_h: int) -> Geometry:
    cw, ch = canvas_size(layout, main_w, main_h)
    fit = resolve_fit(layout.fit, main_w, main_h, cw, ch)
    main_scale, main_pos = _main_geometry(fit, layout.main_align, main_w, main_h, cw, ch)
    box = avatar_box(layout, src_w, src_h, cw, ch)
    crop = crop_rect(src_w, src_h, box[0], box[1], layout.crop_mode, layout.crop_cx, layout.crop_cy, layout.crop_zoom)
    pos = avatar_position(layout, cw, ch, *box)
    geo = Geometry(cw, ch, fit, main_scale, main_pos, crop, box, pos,
                   shadow_pad=layout.shadow_blur * 2 if layout.shadow else 0)
    if crop[0] < box[0] * 0.75:
        geo.notes.append("avatar.upscaled")
    return geo


# ---------------------------------------------------------------------------
# Masks
# ---------------------------------------------------------------------------
def _shape_points(shape: str, x0: float, y0: float, x1: float, y1: float) -> list[tuple[float, float]]:
    cx, cy, rx, ry = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2, (y1 - y0) / 2
    if shape == "hexagon":
        return [(cx + rx * math.cos(math.radians(a)), cy + ry * math.sin(math.radians(a))) for a in range(-90, 270, 60)]
    if shape == "diamond":
        return [(cx, y0), (x1, cy), (cx, y1), (x0, cy)]
    if shape == "squircle":
        pts = []
        for i in range(240):
            t = 2 * math.pi * i / 240
            c, s = math.cos(t), math.sin(t)
            pts.append((cx + rx * math.copysign(abs(c) ** 0.5, c), cy + ry * math.copysign(abs(s) ** 0.5, s)))
        return pts
    if shape == "heart":
        raw = []
        for i in range(240):
            t = 2 * math.pi * i / 240
            raw.append((16 * math.sin(t) ** 3, -(13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t))))
        min_x, max_x = min(p[0] for p in raw), max(p[0] for p in raw)
        min_y, max_y = min(p[1] for p in raw), max(p[1] for p in raw)
        return [(x0 + (px - min_x) / (max_x - min_x) * (x1 - x0), y0 + (py - min_y) / (max_y - min_y) * (y1 - y0))
                for px, py in raw]
    raise ValueError(shape)


def draw_shape(size: tuple[int, int], shape: str, inset: float = 0.0, radius: int = 40, scale: int = 4) -> Image.Image:
    """Anti-aliased white-on-black shape mask (mode L), supersampled."""
    w, h = size[0] * scale, size[1] * scale
    img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(img)
    d = inset * scale
    x0, y0, x1, y1 = d, d, w - 1 - d, h - 1 - d
    if x1 <= x0 or y1 <= y0:
        return Image.new("L", size, 0)
    if shape == "circle":
        r = min(x1 - x0, y1 - y0) / 2
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=255)
    elif shape == "ellipse":
        draw.ellipse((x0, y0, x1, y1), fill=255)
    elif shape == "rounded":
        r = max(0, radius * scale - d)
        draw.rounded_rectangle((x0, y0, x1, y1), radius=min(r, (x1 - x0) / 2, (y1 - y0) / 2), fill=255)
    elif shape == "rect":
        draw.rectangle((x0, y0, x1, y1), fill=255)
    else:
        draw.polygon(_shape_points(shape, x0, y0, x1, y1), fill=255)
    return img.resize(size, Image.LANCZOS)


@dataclass
class MaskFiles:
    mask: Path
    border: Path | None
    shadow: Path | None


def build_masks(layout: Layout, geo: Geometry, folder: Path) -> MaskFiles:
    folder.mkdir(parents=True, exist_ok=True)
    w, h = geo.box
    shape = layout.shape if layout.shape in SHAPES else "circle"
    base = draw_shape((w, h), shape, radius=layout.radius)
    mask = base
    if layout.feather > 0:
        # feather inside the edge so the border still covers the soft rim
        mask = draw_shape((w, h), shape, inset=layout.feather / 2, radius=layout.radius).filter(
            ImageFilter.GaussianBlur(layout.feather / 2))
    mask_path = folder / "mask.png"
    mask.save(mask_path)

    border_path = None
    if layout.border_w > 0:
        inner = draw_shape((w, h), shape, inset=layout.border_w, radius=layout.radius)
        ring = ImageChops.subtract(base, inner)
        color = Image.new("RGBA", (w, h), "#" + hex_color(layout.border_color, "ffffff"))
        color.putalpha(ring)
        border_path = folder / "border.png"
        color.save(border_path)

    shadow_path = None
    if layout.shadow and layout.shadow_opacity > 0:
        pad = geo.shadow_pad
        canvas = Image.new("L", (w + 2 * pad, h + 2 * pad), 0)
        canvas.paste(base, (pad, pad + max(2, layout.shadow_blur // 3)))
        canvas = canvas.filter(ImageFilter.GaussianBlur(max(1, layout.shadow_blur / 2)))
        alpha = canvas.point(lambda v: int(v * min(layout.shadow_opacity, 100) / 100))
        shadow = Image.new("RGBA", canvas.size, (0, 0, 0, 255))
        shadow.putalpha(alpha)
        shadow_path = folder / "shadow.png"
        shadow.save(shadow_path)
    return MaskFiles(mask_path, border_path, shadow_path)


# ---------------------------------------------------------------------------
# ffmpeg command
# ---------------------------------------------------------------------------
@dataclass
class Timing:
    main_duration: float
    clip_duration: float
    delay: float = 0.0
    end_mode: str = "hide"  # hide | freeze | loop
    max_duration: float = 0.0
    fps: float = 30.0

    @property
    def total(self) -> float:
        total = self.main_duration or (self.delay + self.clip_duration)
        if self.max_duration > 0:
            total = min(total, self.max_duration)
        return max(total, 0.1)

    @property
    def visible_until(self) -> float:
        return self.total if self.end_mode in ("freeze", "loop") else self.delay + self.clip_duration


def _main_chain(geo: Geometry, layout: Layout, tonemap: str | None) -> str:
    cw, ch = geo.canvas_w, geo.canvas_h
    sw, sh = geo.main_scale
    px, py = geo.main_pos
    pre = f"{tonemap}," if tonemap else ""
    if geo.fit == "cover":
        return f"[0:v]{pre}scale={sw}:{sh},crop={cw}:{ch}:{px}:{py},setsar=1[base0]"
    if geo.fit == "color":
        color = hex_color(layout.bg_color)
        return f"[0:v]{pre}scale={sw}:{sh},pad={cw}:{ch}:{px}:{py}:color=0x{color},setsar=1[base0]"
    # blur: a cheap blur on a 1/8 size copy, scaled back up behind the letterboxed video
    bw, bh = even(cw / 8), even(ch / 8)
    radius = max(1, min(int(layout.blur / 4) or 1, min(bw, bh) // 2 - 1))
    return (f"[0:v]{pre}split=2[m0][m1];"
            f"[m0]scale={bw}:{bh}:force_original_aspect_ratio=increase,crop={bw}:{bh},boxblur={radius}:2,"
            f"scale={cw}:{ch},setsar=1[bg];"
            f"[m1]scale={sw}:{sh},setsar=1[fg];[bg][fg]overlay={px}:{py}[base0]")


def build_filter(layout: Layout, geo: Geometry, timing: Timing, masks: MaskFiles, *, tonemap: str | None = None,
                 audio: str = "none", main_volume: float = 1.0, avatar_volume: float = 1.0,
                 preview: bool = False) -> tuple[str, list[str]]:
    """Return (filter_complex, maps). Input order: 0 main, 1 avatar, 2 mask, 3 shadow?, 4 border?."""
    fps = timing.fps
    cw_, ch_, cx_, cy_ = geo.crop
    w, h = geo.box
    x, y = geo.pos
    delay = 0.0 if preview else max(timing.delay, 0.0)
    parts = [_main_chain(geo, layout, tonemap), f"[base0]fps={fps:g},format=yuv420p[base]"]

    av = f"[1:v]fps={fps:g},crop={cw_}:{ch_}:{cx_}:{cy_}"
    if layout.mirror:
        av += ",hflip"
    av += f",scale={w}:{h},format=rgba"
    if layout.chroma:
        av += (f",colorkey=0x{hex_color(layout.key_color, '00ff00')}:"
               f"{min(max(layout.key_similarity, 0.01), 1):.3f}:{min(max(layout.key_blend, 0), 1):.3f}")
        parts.append(av + "[av0]")
        parts.append("[2:v]format=gray[mk]")
        parts.append("[av0]split=2[a1][a2];[a1]alphaextract[ka];[ka][mk]blend=all_mode=multiply[am];"
                     "[a2][am]alphamerge[av1]")
    else:
        parts.append(av + "[av0]")
        parts.append("[2:v]format=gray[mk]")
        parts.append("[av0][mk]alphamerge[av1]")

    post = []
    if layout.opacity < 100:
        post.append(f"colorchannelmixer=aa={max(layout.opacity, 0) / 100:.3f}")
    if not preview and layout.fade_in > 0:
        post.append(f"fade=t=in:st=0:d={layout.fade_in:g}:alpha=1")
    if not preview and layout.fade_out > 0 and timing.end_mode == "hide" and timing.clip_duration > layout.fade_out:
        post.append(f"fade=t=out:st={timing.clip_duration - layout.fade_out:.3f}:d={layout.fade_out:g}:alpha=1")
    post.append(f"setpts=PTS-STARTPTS+{delay:.3f}/TB")
    parts.append("[av1]" + ",".join(post) + "[av]")

    enable = "" if preview else f":enable='between(t,{delay:.3f},{timing.visible_until:.3f})'"
    current = "[base]"
    next_input = 3
    if masks.shadow:
        pad = geo.shadow_pad
        parts.append(f"{current}[{next_input}:v]overlay={x - pad}:{y - pad}{enable}[s1]")
        current, next_input = "[s1]", next_input + 1
    eof = "repeat" if timing.end_mode == "freeze" else "pass"
    parts.append(f"{current}[av]overlay={x}:{y}:eof_action={eof}:format=auto[s2]")
    current = "[s2]"
    if masks.border:
        parts.append(f"{current}[{next_input}:v]overlay={x}:{y}{enable}[s3]")
        current = "[s3]"
    parts.append(f"{current}format=yuv420p[v]")

    maps = ["-map", "[v]"]
    ms = int(round(delay * 1000))
    if audio == "main":
        maps += ["-map", "0:a:0?"]
    elif audio == "avatar":
        parts.append(f"[1:a]adelay={ms}|{ms},volume={avatar_volume:.2f}[aout]")
        maps += ["-map", "[aout]"]
    elif audio == "mix":
        parts.append(f"[0:a]volume={main_volume:.2f}[a0];[1:a]adelay={ms}|{ms},volume={avatar_volume:.2f}[a1];"
                     "[a0][a1]amix=inputs=2:duration=first:dropout_transition=0,volume=2[aout]")
        maps += ["-map", "[aout]"]
    return ";".join(parts), maps


def build_inputs(main: Path, avatar: Path, masks: MaskFiles, timing: Timing, *, preview_at: float | None = None) -> list[str]:
    still_t = 1.0 if preview_at is not None else timing.total + 1.0
    args: list[str] = []
    if preview_at is not None:
        args += ["-ss", f"{preview_at:.3f}", "-i", str(main)]
        args += ["-ss", f"{max(0.0, min(preview_at - timing.delay, timing.clip_duration - 0.1)):.3f}", "-i", str(avatar)]
    else:
        args += ["-i", str(main)]
        if timing.end_mode == "loop":
            args += ["-stream_loop", "-1"]
        args += ["-i", str(avatar)]
    # The mask is paired frame-by-frame with the avatar (alphamerge). On newer
    # ffmpeg alphamerge repeats the last avatar frame while the mask lasts, so in
    # "hide" mode the mask must end with the clip or the avatar would freeze.
    mask_t = still_t
    if preview_at is None and timing.end_mode == "hide" and timing.clip_duration:
        mask_t = timing.clip_duration
    for still, seconds in ((masks.mask, mask_t), (masks.shadow, still_t), (masks.border, still_t)):
        if still:
            args += ["-loop", "1", "-framerate", f"{timing.fps:g}", "-t", f"{seconds:.3f}", "-i", str(still)]
    return args
