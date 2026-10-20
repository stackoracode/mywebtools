import asyncio
import shutil

import pytest
from PIL import Image

from mywebtools import composite
from mywebtools.composite import Layout, Timing

HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def test_auto_fit_landscape_on_shorts_uses_blur():
    geo = composite.compute_geometry(Layout(), 1920, 1080, 1080, 1920)
    assert (geo.canvas_w, geo.canvas_h) == (1080, 1920)
    assert geo.fit == "blur"
    assert geo.main_scale == (1080, 608)
    assert geo.main_pos == (0, (1920 - 608) // 2)


def test_auto_fit_portrait_fills():
    geo = composite.compute_geometry(Layout(), 1080, 1920, 1080, 1920)
    assert geo.fit == "cover"


def test_bottom_left_respects_shorts_safe_area():
    layout = Layout(size=360, margin_x=40, margin_y=40, anchor="bottom_left", safe_area=True)
    geo = composite.compute_geometry(layout, 1080, 1920, 1080, 1920)
    x, y = geo.pos
    assert x == round(0.03 * 1080 + 40)
    assert y + 360 <= 1920 * (1 - composite.SHORTS_SAFE["bottom"])


def test_safe_area_off_hugs_corner():
    layout = Layout(size=300, margin_x=10, margin_y=10, anchor="top_right", safe_area=False)
    geo = composite.compute_geometry(layout, 1080, 1920, 1080, 1920)
    assert geo.pos == (1080 - 10 - 300, 10)


def test_auto_crop_matches_reference_square():
    # the reference command used crop=1080:1080:0:210 on a 1080x1920 avatar
    w, h, x, y = composite.crop_rect(1080, 1920, 230, 230, "auto", 0, 0, 1)
    assert (w, h, x) == (1080, 1080, 0)
    assert abs(y - 210) < 30


def test_custom_crop_is_clamped():
    w, h, x, y = composite.crop_rect(1080, 1920, 200, 200, "custom", 1.0, 1.0, 2.0)
    assert (w, h) == (540, 540)
    assert x == 1080 - 540 and y == 1920 - 540


def test_full_crop_keeps_source_aspect():
    layout = Layout(size=400, crop_mode="full")
    geo = composite.compute_geometry(layout, 1080, 1920, 720, 1280)
    assert geo.box == (400, 712)
    assert geo.crop == (720, 1280, 0, 0)


@pytest.mark.parametrize("shape", composite.SHAPES)
def test_masks_for_every_shape(tmp_path, shape):
    layout = Layout(shape=shape, size=200, border_w=6, shadow=True)
    geo = composite.compute_geometry(layout, 1080, 1920, 1080, 1920)
    masks = composite.build_masks(layout, geo, tmp_path)
    mask = Image.open(masks.mask)
    assert mask.size == (200, 200)
    assert mask.getpixel((100, 100)) > 200  # centre is opaque
    assert mask.getpixel((0, 0)) < 50 or shape == "rect"
    assert masks.border and masks.shadow


def test_filter_graph_contents(tmp_path):
    layout = Layout(chroma=True, mirror=True, opacity=80)
    geo = composite.compute_geometry(layout, 1920, 1080, 1080, 1920)
    masks = composite.build_masks(layout, geo, tmp_path)
    timing = Timing(main_duration=10, clip_duration=6, delay=2, end_mode="hide", fps=30)
    graph, maps = composite.build_filter(layout, geo, timing, masks, audio="mix")
    assert "colorkey=0x00ff00" in graph and "hflip" in graph and "alphamerge" in graph
    assert "setpts=PTS-STARTPTS+2.000/TB" in graph
    assert "between(t,2.000,8.000)" in graph
    assert "amix=inputs=2" in graph and "adelay=2000|2000" in graph
    assert maps[-1] == "[aout]"
    inputs = composite.build_inputs(tmp_path / "m.mp4", tmp_path / "a.mp4", masks, timing)
    # mask lasts exactly as long as the clip in "hide" mode
    assert inputs[inputs.index(str(masks.mask)) - 2] == "6.000"


def _make(path, size, seconds, color, audio=True):
    import subprocess

    args = ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", f"color=c={color}:s={size}:r=25:d={seconds}"]
    if audio:
        args += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}", "-shortest"]
    subprocess.run(args + ["-pix_fmt", "yuv420p", str(path)], check=True)


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
@pytest.mark.parametrize("end_mode,audio", [("hide", "main"), ("freeze", "mix"), ("loop", "avatar")])
def test_ffmpeg_composite_runs(tmp_path, end_mode, audio):
    from mywebtools.media import ffmpeg_base, probe, run_cmd, video_encoder_args

    main, avatar = tmp_path / "main.mp4", tmp_path / "avatar.mp4"
    _make(main, "640x360", 4, "blue")
    _make(avatar, "360x640", 2, "green")
    layout = Layout(canvas="shorts_720", size=240, chroma=end_mode == "freeze", fade_in=0.2, fade_out=0.2)
    geo = composite.compute_geometry(layout, 640, 360, 360, 640)
    masks = composite.build_masks(layout, geo, tmp_path / "masks")
    timing = Timing(4, 2, 0.5, end_mode, 0, 25)
    graph, maps = composite.build_filter(layout, geo, timing, masks, audio=audio)
    out = tmp_path / "out.mp4"
    args = ffmpeg_base() + composite.build_inputs(main, avatar, masks, timing) + ["-filter_complex", graph] + maps
    args += ["-t", f"{timing.total:.3f}"] + video_encoder_args("libx264", 28, "ultrafast") + ["-c:a", "aac", str(out)]
    asyncio.run(run_cmd(args, timeout=120))
    info = asyncio.run(probe(out))
    assert (info.width, info.height) == (720, 1280)
    assert 3.5 <= info.duration <= 4.6
    assert info.has_audio


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_ffmpeg_preview_frame(tmp_path):
    from mywebtools.media import ffmpeg_base, run_cmd

    main, avatar = tmp_path / "main.mp4", tmp_path / "avatar.mp4"
    _make(main, "1280x720", 3, "red", audio=False)
    _make(avatar, "720x1280", 3, "white", audio=False)
    layout = Layout(shape="heart", size=300)
    geo = composite.compute_geometry(layout, 1280, 720, 720, 1280)
    masks = composite.build_masks(layout, geo, tmp_path / "m")
    timing = Timing(3, 3, 0, "freeze", 0, 30)
    graph, _ = composite.build_filter(layout, geo, timing, masks, preview=True)
    out = tmp_path / "p.jpg"
    args = ffmpeg_base() + composite.build_inputs(main, avatar, masks, timing, preview_at=1.0)
    asyncio.run(run_cmd(args + ["-filter_complex", graph, "-map", "[v]", "-frames:v", "1", str(out)], timeout=60))
    img = Image.open(out)
    assert img.size == (1080, 1920)
    cx, cy = geo.pos[0] + geo.box[0] // 2, geo.pos[1] + geo.box[1] // 2
    assert min(img.getpixel((cx, cy))) > 200  # white avatar inside the heart


def _has_filter(name):
    import subprocess

    if not HAS_FFMPEG:
        return False
    out = subprocess.run(["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True).stdout
    return f" {name} " in out


@pytest.mark.skipif(not _has_filter("ass"), reason="ffmpeg without libass")
def test_ffmpeg_burns_subtitles(tmp_path):
    from mywebtools import subtitles
    from mywebtools.media import ffmpeg_base, run_cmd

    main, avatar = tmp_path / "main.mp4", tmp_path / "avatar.mp4"
    _make(main, "720x1280", 2, "black", audio=False)
    _make(avatar, "720x1280", 2, "gray", audio=False)
    layout = Layout(canvas="shorts_720", size=200, anchor="top_left", safe_area=False)
    geo = composite.compute_geometry(layout, 720, 1280, 720, 1280)
    masks = composite.build_masks(layout, geo, tmp_path / "m")
    style = subtitles.SubStyle(size=80, color="#ffffff", outline=0, position="middle", box=False)
    (tmp_path / "subs.ass").write_text(subtitles.to_ass([subtitles.Cue(0, 5, "████")], style, 720, 1280), encoding="utf-8")
    timing = Timing(2, 2, 0, "freeze", 0, 25)
    graph, _ = composite.build_filter(layout, geo, timing, masks, preview=True, subtitles="subs.ass")
    out = tmp_path / "frame.png"
    args = ffmpeg_base() + composite.build_inputs(main, avatar, masks, timing, preview_at=0.5)
    asyncio.run(run_cmd(args + ["-filter_complex", graph, "-map", "[v]", "-frames:v", "1", str(out)], cwd=tmp_path, timeout=60))
    img = Image.open(out).convert("L")
    # the black frame has bright subtitle pixels in the middle band
    band = img.crop((0, 560, 720, 720))
    assert max(band.getdata()) > 200
