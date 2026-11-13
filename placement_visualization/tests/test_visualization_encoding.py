# This file encodes one tiny video per layout with the real ffmpeg found beside
# the running interpreter (or on the PATH) and reads the result back with the
# ffprobe that sits beside that ffmpeg. It checks the codec, pixel format, size,
# frame rate and the counted frames against the length line printed on stderr,
# that the matplotlib ffmpeg setting is unchanged afterwards, and that the
# folder holds exactly the two saved files and the video. The products are away
# from a half (3 shown steps, 1 s, 10 fps is 10 frames). A missing ffmpeg or
# ffprobe fails these tests; they never skip, so a run without ffmpeg cannot
# report the video as proven.

import os
import subprocess
import sys
from pathlib import Path

import matplotlib
import pytest

from placement_simulation import saving
from placement_simulation.placement import PlacementResult, PlacementRow
from placement_visualization import video, write_video

SQUARE = {"shape": "square", "side_um": 100.0}
INPUT = {"kind": "generator", "measurements_sha256": "0" * 64}
FPS = 10
SECONDS = 1.0


def make_rows():
    """Return five small particles in two classes; the third is not placed."""
    rows = []
    for order in range(1, 6):
        placed = order != 3
        rows.append(
            PlacementRow(
                placement_order=order,
                source_index=order,
                object_id=f"p{order}",
                feature_label="AB"[order % 2],
                equivalent_diameter_um=10.0,
                placed=placed,
                x_um=-30.0 + 15.0 * order if placed else None,
                y_um=10.0 * (order % 3) if placed else None,
                tries=1,
                edge_rejections=0,
                overlap_rejections=0,
            )
        )
    return rows


def find_tools():
    """Return (ffmpeg, ffprobe) found as write_video finds ffmpeg; fail, never skip, when either is missing."""
    try:
        ffmpeg = video.find_ffmpeg(Path(sys.executable).parent, os.environ.get("PATH", ""))
    except RuntimeError as problem:
        pytest.fail(f"the encoding test needs ffmpeg and found none: {problem}")
    name = "ffprobe.exe" if os.name == "nt" else "ffprobe"
    ffprobe = Path(ffmpeg).with_name(name)
    if not ffprobe.is_file():
        pytest.fail(f"the encoding test needs ffprobe beside {ffmpeg} and found none")
    return ffmpeg, str(ffprobe)


def probe(ffprobe, path):
    """Return the first video stream's (codec, width, height, pixel format, frame rate, counted frames) from ffprobe."""
    fields = "stream=codec_name,width,height,pix_fmt,r_frame_rate,nb_read_frames"
    command = [ffprobe, "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries", fields, "-of", "csv=p=0", str(path)]
    output = subprocess.run(command, capture_output=True, text=True, check=True).stdout.strip()
    return output.split(",")


def encode_and_probe(tmp_path, layout, capsys):
    """Encode the tiny run in the layout and return (probe fields, stderr text, folder names)."""
    ffmpeg, ffprobe = find_tools()
    folder = tmp_path / "run_files"
    folder.mkdir()
    result = PlacementResult(
        slate=SQUARE,
        mode="random",
        max_tries=5,
        buffer_um=1.0,
        pixel_um=1.0,
        order="shuffled",
        seed=1,
        input=INPUT,
        requested_coverage=0.0,
        achieved_coverage=0.0,
        rows=tuple(make_rows()),
    )
    base = folder / "run"
    saving.write_placement_outputs(result, base)
    capsys.readouterr()
    setting = matplotlib.rcParams["animation.ffmpeg_path"]
    path = write_video(base, 2, SECONDS, fps=FPS, layout=layout)
    assert matplotlib.rcParams["animation.ffmpeg_path"] == setting
    assert path == folder / f"run_{layout}.mp4"
    return probe(ffprobe, path), capsys.readouterr().err, sorted(entry.name for entry in folder.iterdir())


def check_encoding(fields, error, names, layout):
    """Assert the probe against the stderr line: h264, yuv420p, 1920 x 1080, the frame rate and the frame count."""
    codec, width, height, pixel_format, rate, counted = fields
    assert codec == "h264"
    assert pixel_format == "yuv420p"
    assert (width, height) == ("1920", "1080")
    assert rate == f"{FPS}/1"
    assert "video: 1.0 s, 3 shown steps, 10.0 fps, 10 frames\n" in error
    assert int(counted) == 10
    assert names == sorted(["run_placements.csv", f"run_{layout}.mp4", "run_summary.txt"])


def test_the_slate_layout_encodes_with_the_real_ffmpeg(tmp_path, capsys):
    """The slate video is h264, yuv420p, 1920 x 1080 at the chosen rate with the reported frame count."""
    fields, error, names = encode_and_probe(tmp_path, "slate", capsys)
    check_encoding(fields, error, names, "slate")


def test_the_dashboard_layout_encodes_with_the_real_ffmpeg(tmp_path, capsys):
    """The dashboard video is h264, yuv420p, 1920 x 1080 at the chosen rate with the reported frame count."""
    fields, error, names = encode_and_probe(tmp_path, "dashboard", capsys)
    check_encoding(fields, error, names, "dashboard")
