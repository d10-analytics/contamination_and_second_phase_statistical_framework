# This file encodes one tiny video per layout with the real ffmpeg found beside
# the running interpreter (or on the PATH) and reads the result back with the
# ffprobe that sits beside that ffmpeg. It checks the codec, pixel format, size,
# frame rate and the counted frames against the length line printed on stderr,
# that the matplotlib ffmpeg setting is unchanged afterwards, and that the
# folder holds exactly the two saved files and the video. The layout videos are
# away from a half (3 shown steps, 1 s, 10 fps is 10 frames); two slate videos
# whose seconds x fps ends in a half (5 steps, 1.3 s, 25 fps is 33 frames and
# 3 steps, 2.5 s, 3 fps is 8) prove the half that rounds up is encoded and not
# dropped. A missing ffmpeg or
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


def encode_and_probe(tmp_path, layout, capsys, every=2, seconds=SECONDS, fps=FPS):
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
    path = write_video(base, every, seconds, fps=fps, layout=layout)
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


def test_a_half_frame_video_has_the_frame_count_the_stderr_line_reports(tmp_path, capsys):
    """5 shown steps in 1.3 s at 25 fps (32.5 frames) and 3 shown steps in 2.5 s at 3 fps (7.5 frames) report 33 and 8 frames, and ffprobe counts exactly that many at the chosen rate."""
    cases = [(1, 1.3, 25, "video: 1.3 s, 5 shown steps, 25.0 fps, 33 frames\n", 33), (2, 2.5, 3, "video: 2.5 s, 3 shown steps, 3.0 fps, 8 frames\n", 8)]
    for every, seconds, fps, line, count in cases:
        case_folder = tmp_path / f"every_{every}"
        case_folder.mkdir()
        fields, error, names = encode_and_probe(case_folder, "slate", capsys, every, seconds, fps)
        assert line in error
        assert fields[4] == f"{fps}/1"
        assert int(fields[5]) == count
