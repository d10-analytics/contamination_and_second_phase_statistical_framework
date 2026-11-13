# This file checks the time-lapse video without needing a real ffmpeg. The
# arguments are checked through write_video with real values: a video too short
# to give each shown step one frame names the smallest every that passes (and
# that every is checked against a brute-force search, one less fails), and
# every other refusal leaves the folder unchanged with no ffmpeg anywhere. The
# lookup is checked in a temporary folder for the interpreter-first, path-only
# and neither cases, and through write_video with the interpreter and PATH
# pointed at empty folders. A fake ffmpeg script, written beside a temporary
# interpreter path, records its arguments and the bytes sent to it, which
# proves the rates and options, that each shown step is sent once as a
# 1920 x 1080 RGBA frame, the stderr line, which layout reaches the slate
# drawing and the dashboard figure, and that a failing ffmpeg leaves an earlier
# video untouched and no temporary file. The real encode is in the encoding file.

import json
import math
import os
import stat
import subprocess
import sys

import pytest

from placement_simulation import saving
from placement_simulation.placement import PlacementResult, PlacementRow
from placement_visualization import dashboard, slate_drawing, stills, video, write_packing_map, write_still_images, write_video

REAL_PYTHON = sys.executable
SQUARE = {"shape": "square", "side_um": 100.0}
INPUT = {"kind": "generator", "measurements_sha256": "0" * 64}
FRAME_BYTES = 1920 * 1080 * 4

FAKE_RECORDING = """#!{python}
import json
import sys

data = sys.stdin.buffer.read()
with open({record!r}, "w") as handle:
    json.dump({{"args": sys.argv[1:], "stdin_bytes": len(data)}}, handle)
with open(sys.argv[-1], "wb") as handle:
    handle.write(b"fake video bytes")
"""

FAKE_FAILING = """#!{python}
import sys

sys.stdin.buffer.read()
with open(sys.argv[-1], "wb") as handle:
    handle.write(b"partial bytes")
sys.exit(1)
"""


class MarkerError(Exception):
    """Raised by a replacement so a test can tell the replacement was reached."""


def make_row(order, label, diameter, x, y):
    """Return a placement row; x of None means the particle was not placed."""
    return PlacementRow(
        placement_order=order,
        source_index=order,
        object_id=f"p{order}",
        feature_label=label,
        equivalent_diameter_um=diameter,
        placed=x is not None,
        x_um=x,
        y_um=y,
        tries=1,
        edge_rejections=0,
        overlap_rejections=0,
    )


def make_rows(count):
    """Return count small particles in two classes along a line, every fourth one not placed."""
    rows = []
    for order in range(1, count + 1):
        label = "AB"[order % 2]
        if order % 4 == 0:
            rows.append(make_row(order, label, 4.0, None, None))
        else:
            rows.append(make_row(order, label, 4.0 + order % 3, -40.0 + 7.0 * (order % 10), -30.0 + 6.0 * (order % 8)))
    return rows


def save_rows(folder, rows):
    """Save a hand-built run in the folder and return its base path."""
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
        rows=tuple(rows),
    )
    base = folder / "run"
    saving.write_placement_outputs(result, base)
    return base


def folder_state(folder):
    """Return every file in the folder as (name, bytes), sorted, so a change of any file or name shows."""
    return sorted((entry.name, entry.read_bytes()) for entry in folder.iterdir())


def no_ffmpeg_anywhere(tmp_path, monkeypatch):
    """Point the interpreter path and PATH at empty folders so no ffmpeg can be found."""
    empty = tmp_path / "empty_bin"
    empty.mkdir()
    monkeypatch.setattr(sys, "executable", str(empty / "python"))
    monkeypatch.setenv("PATH", str(empty))


def install_fake_ffmpeg(tmp_path, monkeypatch, template):
    """Write the fake ffmpeg script beside a temporary interpreter path, hide the real PATH, and return the record file path."""
    fake_bin = tmp_path / "fake_bin"
    fake_bin.mkdir()
    empty = tmp_path / "empty_path"
    empty.mkdir()
    record = tmp_path / "record.json"
    script = fake_bin / "ffmpeg"
    script.write_text(template.format(python=REAL_PYTHON, record=str(record)), encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setattr(sys, "executable", str(fake_bin / "python"))
    monkeypatch.setenv("PATH", str(empty))
    return record


def run_folder(tmp_path, count):
    """Return (base path, a separate folder) for a saved run of count particles; the run's own folder is tmp_path / run_files."""
    folder = tmp_path / "run_files"
    folder.mkdir()
    return save_rows(folder, make_rows(count)), folder


def test_the_example_refusal_names_every_two(tmp_path, monkeypatch):
    """Ten particles, every 1, 0.2 s at 30 fps is six frames for ten steps, and the error names every=2."""
    base, folder = run_folder(tmp_path, 10)
    no_ffmpeg_anywhere(tmp_path, monkeypatch)
    before = folder_state(folder)
    with pytest.raises(ValueError) as caught:
        write_video(base, 1, 0.2, fps=30)
    assert "every=2" in str(caught.value)
    assert "longer video" in str(caught.value)
    assert folder_state(folder) == before


def test_the_named_every_passes_and_one_less_fails(tmp_path):
    """For several runs and lengths the named every is the smallest one a brute-force search finds passing."""
    cases = [(10, 0.2, 30), (7, 0.1, 30), (100, 1.0, 30), (13, 0.5, 24), (50, 1.7, 10), (9, 0.3, 30)]
    for count, seconds, fps in cases:
        available = seconds * fps
        passing = []
        for every in range(1, count + 1):
            if available / len(stills.shown_steps(count, every)) >= 1:
                passing.append(every)
        named = video.smallest_passing_every(count, seconds, fps)
        assert named == passing[0]
        video.check_schedule(stills.shown_steps(count, named), count, named, seconds, fps)
        if named > 1:
            with pytest.raises(ValueError, match=f"every={named}"):
                video.check_schedule(stills.shown_steps(count, named - 1), count, named - 1, seconds, fps)


def test_a_video_shorter_than_one_frame_says_so_and_names_no_every(tmp_path, monkeypatch):
    """0.02 s at 30 fps is 0.6 frames: no every passes, and the message says the video is shorter than one frame."""
    base, folder = run_folder(tmp_path, 5)
    no_ffmpeg_anywhere(tmp_path, monkeypatch)
    before = folder_state(folder)
    with pytest.raises(ValueError, match="shorter than one frame") as caught:
        write_video(base, 5, 0.02, fps=30)
    assert "every=" not in str(caught.value)
    assert folder_state(folder) == before


def test_a_schedule_with_exactly_one_frame_per_step_is_accepted(tmp_path, monkeypatch):
    """Three shown steps in 0.1 s at 30 fps is exactly one frame each; the refusal is not reached (the missing ffmpeg is)."""
    base, folder = run_folder(tmp_path, 3)
    no_ffmpeg_anywhere(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError):
        write_video(base, 1, 0.1, fps=30)


def test_the_frame_count_rounds_a_half_up():
    """A half frame rounds up, as ffmpeg does, where Python's round goes to the even number."""
    assert video.frame_count(1.3, 25) == 33
    assert video.frame_count(2.35, 30) == 71
    assert video.frame_count(1.37, 30) == 41
    assert video.frame_count(0.1, 30) == 3


def test_the_shown_steps_marker_is_reached_before_the_ffmpeg_lookup(tmp_path, monkeypatch):
    """A replacement of the stills shown-steps function is called by write_video, and its error beats the missing-ffmpeg error."""
    base, folder = run_folder(tmp_path, 6)
    no_ffmpeg_anywhere(tmp_path, monkeypatch)

    def marker(particle_count, every):
        """Fail as a marker."""
        raise MarkerError("shown steps reached")

    monkeypatch.setattr(stills, "shown_steps", marker)
    with pytest.raises(MarkerError):
        write_video(base, 2, 1.0)


def test_find_ffmpeg_prefers_the_interpreter_folder(tmp_path):
    """With an ffmpeg in both places the interpreter folder's is returned."""
    interpreter = tmp_path / "interpreter"
    on_path = tmp_path / "on_path"
    interpreter.mkdir()
    on_path.mkdir()
    for folder in (interpreter, on_path):
        script = folder / "ffmpeg"
        script.write_text("#!/bin/sh\n", encoding="utf-8")
        script.chmod(0o755)
    assert video.find_ffmpeg(interpreter, str(on_path)) == str(interpreter / "ffmpeg")


def test_find_ffmpeg_falls_back_to_the_search_path(tmp_path):
    """With none beside the interpreter, the one on the search path is returned."""
    interpreter = tmp_path / "interpreter"
    on_path = tmp_path / "on_path"
    interpreter.mkdir()
    on_path.mkdir()
    script = on_path / "ffmpeg"
    script.write_text("#!/bin/sh\n", encoding="utf-8")
    script.chmod(0o755)
    assert video.find_ffmpeg(interpreter, str(on_path)) == str(script)


def test_find_ffmpeg_with_neither_raises_install_guidance(tmp_path):
    """With none in either place a RuntimeError tells the user to install ffmpeg."""
    interpreter = tmp_path / "interpreter"
    interpreter.mkdir()
    with pytest.raises(RuntimeError, match="install ffmpeg"):
        video.find_ffmpeg(interpreter, str(tmp_path / "nowhere"))
    with pytest.raises(RuntimeError, match="install ffmpeg"):
        video.find_ffmpeg(interpreter, "")


def test_a_missing_ffmpeg_fails_only_the_video(tmp_path, monkeypatch):
    """With no ffmpeg anywhere write_video raises RuntimeError and writes nothing, while the pictures still succeed."""
    base, folder = run_folder(tmp_path, 6)
    no_ffmpeg_anywhere(tmp_path, monkeypatch)
    before = folder_state(folder)
    with pytest.raises(RuntimeError, match="ffmpeg"):
        write_video(base, 2, 1.0, fps=10)
    assert folder_state(folder) == before
    paths = write_still_images(base, 3, long_side_px=60)
    assert len(paths) == 2 and all(path.is_file() for path in paths)
    assert write_packing_map(base, long_side_px=60).is_file()


def test_every_argument_refusal_is_a_value_error_with_the_folder_unchanged(tmp_path, monkeypatch):
    """Bad every, video_seconds, fps and layout are ValueErrors raised before the (absent) ffmpeg lookup."""
    base, folder = run_folder(tmp_path, 6)
    no_ffmpeg_anywhere(tmp_path, monkeypatch)
    before = folder_state(folder)
    for bad in [0, -1, 1.5, True, "2"]:
        with pytest.raises(ValueError, match="every"):
            write_video(base, bad, 1.0)
    for bad in [0, -1.0, math.nan, math.inf, True, "1"]:
        with pytest.raises(ValueError, match="video_seconds"):
            write_video(base, 2, bad)
        with pytest.raises(ValueError, match="fps"):
            write_video(base, 2, 1.0, fps=bad)
    for bad in ["grid", "", "Slate", None]:
        with pytest.raises(ValueError, match="layout"):
            write_video(base, 2, 1.0, layout=bad)
    assert folder_state(folder) == before


def test_a_file_with_no_particles_and_reader_refusals_are_value_errors(tmp_path, monkeypatch):
    """An empty run, a damaged header and a missing run are ValueErrors with nothing written and no ffmpeg."""
    no_ffmpeg_anywhere(tmp_path, monkeypatch)
    empty_folder = tmp_path / "empty_run"
    empty_folder.mkdir()
    empty_base = save_rows(empty_folder, [])
    before = folder_state(empty_folder)
    with pytest.raises(ValueError, match="no particles"):
        write_video(empty_base, 1, 1.0)
    assert folder_state(empty_folder) == before
    base, folder = run_folder(tmp_path, 6)
    placements = folder / "run_placements.csv"
    placements.write_text(placements.read_text(encoding="utf-8").replace("placement_order", "order"), encoding="utf-8")
    before = folder_state(folder)
    with pytest.raises(ValueError, match="header"):
        write_video(base, 2, 1.0)
    assert folder_state(folder) == before
    with pytest.raises(ValueError, match="does not exist"):
        write_video(folder / "other", 2, 1.0)


def test_an_existing_output_is_refused_without_overwrite(tmp_path, monkeypatch):
    """An existing video, and a folder at its name, are ValueErrors before the ffmpeg lookup; the file is untouched."""
    base, folder = run_folder(tmp_path, 6)
    no_ffmpeg_anywhere(tmp_path, monkeypatch)
    existing = folder / "run_slate.mp4"
    existing.write_bytes(b"old video")
    before = folder_state(folder)
    with pytest.raises(ValueError, match="overwrite=True"):
        write_video(base, 2, 1.0)
    assert folder_state(folder) == before
    existing.unlink()
    existing.mkdir()
    with pytest.raises(ValueError, match="folder"):
        write_video(base, 2, 1.0)


def test_the_fake_ffmpeg_receives_the_rates_options_and_one_frame_per_shown_step(tmp_path, monkeypatch):
    """Arguments carry the input rate, codec, bitrate, pixel format, output rate and mp4; stdin holds exactly the shown steps of RGBA 1920 x 1080."""
    base, folder = run_folder(tmp_path, 7)
    record = install_fake_ffmpeg(tmp_path, monkeypatch, FAKE_RECORDING)
    path = write_video(base, 3, 2.0, fps=10)
    assert path == folder / "run_slate.mp4"
    assert path.read_bytes() == b"fake video bytes"
    recorded = json.loads(record.read_text(encoding="utf-8"))
    args = recorded["args"]
    shown = len(stills.shown_steps(7, 3))
    assert shown == 3
    input_position = args.index("pipe:")
    assert args[input_position - 1] == "-i"
    assert args[args.index("-framerate") + 1] == str(shown / 2.0)
    assert args.index("-framerate") < input_position
    assert args[args.index("-s") + 1] == "1920x1080"
    assert args[args.index("-pix_fmt") + 1] == "rgba"
    after = args[input_position + 1 :]
    assert after[after.index("-vcodec") + 1] == "libopenh264"
    assert after[after.index("-b") + 1] == "8000k"
    assert after[after.index("-pix_fmt") + 1] == "yuv420p"
    assert after[after.index("-r") + 1] == "10"
    assert after[after.index("-f") + 1] == "mp4"
    assert recorded["stdin_bytes"] == shown * FRAME_BYTES
    temporary_name = os.path.basename(args[-1])
    assert temporary_name.startswith(".run_slate.mp4.") and temporary_name.endswith(".tmp")
    assert sorted(entry.name for entry in folder.iterdir()) == ["run_placements.csv", "run_slate.mp4", "run_summary.txt"]


def test_the_stderr_line_carries_the_four_values_before_rendering(tmp_path, monkeypatch, capsys):
    """The line shows seconds, shown steps, fps and frames, and is printed before the first frame is drawn."""
    base, folder = run_folder(tmp_path, 5)
    install_fake_ffmpeg(tmp_path, monkeypatch, FAKE_RECORDING)
    write_video(base, 2, 1.0, fps=10)
    assert "video: 1.0 s, 3 shown steps, 10.0 fps, 10 frames\n" in capsys.readouterr().err
    write_video(base, 2, 2.35, fps=30, overwrite=True)
    assert "video: 2.35 s, 3 shown steps, 30.0 fps, 71 frames\n" in capsys.readouterr().err

    def marker(axes, slate, rows, colours):
        """Fail as a marker before any frame exists."""
        raise MarkerError("drawing reached")

    monkeypatch.setattr(slate_drawing, "draw_slate", marker)
    with pytest.raises(MarkerError):
        write_video(base, 2, 1.0, fps=10, overwrite=True)
    assert "video: 1.0 s, 3 shown steps, 10.0 fps, 10 frames\n" in capsys.readouterr().err


def test_the_slate_layout_draws_through_the_slate_drawing(tmp_path, monkeypatch):
    """A replacement of the slate-drawing function is reached by the slate layout, and nothing is left behind."""
    base, folder = run_folder(tmp_path, 5)
    install_fake_ffmpeg(tmp_path, monkeypatch, FAKE_RECORDING)
    before = folder_state(folder)

    def marker(axes, slate, rows, colours):
        """Fail as a marker."""
        raise MarkerError("slate drawing reached")

    monkeypatch.setattr(slate_drawing, "draw_slate", marker)
    with pytest.raises(MarkerError):
        write_video(base, 2, 1.0, fps=10)
    assert folder_state(folder) == before


def test_the_dashboard_layout_uses_the_dashboard_figure_and_the_slate_drawing(tmp_path, monkeypatch):
    """layout="dashboard" reaches the dashboard figure function, and that figure draws its slate through the slate drawing."""
    base, folder = run_folder(tmp_path, 5)
    install_fake_ffmpeg(tmp_path, monkeypatch, FAKE_RECORDING)
    before = folder_state(folder)

    def figure_marker(result, data, step, width_px, height_px):
        """Fail as a marker."""
        raise MarkerError("dashboard figure reached")

    original = dashboard.build_dashboard_figure
    monkeypatch.setattr(dashboard, "build_dashboard_figure", figure_marker)
    with pytest.raises(MarkerError, match="dashboard figure"):
        write_video(base, 2, 1.0, fps=10, layout="dashboard")
    assert folder_state(folder) == before
    monkeypatch.setattr(dashboard, "build_dashboard_figure", original)

    def slate_marker(axes, slate, rows, colours):
        """Fail as a marker."""
        raise MarkerError("slate drawing reached")

    monkeypatch.setattr(slate_drawing, "draw_slate", slate_marker)
    with pytest.raises(MarkerError, match="slate drawing"):
        write_video(base, 2, 1.0, fps=10, layout="dashboard")
    assert folder_state(folder) == before


def test_the_dashboard_layout_sends_one_frame_per_shown_step_to_the_fake(tmp_path, monkeypatch):
    """The dashboard layout writes run_dashboard.mp4 and pipes exactly the shown steps of 1920 x 1080 frames."""
    base, folder = run_folder(tmp_path, 9)
    record = install_fake_ffmpeg(tmp_path, monkeypatch, FAKE_RECORDING)
    path = write_video(base, 4, 1.0, fps=24, layout="dashboard")
    assert path == folder / "run_dashboard.mp4"
    recorded = json.loads(record.read_text(encoding="utf-8"))
    assert recorded["stdin_bytes"] == len(stills.shown_steps(9, 4)) * FRAME_BYTES


def test_a_failing_ffmpeg_raises_and_leaves_the_earlier_video_and_no_temporary_file(tmp_path, monkeypatch, capsys):
    """An ffmpeg that writes partial bytes and exits 1 makes write_video raise after the stderr line; the old video is byte-identical."""
    base, folder = run_folder(tmp_path, 5)
    install_fake_ffmpeg(tmp_path, monkeypatch, FAKE_FAILING)
    existing = folder / "run_slate.mp4"
    existing.write_bytes(b"earlier good video")
    before = folder_state(folder)
    with pytest.raises((OSError, subprocess.CalledProcessError)):
        write_video(base, 2, 1.0, fps=10, overwrite=True)
    assert "video: 1.0 s, 3 shown steps, 10.0 fps, 10 frames" in capsys.readouterr().err
    assert existing.read_bytes() == b"earlier good video"
    assert folder_state(folder) == before
    assert [entry.name for entry in folder.iterdir() if entry.name.endswith(".tmp")] == []
