# This file runs the drawing example once, the way its guide says: as a
# separate Python process from the repository root, on the interpreter running
# the tests. Files left by an earlier run are removed first, so finding the
# guide's files afterwards proves this run wrote them. The test never skips:
# without ffmpeg beside the interpreter or on the PATH the example fails, and so
# does this test.

import os
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_FOLDER = REPOSITORY_ROOT / "placement_visualization" / "example" / "output"
EXPECTED_FILES = [
    "example_dashboard.mp4",
    "example_packing_map.png",
    "example_placements.csv",
    "example_step_100.png",
    "example_step_200.png",
    "example_step_300.png",
    "example_step_400.png",
    "example_summary.txt",
]
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def test_module_run_writes_exactly_the_guides_eight_files():
    """Running the module the way the guide says exits 0 and leaves exactly the eight listed files, the pictures being PNGs and the video not empty."""
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    for path in OUTPUT_FOLDER.iterdir():
        path.unlink()
    command = [sys.executable, "-m", "placement_visualization.example.run_example"]
    finished = subprocess.run(command, cwd=REPOSITORY_ROOT, capture_output=True, text=True, env=os.environ.copy())
    assert finished.returncode == 0, finished.stderr
    assert sorted(path.name for path in OUTPUT_FOLDER.iterdir()) == EXPECTED_FILES
    for name in EXPECTED_FILES:
        if name.endswith(".png"):
            assert (OUTPUT_FOLDER / name).read_bytes().startswith(PNG_SIGNATURE), name
    assert (OUTPUT_FOLDER / "example_dashboard.mp4").stat().st_size > 0
