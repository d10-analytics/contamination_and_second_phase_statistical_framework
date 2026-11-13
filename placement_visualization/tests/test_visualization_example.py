# This file runs the drawing example once, the way its guide says: as a
# separate Python process from the repository root, on the interpreter running
# the tests. Files left by an earlier run are removed from both output folders
# first, so finding the guide's files afterwards proves this run wrote them.
# The random folder must hold a random run of 400 particles and the center
# folder a centre run of 800, which fills the slate so that some are not placed.
# The test never skips: without ffmpeg beside the interpreter or on the PATH
# the example fails, and so does this test.

import os
import subprocess
import sys
from pathlib import Path

from placement_simulation import saving

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent.parent
EXAMPLE_FOLDER = REPOSITORY_ROOT / "placement_visualization" / "example"
SHARED_FILES = [
    "example_dashboard.mp4",
    "example_packing_map.png",
    "example_packing_map_sigma_120.0.png",
    "example_packing_map_sigma_15.0.png",
    "example_packing_map_sigma_240.0.png",
    "example_packing_map_sigma_30.0.png",
    "example_packing_map_sigma_5.0.png",
    "example_packing_map_sigma_series.png",
    "example_placements.csv",
    "example_summary.txt",
]
OUTPUT_FOLDERS = {
    "output_random": ("random", 400, "example_packing_map_sigma_60.2.png"),
    "output_center": ("centre", 800, "example_packing_map_sigma_60.9.png"),
}


def expected_files(particle_count, default_sigma_map):
    """Return the sorted file names of one run: the shared files, its default-sigma series map and one still per 100 particles."""
    stills = [f"example_step_{step}.png" for step in range(100, particle_count + 1, 100)]
    return sorted(SHARED_FILES + [default_sigma_map] + stills)
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def test_module_run_writes_the_guides_files_for_each_mode_in_its_folder():
    """Running the module the way the guide says exits 0 and leaves exactly the listed files in both folders, the pictures being PNGs and the videos not empty, with a random run of 400 particles in output_random and a centre run of 800 in output_center that leaves some unplaced."""
    for name in OUTPUT_FOLDERS:
        folder = EXAMPLE_FOLDER / name
        folder.mkdir(parents=True, exist_ok=True)
        for path in folder.iterdir():
            path.unlink()
    command = [sys.executable, "-m", "placement_visualization.example.run_example"]
    finished = subprocess.run(command, cwd=REPOSITORY_ROOT, capture_output=True, text=True, env=os.environ.copy())
    assert finished.returncode == 0, finished.stderr
    for name, (mode, particle_count, default_sigma_map) in OUTPUT_FOLDERS.items():
        folder = EXAMPLE_FOLDER / name
        expected = expected_files(particle_count, default_sigma_map)
        assert sorted(path.name for path in folder.iterdir()) == expected, name
        for file_name in expected:
            if file_name.endswith(".png"):
                assert (folder / file_name).read_bytes().startswith(PNG_SIGNATURE), (name, file_name)
        assert (folder / "example_dashboard.mp4").stat().st_size > 0, name
        result = saving.read_placement_outputs(folder / "example")
        assert result.mode == mode, name
        assert len(result.rows) == particle_count, name
    centre = saving.read_placement_outputs(EXAMPLE_FOLDER / "output_center" / "example")
    assert 0 < sum(not row.placed for row in centre.rows) < len(centre.rows)
