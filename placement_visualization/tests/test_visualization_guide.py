# This file checks the drawing guide and the links that lead to it. It reads
# text only: it runs no example and needs no ffmpeg, so every test lane
# collects it. The tests check that the guide holds its command, its words and
# every argument, that it links back to the placement guide, that every
# placement_visualization path it names exists (or is one of the two output
# folders or a file the example writes there), and that the two READMEs point
# to it.

import re
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent.parent
GUIDE_PATH = REPOSITORY_ROOT / "placement_visualization" / "README.md"
ROOT_LINK = "](placement_visualization/README.md)"
PLACEMENT_LINK = "](../placement_visualization/README.md)"
BACK_LINK = "](../placement_simulation/README.md)"
RUN_COMMAND = "python3 -m placement_visualization.example.run_example"
EXAMPLE_FILES = [
    "example_dashboard.mp4",
    "example_packing_map.png",
    "example_packing_map_sigma_series.png",
    "example_placements.csv",
    "example_step_100.png",
    "example_step_200.png",
    "example_step_300.png",
    "example_step_400.png",
    "example_summary.txt",
]
FUNCTION_WORDS = ["write_still_images", "write_video", "write_packing_map", "write_packing_map_series", "read_placement_outputs"]
ARGUMENT_WORDS = ["every", "video_seconds", "fps", "layout", "slate", "dashboard", "long_side_px", "sigma_um", "sigmas_um", "overwrite"]
OTHER_WORDS = ["ffmpeg", "PATH", "conda install ffmpeg", "libopenh264", "<base>_step_<k>.png", "<base>_slate.mp4", "<base>_dashboard.mp4", "<base>_packing_map.png", "<base>_packing_map_sigma_<sigma>.png", "<base>_packing_map_sigma_series.png"]
OUTPUT_PREFIXES = ["placement_visualization/example/output_random/", "placement_visualization/example/output_center/"]


def guide_text():
    """Return the text of the drawing guide."""
    return GUIDE_PATH.read_text(encoding="utf-8")


def test_guide_holds_the_command_the_functions_the_arguments_and_the_names():
    """The guide has the run command, the back link, every function and argument word, the ffmpeg words, every output name pattern and every example file."""
    guide = guide_text()
    assert RUN_COMMAND in guide
    assert BACK_LINK in guide
    for word in FUNCTION_WORDS + ARGUMENT_WORDS + OTHER_WORDS + EXAMPLE_FILES:
        assert word in guide, word


def test_guide_states_what_long_side_px_measures_for_each_picture():
    """The guide says long_side_px is the slate's long side in each still PNG, with the legend added below it, and the number of grid cells along the slate's long side for the packing map."""
    guide = guide_text()
    assert "long side of the slate in each PNG" in guide
    assert "at least 400" in guide
    assert "number of grid cells along the long side of the slate" in guide


def test_guide_names_only_existing_paths_or_written_files():
    """Every placement_visualization/ path in the guide exists, or is one of the two output folders or a file the example writes there."""
    found = re.findall(r"placement_visualization/[A-Za-z0-9_./-]+", guide_text())
    assert len(found) > 0
    for text in found:
        cleaned = text.rstrip(".,:;")
        if cleaned in OUTPUT_PREFIXES:
            continue
        if any(cleaned.startswith(prefix) and cleaned[len(prefix):] in EXAMPLE_FILES for prefix in OUTPUT_PREFIXES):
            continue
        assert (REPOSITORY_ROOT / cleaned).exists(), cleaned


def test_root_readme_links_to_the_guide():
    """The top-level README links the drawing guide by its exact path."""
    assert ROOT_LINK in (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")


def test_placement_guide_points_to_the_drawing_guide_and_names_the_reader():
    """The placement guide links the drawing guide and names read_placement_outputs."""
    text = (REPOSITORY_ROOT / "placement_simulation" / "README.md").read_text(encoding="utf-8")
    assert PLACEMENT_LINK in text
    assert "read_placement_outputs" in text
