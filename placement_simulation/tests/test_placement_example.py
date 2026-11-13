# This file tests the placement example and its guide. The example is run once
# the way the guide says, as a separate Python process from the repository root
# on the interpreter running the tests (which need not have matplotlib), and its
# exit code, last line, six files and two summaries are checked. Other tests
# check that the example reaches the shared replay function, that it imports
# only the allowed modules, that its seed and limits equal the generator
# example's, and that the guide is reached from the top-level README, names
# every setting, column and file, and names only paths that exist.

import ast
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from placement_simulation import replay
from placement_simulation.example import run_example

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_FOLDER = REPOSITORY_ROOT / "placement_simulation" / "example" / "output"
GENERATOR_EXAMPLE = REPOSITORY_ROOT / "synthetic_data_generation" / "example" / "run_example.py"
GUIDE_PATH = REPOSITORY_ROOT / "placement_simulation" / "README.md"
GUIDE_LINK = "](placement_simulation/README.md)"
EXPECTED_FILES = [
    "centre_placements.csv",
    "centre_summary.txt",
    "example_particles.csv",
    "example_particles_summary.txt",
    "random_placements.csv",
    "random_summary.txt",
]
ALLOWED_MODULES = {"pathlib", "synthetic_data_generation", "placement_simulation", "numpy"}
SHARED_CONSTANTS = ["SEED", "SIEVE_LOWER_UM", "SIEVE_UPPER_UM"]
RUN_COMMAND = "python3 -m placement_simulation.example.run_example"
SETTING_WORDS = [
    "square",
    "rectangle",
    "circle",
    "side_um",
    "width_um",
    "height_um",
    "diameter_um",
    "random",
    "centre",
    "max_tries",
    "buffer_um",
    "pixel_um",
    "order",
    "as_given",
    "shuffled",
    "seed",
    "Requested coverage",
    "requested coverage",
    "slate",
    "mode",
    "input",
    "library_versions",
]
COLUMN_WORDS = [
    "placement_order",
    "source_index",
    "object_id",
    "feature_label",
    "equivalent_diameter_um",
    "placed",
    "x_um",
    "y_um",
    "tries",
    "edge_rejections",
    "overlap_rejections",
]
MARKER_TEXT = "replay marker"


class ReplayMarkerError(Exception):
    """Raised by a stand-in replay function so a test can see it reach the example."""


def read_summary_lines(name):
    """Return the lines of one of the example's saved summaries."""
    return (OUTPUT_FOLDER / name).read_text(encoding="utf-8").splitlines()


def requested_coverage_of(name):
    """Return the requested coverage number written in one of the example's summaries."""
    for line in read_summary_lines(name):
        if line.startswith("Requested coverage: "):
            return float(line.split(": ", 1)[1])
    raise AssertionError("no requested coverage line in " + name)


def input_line_of(name):
    """Return the readable input line of one of the example's summaries."""
    for line in read_summary_lines(name):
        if line.startswith("Input: "):
            return line
    raise AssertionError("no input line in " + name)


def rejection_total_of(name):
    """Return the total of edge and overlap rejections in one placements file."""
    lines = (OUTPUT_FOLDER / name).read_text(encoding="utf-8").splitlines()
    names = lines[0].split(",")
    edge_column = names.index("edge_rejections")
    overlap_column = names.index("overlap_rejections")
    total = 0
    for line in lines[1:]:
        cells = line.split(",")
        total += int(cells[edge_column]) + int(cells[overlap_column])
    return total


def test_module_run_writes_six_files_and_reports_identical():
    """Running the module the way the guide says exits 0, ends with identical: True and leaves exactly the six files, with a rejection in the random run."""
    # Files left by an earlier run are removed first, so finding them afterwards
    # proves this run wrote them into this exact folder.
    for name in EXPECTED_FILES:
        (OUTPUT_FOLDER / name).unlink(missing_ok=True)
    command = [sys.executable, "-m", "placement_simulation.example.run_example"]
    finished = subprocess.run(command, cwd=REPOSITORY_ROOT, capture_output=True, text=True, env=os.environ.copy())
    assert finished.returncode == 0, finished.stderr
    assert finished.stdout.splitlines()[-1] == "identical: True"
    assert sorted(path.name for path in OUTPUT_FOLDER.iterdir()) == EXPECTED_FILES
    assert rejection_total_of("random_placements.csv") > 0
    for name in ["random_summary.txt", "centre_summary.txt"]:
        coverage = requested_coverage_of(name)
        assert 0.35 <= coverage <= 0.45, name
        assert "'example_particles.csv'" in input_line_of(name)
        assert "measurements file" in input_line_of(name)


def test_example_compares_through_the_shared_replay(tmp_path, monkeypatch):
    """A replay function that raises a marker error makes main fail with that marker, so the random replay is the shared one."""

    def raising_replay(summary_path, measurements_path=None):
        """Stand in for the shared replay and raise the marker error."""
        raise ReplayMarkerError(MARKER_TEXT)

    monkeypatch.setattr(replay, "replay_placement", raising_replay)
    with pytest.raises(ReplayMarkerError, match=MARKER_TEXT):
        run_example.main(tmp_path)


def imported_module_names(path):
    """Return the top-level module names a Python file imports."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            names.add(node.module.split(".")[0])
    return names


def test_example_imports_only_allowed_modules():
    """The example imports only the standard library path module, NumPy and the two repository packages, never matplotlib."""
    names = imported_module_names(Path(run_example.__file__))
    assert names <= ALLOWED_MODULES, names
    assert "matplotlib" not in names


def constants_of(path):
    """Return the top-level constants of a Python file whose names are in SHARED_CONSTANTS, read from its source text."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in SHARED_CONSTANTS:
                values[node.targets[0].id] = ast.literal_eval(node.value)
    return values


def test_seed_and_limits_equal_the_generator_examples():
    """The example's seed and size limits equal the constants in the generator example's source, and the module values agree."""
    generator = constants_of(GENERATOR_EXAMPLE)
    placement = constants_of(Path(run_example.__file__))
    assert sorted(generator) == sorted(SHARED_CONSTANTS)
    assert placement == generator
    assert run_example.SEED == generator["SEED"]
    assert run_example.SIEVE_LOWER_UM == generator["SIEVE_LOWER_UM"]
    assert run_example.SIEVE_UPPER_UM == generator["SIEVE_UPPER_UM"]


def test_guide_is_reached_from_the_root_readme():
    """The top-level README links the guide, and the guide file exists."""
    text = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")
    assert GUIDE_LINK in text
    assert GUIDE_PATH.is_file()


def test_guide_holds_the_command_and_every_setting_column_and_file():
    """The guide has the run command, the generator guide link, every setting and column word and every file name the example writes."""
    guide = GUIDE_PATH.read_text(encoding="utf-8")
    assert RUN_COMMAND in guide
    assert "](../synthetic_data_generation/README.md)" in guide
    for word in SETTING_WORDS + COLUMN_WORDS + EXPECTED_FILES:
        assert word in guide, word


def test_guide_names_only_existing_paths_or_written_files():
    """Every placement_simulation/ path in the guide exists, or is the output folder or a file the example writes there."""
    guide = GUIDE_PATH.read_text(encoding="utf-8")
    found = re.findall(r"placement_simulation/[A-Za-z0-9_./-]+", guide)
    assert len(found) > 0
    output_prefix = "placement_simulation/example/output/"
    for text in found:
        cleaned = text.rstrip(".,:;")
        if cleaned == output_prefix:
            continue
        if cleaned.startswith(output_prefix) and cleaned[len(output_prefix):] in EXPECTED_FILES:
            continue
        assert (REPOSITORY_ROOT / cleaned).exists(), cleaned
