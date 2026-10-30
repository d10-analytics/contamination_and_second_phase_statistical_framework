# This file tests the runnable example and the step-by-step guide. The example
# test runs the real script end to end in a temporary folder with no screen
# available, then checks the files, the plot function on a hand-written CSV, and
# that step 5 really goes through the shared replay function. The guide tests
# read the guide the way a newcomer would: following the link from the top-level
# README, finding the numbered steps in order, checking every path it names, and
# running its command-line line. Matplotlib is needed, so these tests belong to
# the full lane only.

import ast
import importlib.util
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from synthetic_data_generation import dataset
from synthetic_data_generation import entries as entries_module
from synthetic_data_generation import outputs
from synthetic_data_generation.example import run_example

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent.parent
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
EXPECTED_FILES = ["example_particles.csv", "example_particles_plot.png", "example_particles_summary.txt"]
HAND_WRITTEN_CSV = (
    "feature_label,object_id,contour_area_um2,equivalent_diameter_um,volume_um3\n"
    "X,C1_o00001,1.0,10.0,500.0\n"
    "X,C1_o00002,2.0,20.0,600.0\n"
    "Y,C2_o00001,3.0,30.0,700.0\n"
    "X,C1_o00003,4.0,40.0,800.0\n"
)
ALLOWED_MODULES = {"matplotlib", "synthetic_data_generation"}
CLI_LINE_START = "python3 -m synthetic_data_generation "
MARKER_TEXT = "replay marker"


class ReplayMarkerError(Exception):
    """Raised by a stand-in replay function so a test can see it reach the example."""


def hide_display(monkeypatch):
    """Remove the variables that tell a program a screen exists."""
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)


def printed_lines(capsys):
    """Return the lines printed so far on stdout."""
    return capsys.readouterr().out.splitlines()


def test_example_writes_exactly_three_files_and_reports_identical(tmp_path, capsys, monkeypatch):
    """A run into a folder that does not exist yet creates it, leaves the three files and ends with identical: True."""
    hide_display(monkeypatch)
    folder = tmp_path / "new_folder" / "output"
    run_example.main(folder)
    assert sorted(path.name for path in folder.iterdir()) == EXPECTED_FILES
    plot_bytes = (folder / "example_particles_plot.png").read_bytes()
    assert len(plot_bytes) > len(PNG_SIGNATURE)
    assert plot_bytes.startswith(PNG_SIGNATURE)
    lines = printed_lines(capsys)
    assert lines[-1] == "identical: True"
    assert "'matrix': 300" in lines[0]


def test_example_uses_its_fixed_seed_and_sieve_limits(tmp_path, monkeypatch):
    """The written summary records the script's seed and limits, every diameter is inside them, and two runs match byte for byte."""
    hide_display(monkeypatch)
    first = tmp_path / "first"
    second = tmp_path / "second"
    run_example.main(first)
    run_example.main(second)
    parameters = outputs.read_summary(outputs.summary_path_for(first / "example_particles.csv"))
    assert parameters.seed == run_example.SEED == 20261029
    assert parameters.threshold_lower == run_example.SIEVE_LOWER_UM == 10.0
    assert parameters.threshold_upper == run_example.SIEVE_UPPER_UM == 120.0
    diameters_by_class = run_example.read_csv_back(first / "example_particles.csv")
    for class_name, diameters in diameters_by_class.items():
        for diameter in diameters:
            assert 10.0 <= diameter <= 120.0, class_name
    assert (first / "example_particles.csv").read_bytes() == (second / "example_particles.csv").read_bytes()


def test_module_entry_point_runs_from_the_repository_root(monkeypatch):
    """Running the module the way the guide says exits 0, prints identical: True and fills the ignored output folder."""
    hide_display(monkeypatch)
    command = [sys.executable, "-m", "synthetic_data_generation.example.run_example"]
    finished = subprocess.run(command, cwd=REPOSITORY_ROOT, capture_output=True, text=True, env=os.environ.copy())
    assert finished.returncode == 0, finished.stderr
    assert finished.stdout.splitlines()[-1] == "identical: True"
    output_folder = REPOSITORY_ROOT / "synthetic_data_generation" / "example" / "output"
    assert sorted(path.name for path in output_folder.iterdir()) == EXPECTED_FILES


def test_example_can_run_twice_into_the_same_folder(tmp_path, capsys, monkeypatch):
    """A second run replaces the files and still ends with identical: True."""
    hide_display(monkeypatch)
    run_example.main(tmp_path)
    capsys.readouterr()
    run_example.main(tmp_path)
    assert printed_lines(capsys)[-1] == "identical: True"
    assert sorted(path.name for path in tmp_path.iterdir()) == EXPECTED_FILES


def test_example_input_holds_three_classes_in_order():
    """The bundled input file reads as gamma, gaussian, gev entries for three different classes."""
    entries = entries_module.read_entries(run_example.INPUT_PATH)
    assert [entry["distribution"] for entry in entries] == ["gamma", "gaussian", "gev"]
    assert len({entry["class"] for entry in entries}) == 3


def test_plot_function_reads_the_csv_file_it_is_given(tmp_path, monkeypatch):
    """Given only a hand-written CSV path, the plot function returns the counts in that file and saves a PNG beside it."""
    hide_display(monkeypatch)
    csv_path = tmp_path / "hand.csv"
    csv_path.write_text(HAND_WRITTEN_CSV, encoding="utf-8")
    counts = run_example.plot_csv(csv_path)
    assert counts == {"X": 3, "Y": 1}
    assert (tmp_path / "hand_plot.png").read_bytes().startswith(PNG_SIGNATURE)


def test_step_five_prints_false_when_the_shared_replay_gives_another_dataset(tmp_path, capsys, monkeypatch):
    """A replay function returning a dataset from a different seed makes the example print identical: False."""
    hide_display(monkeypatch)

    def other_seed_replay(summary_path):
        """Return a dataset made from the example's entries but with a different seed."""
        entries = entries_module.read_entries(run_example.INPUT_PATH)
        return dataset.generate_dataset(
            entries,
            threshold_lower=run_example.SIEVE_LOWER_UM,
            threshold_upper=run_example.SIEVE_UPPER_UM,
            seed=run_example.SEED + 1,
        )

    monkeypatch.setattr(outputs, "replay_summary", other_seed_replay)
    run_example.main(tmp_path)
    assert printed_lines(capsys)[-1] == "identical: False"


def test_step_five_fails_with_the_error_of_the_shared_replay(tmp_path, monkeypatch):
    """A replay function that raises a marker error makes main fail with that marker."""
    hide_display(monkeypatch)

    def raising_replay(summary_path):
        """Raise the marker error instead of replaying."""
        raise ReplayMarkerError(MARKER_TEXT)

    monkeypatch.setattr(outputs, "replay_summary", raising_replay)
    with pytest.raises(ReplayMarkerError, match=MARKER_TEXT):
        run_example.main(tmp_path)


def test_example_script_imports_only_allowed_modules():
    """Every import in run_example.py is standard library, matplotlib or the package itself."""
    tree = ast.parse(Path(run_example.__file__).read_text(encoding="utf-8"))
    top_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top_names.add(alias.name.split(".")[0])
        if isinstance(node, ast.ImportFrom):
            assert node.level == 0
            top_names.add(node.module.split(".")[0])
    assert top_names
    for name in top_names:
        assert name in sys.stdlib_module_names or name in ALLOWED_MODULES, name


def test_non_display_backend_is_chosen_before_pyplot_is_imported():
    """In the script, matplotlib.use("Agg") comes before the pyplot import."""
    lines = Path(run_example.__file__).read_text(encoding="utf-8").splitlines()
    use_line = lines.index('matplotlib.use("Agg")')
    pyplot_line = lines.index("import matplotlib.pyplot as pyplot")
    assert use_line < pyplot_line


def read_guide():
    """Follow the link in the top-level README and return the guide's path and text."""
    root_text = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")
    found = re.search(r"\]\((synthetic_data_generation/README\.md)\)", root_text)
    assert found is not None
    guide_path = REPOSITORY_ROOT / found.group(1)
    return guide_path, guide_path.read_text(encoding="utf-8")


def split_steps(guide_text):
    """Return a list of (number, section text) for the guide's headings written like "## 2. Title"."""
    steps = []
    current_number = None
    current_lines = []
    for line in guide_text.splitlines():
        heading = re.match(r"## (\d+)\. ", line)
        if heading is not None or line.startswith("## "):
            if current_number is not None:
                steps.append((current_number, "\n".join(current_lines)))
            current_number = int(heading.group(1)) if heading is not None else None
            current_lines = []
        if current_number is not None:
            current_lines.append(line)
    if current_number is not None:
        steps.append((current_number, "\n".join(current_lines)))
    return steps


def test_guide_has_the_numbered_steps_in_order():
    """The guide's numbered steps appear in the order: Python check, environment, install, run, plot, command line."""
    guide_path, guide_text = read_guide()
    steps = split_steps(guide_text)
    assert [number for number, text in steps] == [1, 2, 3, 4, 5, 6]
    texts = [text for number, text in steps]
    assert "python3 --version" in texts[0]
    assert "3.10" in texts[0]
    assert "install a newer Python" in texts[0]
    venv_line = [line for line in texts[1].splitlines() if line.startswith("python3 -m venv ")]
    assert len(venv_line) == 1
    assert shlex.split(venv_line[0])[-1].startswith(("~/", "/"))
    assert "pip install -r synthetic_data_generation/requirements.txt" in texts[2]
    assert "python3 -m synthetic_data_generation.example.run_example" in texts[3]
    assert "example_particles_plot.png" in texts[4]
    assert CLI_LINE_START in texts[5]


def test_guide_paths_exist_and_output_names_match_written_files(tmp_path, monkeypatch):
    """Every repository path in the guide exists, and every output file name is one that main writes."""
    hide_display(monkeypatch)
    guide_path, guide_text = read_guide()
    run_example.main(tmp_path)
    output_prefix = "synthetic_data_generation/example/output/"
    checked_count = 0
    for found in re.findall(r"synthetic_data_generation/[\w./-]*", guide_text):
        path_text = found.rstrip(".,:;")
        if path_text == output_prefix:
            continue
        if path_text.startswith(output_prefix):
            assert path_text[len(output_prefix):] in EXPECTED_FILES, path_text
            assert (tmp_path / path_text[len(output_prefix):]).is_file()
        else:
            assert (REPOSITORY_ROOT / path_text).exists(), path_text
        checked_count += 1
    assert checked_count >= 5


def test_example_module_path_resolves_to_the_script():
    """The module path in the guide's run step finds example/run_example.py."""
    specification = importlib.util.find_spec("synthetic_data_generation.example.run_example")
    expected = REPOSITORY_ROOT / "synthetic_data_generation" / "example" / "run_example.py"
    assert Path(specification.origin).resolve() == expected.resolve()


def test_guide_windows_notes_name_py_and_scripts_activate():
    """The guide carries Windows notes that use the py launcher and Scripts\\activate."""
    guide_path, guide_text = read_guide()
    lines = guide_text.splitlines()
    assert any("py -m venv" in line for line in lines)
    assert any("Scripts\\activate" in line for line in lines)


def test_guide_command_line_step_runs_and_writes_both_files(tmp_path):
    """The guide's own command-line line, run with this Python and a temporary --output, exits 0 and writes both files."""
    guide_path, guide_text = read_guide()
    cli_lines = [line for line in guide_text.splitlines() if line.startswith(CLI_LINE_START)]
    assert len(cli_lines) == 1
    assert "--output" in cli_lines[0]
    words = shlex.split(cli_lines[0])
    words[0] = sys.executable
    words[words.index("--output") + 1] = str(tmp_path / "from_guide.csv")
    finished = subprocess.run(words, cwd=REPOSITORY_ROOT, capture_output=True, text=True)
    assert finished.returncode == 0, finished.stderr
    assert (tmp_path / "from_guide.csv").is_file()
    assert (tmp_path / "from_guide_summary.txt").is_file()


def test_guide_command_line_reproduces_the_example_csv(tmp_path, monkeypatch):
    """The guide's command-line line writes a CSV with the same bytes as the example's CSV."""
    hide_display(monkeypatch)
    run_example.main(tmp_path / "example")
    guide_path, guide_text = read_guide()
    cli_lines = [line for line in guide_text.splitlines() if line.startswith(CLI_LINE_START)]
    words = shlex.split(cli_lines[0])
    words[0] = sys.executable
    words[words.index("--output") + 1] = str(tmp_path / "from_guide.csv")
    finished = subprocess.run(words, cwd=REPOSITORY_ROOT, capture_output=True, text=True)
    assert finished.returncode == 0, finished.stderr
    example_bytes = (tmp_path / "example" / "example_particles.csv").read_bytes()
    assert (tmp_path / "from_guide.csv").read_bytes() == example_bytes


def test_guide_explains_input_file_risk_and_worst_case():
    """The guide mentions the input file keys, the two-save risk and the N x max_attempts worst case."""
    guide_path, guide_text = read_guide()
    assert "cannot hold a comment" in guide_text
    assert "N × `max_attempts`" in guide_text
    assert "beside the old summary" in guide_text
