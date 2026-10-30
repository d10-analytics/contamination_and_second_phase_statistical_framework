# This file tests the command line and the public imports by running them the
# way a user would: main([...]) with real files in a temporary folder, plus one
# separate Python process for "python -m synthetic_data_generation". The save
# dialog is the one thing a test cannot click, so tkinter's Tk window and its
# save-file function are replaced by small stand-ins that record how they were
# called. On a Python without Tk, a small stand-in tkinter is installed first, so
# every test here still runs. A "summary" is the text file every run writes beside its CSV; it holds
# every setting needed to repeat the run exactly.

import importlib
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

import synthetic_data_generation
from synthetic_data_generation import cli
from synthetic_data_generation import dataset
from synthetic_data_generation import entries as entries_module
from synthetic_data_generation import outputs

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent.parent
THIS_TEST_NAME = "test_the_command_line_tests_collect_and_run_without_a_real_tkinter"
BEGIN_MARKER = "--- BEGIN REPRODUCTION PARAMETERS ---"
END_MARKER = "--- END REPRODUCTION PARAMETERS ---"

SIMPLE_TEXT = '[{"class": "A", "distribution": "gaussian", "N": 4, "mean": 50, "stdev": 3}]'
MIXED_ENTRIES = [
    {"class": "A", "distribution": "gaussian", "N": 6, "mean": 50, "stdev": 5},
    {"class": "B", "distribution": "gamma", "N": 5, "shape": 9, "scale": 5.5},
    {"class": "C", "distribution": "gev", "N": 5, "shape": 0.1, "loc": 50, "scale": 4},
    {"class": "A", "distribution": "gaussian", "N": 3, "mean": 52, "stdev": 4},
]
IMPOSSIBLE_TEXT = '[{"class": "A", "distribution": "gaussian", "N": 2, "mean": 10, "stdev": 1}]'
IMPOSSIBLE_ARGUMENTS = ["--threshold-lower", "500", "--threshold-upper", "600", "--max-attempts", "20"]


class StandInTclError(Exception):
    """Stands in for the error Tk raises when it cannot open a window."""


def stand_in_window():
    """Pretend to open a window; the dialog tests replace this before use."""
    raise StandInTclError("the stand-in tkinter cannot open a window")


def stand_in_dialog(title=None, confirmoverwrite=True, defaultextension="", filetypes=()):
    """Pretend to show the save dialog; the dialog tests replace this before use."""
    raise StandInTclError("the stand-in tkinter cannot show a dialog")


def load_tkinter():
    """Return the real tkinter, or install and return a small stand-in when it is missing."""
    try:
        real_module = importlib.import_module("tkinter")
        importlib.import_module("tkinter.filedialog")
    except ImportError:
        stand_in = types.ModuleType("tkinter")
        stand_in_dialogs = types.ModuleType("tkinter.filedialog")
        stand_in.Tk = stand_in_window
        stand_in.TclError = StandInTclError
        stand_in.filedialog = stand_in_dialogs
        stand_in_dialogs.asksaveasfilename = stand_in_dialog
        sys.modules["tkinter"] = stand_in
        sys.modules["tkinter.filedialog"] = stand_in_dialogs
        return stand_in
    return real_module


tkinter = load_tkinter()


def mixed_text():
    """Return the mixed entries as JSON text."""
    return json.dumps(MIXED_ENTRIES)


def folder_names(folder):
    """Return the sorted names of everything in a folder."""
    return sorted(path.name for path in folder.iterdir())


def run_main(arguments, capsys):
    """Run cli.main and return (exit code, text printed on stderr)."""
    capsys.readouterr()
    code = cli.main(arguments)
    return code, capsys.readouterr().err


def write_run(tmp_path, capsys, name, extra_arguments):
    """Run the mixed entries with the given extra flags and return the CSV path."""
    csv_path = tmp_path / f"{name}.csv"
    code, _ = run_main(["--input-json", mixed_text(), "--output", str(csv_path)] + extra_arguments, capsys)
    assert code == 0
    return csv_path


def summary_of(csv_path):
    """Return the summary path that goes with a CSV path."""
    return csv_path.with_name(csv_path.stem + "_summary.txt")


def edit_block(summary_path, changes):
    """Rewrite the reproduction block of a summary with the given key changes."""
    lines = summary_path.read_text(encoding="utf-8").split("\n")
    begin = lines.index(BEGIN_MARKER)
    end = lines.index(END_MARKER)
    block = json.loads("\n".join(lines[begin + 1 : end]))
    block.update(changes)
    lines[begin + 1 : end] = [json.dumps(block)]
    summary_path.write_text("\n".join(lines), encoding="utf-8")


class StubRoot:
    """A stand-in for the hidden tkinter window that records that it was closed."""

    def withdraw(self):
        """Pretend to hide the window."""

    def destroy(self):
        """Pretend to close the window."""


def install_dialog(monkeypatch, answer):
    """Replace tkinter's window and save dialog; return the list of dialog calls."""
    calls = []

    def stub_dialog(title=None, confirmoverwrite=True, defaultextension="", filetypes=()):
        """Record how the dialog was called and return the prepared answer."""
        calls.append(
            {
                "confirmoverwrite": confirmoverwrite,
                "defaultextension": defaultextension,
                "filetypes": filetypes,
            }
        )
        return answer

    monkeypatch.setattr(tkinter, "Tk", StubRoot)
    monkeypatch.setattr(tkinter.filedialog, "asksaveasfilename", stub_dialog)
    return calls


def test_public_functions_import_from_the_package():
    """The four public functions are importable from the package itself."""
    assert synthetic_data_generation.generate_dataset is dataset.generate_dataset
    assert synthetic_data_generation.write_outputs is outputs.write_outputs
    assert synthetic_data_generation.parse_entries is entries_module.parse_entries
    assert synthetic_data_generation.read_entries is entries_module.read_entries


def test_text_list_and_file_entries_give_equal_rows(tmp_path):
    """JSON text, the equal list and read_entries output give the same rows for one seed."""
    entries_file = tmp_path / "entries.json"
    entries_file.write_text(mixed_text(), encoding="utf-8")
    from_text = synthetic_data_generation.generate_dataset(mixed_text(), seed=5)
    from_list = synthetic_data_generation.generate_dataset(MIXED_ENTRIES, seed=5)
    from_file = synthetic_data_generation.generate_dataset(
        synthetic_data_generation.read_entries(entries_file), seed=5
    )
    assert from_text.rows == from_list.rows == from_file.rows


def test_an_invalid_list_and_its_text_are_rejected_with_the_same_message():
    """The list and text forms of one bad entry list raise the same message."""
    bad = [{"class": "A", "distribution": "gaussian", "N": 0, "mean": 50, "stdev": 3}]
    with pytest.raises(ValueError) as from_list:
        synthetic_data_generation.generate_dataset(bad)
    with pytest.raises(ValueError) as from_text:
        synthetic_data_generation.generate_dataset(json.dumps(bad))
    assert str(from_list.value) == str(from_text.value)


def test_replay_of_a_seeded_shuffled_run_is_byte_identical(tmp_path, capsys):
    """A seeded, shuffled, thresholded run with three distributions replays to the same CSV bytes."""
    original = write_run(
        tmp_path,
        capsys,
        "original",
        ["--seed", "11", "--shuffle", "--threshold-lower", "35", "--threshold-upper", "65"],
    )
    replay = tmp_path / "replay.csv"
    code, _ = run_main(["--from-summary", str(summary_of(original)), "--output", str(replay)], capsys)
    assert code == 0
    assert replay.read_bytes() == original.read_bytes()
    assert summary_of(replay).exists()
    assert "Inputs: " + repr(str(summary_of(original).resolve())) in summary_of(replay).read_text(encoding="utf-8")


def test_replay_of_a_run_without_a_seed_is_byte_identical(tmp_path, capsys):
    """The seed chosen for a seedless run is recorded and reproduces the CSV."""
    original = write_run(tmp_path, capsys, "original", [])
    replay = tmp_path / "replay.csv"
    code, _ = run_main(["--from-summary", str(summary_of(original)), "--output", str(replay)], capsys)
    assert code == 0
    assert replay.read_bytes() == original.read_bytes()


def test_every_generation_flag_is_rejected_with_from_summary(tmp_path, capsys):
    """Each generation flag, even at its default or a falsy value, stops a replay and writes nothing."""
    original = write_run(tmp_path, capsys, "original", ["--seed", "3"])
    summary_path = summary_of(original)
    files_before = folder_names(tmp_path)
    flag_sets = [
        ["--input-json", SIMPLE_TEXT],
        ["--input-file", str(original)],
        ["--seed", "0"],
        ["--max-attempts", "100000"],
        ["--threshold-lower", "0"],
        ["--threshold-upper", "5"],
        ["--shuffle"],
    ]
    for flags in flag_sets:
        replay = tmp_path / "replay.csv"
        code, error_text = run_main(["--from-summary", str(summary_path), "--output", str(replay)] + flags, capsys)
        assert code != 0, flags
        assert error_text != "", flags
        assert folder_names(tmp_path) == files_before, flags


def test_replay_accepts_output_and_overwrite(tmp_path, capsys):
    """--output and --overwrite are the only other flags a replay takes."""
    original = write_run(tmp_path, capsys, "original", ["--seed", "3"])
    target = tmp_path / "target.csv"
    target.write_text("old", encoding="utf-8")
    arguments = ["--from-summary", str(summary_of(original)), "--output", str(target)]
    code, error_text = run_main(arguments, capsys)
    assert code != 0
    assert "already exists" in error_text
    code, _ = run_main(arguments + ["--overwrite"], capsys)
    assert code == 0
    assert target.read_bytes() == original.read_bytes()


def test_edited_library_versions_warn_but_still_reproduce(tmp_path, capsys):
    """A summary written by other library versions prints a warning, exits 0 and matches byte for byte."""
    original = write_run(tmp_path, capsys, "original", ["--seed", "3"])
    edit_block(summary_of(original), {"library_versions": {"python": "0.0.0", "numpy": "0.0.0", "scipy": "0.0.0"}})
    replay = tmp_path / "replay.csv"
    code, error_text = run_main(["--from-summary", str(summary_of(original)), "--output", str(replay)], capsys)
    assert code == 0
    assert "warning" in error_text
    assert "numpy 0.0.0" in error_text
    assert replay.read_bytes() == original.read_bytes()


def test_a_malformed_summary_fails_and_writes_nothing(tmp_path, capsys):
    """A summary without the reproduction block exits non-zero with a message and writes no file."""
    broken = tmp_path / "broken_summary.txt"
    broken.write_text("this is not a summary\n", encoding="utf-8")
    code, error_text = run_main(["--from-summary", str(broken), "--output", str(tmp_path / "out.csv")], capsys)
    assert code != 0
    assert error_text.startswith("error:")
    assert folder_names(tmp_path) == ["broken_summary.txt"]


def test_a_missing_summary_fails_with_a_message(tmp_path, capsys):
    """Naming a summary that does not exist is reported, not a traceback."""
    code, error_text = run_main(["--from-summary", str(tmp_path / "none.txt"), "--output", str(tmp_path / "out.csv")], capsys)
    assert code != 0
    assert "none.txt" in error_text
    assert folder_names(tmp_path) == []


def test_replay_goes_through_the_shared_replay_function(tmp_path, capsys, monkeypatch):
    """Replacing only the outputs module's replay function makes --from-summary fail with its error."""
    original = write_run(tmp_path, capsys, "original", ["--seed", "3"])

    def failing_replay(summary_path):
        """Raise a recognisable error instead of replaying."""
        raise ValueError("marker-replay-error")

    monkeypatch.setattr(outputs, "replay_summary", failing_replay)
    code, error_text = run_main(["--from-summary", str(summary_of(original)), "--output", str(tmp_path / "r.csv")], capsys)
    assert code != 0
    assert "marker-replay-error" in error_text
    assert not (tmp_path / "r.csv").exists()


def test_input_json_and_input_file_write_identical_csvs(tmp_path, capsys):
    """Equal content and seed give equal CSV bytes, and the file run's summary names the file."""
    entries_file = tmp_path / "entries.jsonl"
    entries_file.write_text(mixed_text(), encoding="utf-8")
    from_text = tmp_path / "from_text.csv"
    from_file = tmp_path / "from_file.csv"
    assert run_main(["--input-json", mixed_text(), "--seed", "2", "--output", str(from_text)], capsys)[0] == 0
    assert run_main(["--input-file", str(entries_file), "--seed", "2", "--output", str(from_file)], capsys)[0] == 0
    assert from_text.read_bytes() == from_file.read_bytes()
    file_summary = summary_of(from_file).read_text(encoding="utf-8")
    assert "Inputs: " + repr(str(entries_file.resolve())) in file_summary
    assert "Inputs: entries given as text" in summary_of(from_text).read_text(encoding="utf-8")


def test_an_entry_file_named_text_or_list_shows_its_path(tmp_path, capsys, monkeypatch):
    """A relative file name equal to an API marker word is still recorded as its absolute path."""
    monkeypatch.chdir(tmp_path)
    for name in ("text", "list"):
        Path(name).write_text(SIMPLE_TEXT, encoding="utf-8")
        csv_name = f"{name}_run.csv"
        assert run_main(["--input-file", name, "--seed", "1", "--output", csv_name], capsys)[0] == 0
        summary = Path(f"{name}_run_summary.txt").read_text(encoding="utf-8")
        assert "Inputs: " + repr(str((tmp_path / name).resolve())) in summary
        assert "entries given as" not in summary


def test_exactly_one_input_form_is_required(tmp_path, capsys):
    """No input form, or two together, exits non-zero and writes nothing."""
    original = write_run(tmp_path, capsys, "original", ["--seed", "3"])
    files_before = folder_names(tmp_path)
    output = str(tmp_path / "new.csv")
    attempts = [
        ["--output", output],
        ["--input-json", SIMPLE_TEXT, "--input-file", str(original), "--output", output],
        ["--input-json", SIMPLE_TEXT, "--from-summary", str(summary_of(original)), "--output", output],
        ["--input-file", str(original), "--from-summary", str(summary_of(original)), "--output", output],
    ]
    for arguments in attempts:
        code, error_text = run_main(arguments, capsys)
        assert code != 0, arguments
        assert error_text != "", arguments
        assert folder_names(tmp_path) == files_before, arguments


def test_the_parser_leaves_omitted_generation_flags_as_none():
    """Omitted generation flags stay None, so the parser spells no default value."""
    args = cli.build_parser().parse_args(["--input-json", SIMPLE_TEXT])
    assert args.seed is None
    assert args.max_attempts is None
    assert args.threshold_lower is None
    assert args.threshold_upper is None
    assert args.shuffle is None
    assert cli.build_parser().parse_args(["--input-json", SIMPLE_TEXT, "--shuffle"]).shuffle is True


def test_a_run_without_settings_records_the_dataset_defaults(tmp_path, capsys):
    """Omitted settings are recorded as 100,000 attempts, no shuffle and no thresholds."""
    csv_path = tmp_path / "run.csv"
    assert run_main(["--input-json", SIMPLE_TEXT, "--seed", "1", "--output", str(csv_path)], capsys)[0] == 0
    parameters = outputs.read_summary(summary_of(csv_path))
    assert parameters.max_attempts == 100_000
    assert parameters.max_attempts == dataset.DEFAULT_MAX_ATTEMPTS
    assert parameters.shuffle is False
    assert parameters.threshold_lower is None
    assert parameters.threshold_upper is None


def test_invalid_seed_and_max_attempts_give_the_generate_dataset_message(tmp_path, capsys):
    """--seed -1 and --max-attempts 0 exit non-zero with the message the API raises."""
    cases = [
        (["--seed", "-1"], -1, 100),
        (["--max-attempts", "0"], 1, 0),
    ]
    for flags, api_seed, api_max_attempts in cases:
        with pytest.raises(ValueError) as api_error:
            synthetic_data_generation.generate_dataset(SIMPLE_TEXT, seed=api_seed, max_attempts=api_max_attempts)
        code, error_text = run_main(["--input-json", SIMPLE_TEXT, "--output", str(tmp_path / "x.csv")] + flags, capsys)
        assert code != 0
        assert str(api_error.value) in error_text
    assert folder_names(tmp_path) == []


def test_a_run_without_a_seed_prints_the_chosen_seed(tmp_path, capsys):
    """The seed chosen for a seedless run appears on stderr and equals the summary's seed."""
    csv_path = tmp_path / "run.csv"
    code, error_text = run_main(["--input-json", SIMPLE_TEXT, "--output", str(csv_path)], capsys)
    assert code == 0
    recorded = outputs.read_summary(summary_of(csv_path)).seed
    assert f"seed used: {recorded}" in error_text
    seeded = tmp_path / "seeded.csv"
    code, error_text = run_main(["--input-json", SIMPLE_TEXT, "--seed", "4", "--output", str(seeded)], capsys)
    assert code == 0
    assert "seed used" not in error_text


def test_the_python_interface_prints_nothing(capsys):
    """generate_dataset never prints the seed it chooses."""
    capsys.readouterr()
    synthetic_data_generation.generate_dataset(SIMPLE_TEXT)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_invalid_input_and_a_sampling_failure_exit_non_zero(tmp_path, capsys):
    """Bad entries and an unreachable window each print one error message and exit 1."""
    output = str(tmp_path / "x.csv")
    code, error_text = run_main(["--input-json", "[", "--output", output], capsys)
    assert code == 1
    assert error_text.startswith("error:")
    code, error_text = run_main(["--input-file", str(tmp_path / "missing.json"), "--output", output], capsys)
    assert code == 1
    assert "missing.json" in error_text
    code, error_text = run_main(["--input-json", IMPOSSIBLE_TEXT, "--output", output] + IMPOSSIBLE_ARGUMENTS, capsys)
    assert code == 1
    assert "attempts" in error_text
    assert folder_names(tmp_path) == []


def test_python_dash_m_runs_the_generator_from_the_repository_root(tmp_path):
    """A separate Python process runs the package with --input-file and --output and writes both files."""
    entries_file = tmp_path / "entries.json"
    entries_file.write_text(SIMPLE_TEXT, encoding="utf-8")
    csv_path = tmp_path / "out.csv"
    completed = subprocess.run(
        [sys.executable, "-m", "synthetic_data_generation", "--input-file", str(entries_file), "--output", str(csv_path)],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert csv_path.exists()
    assert summary_of(csv_path).exists()


def test_a_failing_run_through_python_dash_m_returns_a_non_zero_code(tmp_path):
    """The module entry point exits with the command line's failure code."""
    completed = subprocess.run(
        [sys.executable, "-m", "synthetic_data_generation", "--input-json", "[", "--output", str(tmp_path / "o.csv")],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 1
    assert "error:" in completed.stderr


def test_a_run_with_output_never_loads_tkinter(tmp_path):
    """In a fresh process, a run given --output finishes without tkinter ever being loaded."""
    script = (
        "import sys\n"
        "from synthetic_data_generation import cli\n"
        "code = cli.main(['--input-json', sys.argv[1], '--output', sys.argv[2]])\n"
        "loaded = 'tkinter' in sys.modules\n"
        "sys.exit(code or (5 if loaded else 0))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script, SIMPLE_TEXT, str(tmp_path / "o.csv")],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_blocked_tkinter_does_not_stop_a_run_with_output(tmp_path, capsys, monkeypatch):
    """With tkinter imports made to fail, a run given --output still succeeds."""
    monkeypatch.setitem(sys.modules, "tkinter", None)
    monkeypatch.setitem(sys.modules, "tkinter.filedialog", None)
    code, _ = run_main(["--input-json", SIMPLE_TEXT, "--seed", "1", "--output", str(tmp_path / "o.csv")], capsys)
    assert code == 0
    assert (tmp_path / "o.csv").exists()


def test_the_dialog_is_asked_with_the_documented_options_and_its_path_is_used(tmp_path, capsys, monkeypatch):
    """A returned path names both files, and the dialog gets the CSV type and no overwrite prompt."""
    chosen = tmp_path / "chosen.csv"
    calls = install_dialog(monkeypatch, str(chosen))
    code, _ = run_main(["--input-json", SIMPLE_TEXT, "--seed", "1"], capsys)
    assert code == 0
    assert len(calls) == 1
    assert calls[0]["confirmoverwrite"] is False
    assert calls[0]["defaultextension"] == ".csv"
    assert ("CSV files", "*.csv") in calls[0]["filetypes"]
    assert folder_names(tmp_path) == ["chosen.csv", "chosen_summary.txt"]


def test_input_and_setting_errors_come_before_the_dialog(tmp_path, capsys, monkeypatch):
    """Malformed input and invalid settings are reported without ever opening the dialog."""
    calls = install_dialog(monkeypatch, str(tmp_path / "chosen.csv"))
    cases = [
        (["--input-json", "["], "JSON"),
        (["--input-json", SIMPLE_TEXT, "--threshold-lower", "10", "--threshold-upper", "5"], "threshold_lower"),
        (["--input-json", SIMPLE_TEXT, "--seed", "-1"], "seed"),
        (["--input-json", SIMPLE_TEXT, "--max-attempts", "0"], "max_attempts"),
    ]
    for arguments, expected in cases:
        code, error_text = run_main(arguments, capsys)
        assert code == 1, arguments
        assert expected in error_text, arguments
    assert calls == []
    assert folder_names(tmp_path) == []


def test_a_summary_with_inverted_thresholds_is_rejected_before_the_dialog(tmp_path, capsys, monkeypatch):
    """Reading a summary whose block was edited to inverted thresholds fails before any dialog."""
    original = write_run(tmp_path, capsys, "original", ["--seed", "3"])
    edit_block(summary_of(original), {"threshold_lower": 10, "threshold_upper": 5})
    calls = install_dialog(monkeypatch, str(tmp_path / "chosen.csv"))
    code, error_text = run_main(["--from-summary", str(summary_of(original))], capsys)
    assert code == 1
    assert "threshold_lower" in error_text
    assert calls == []
    assert not (tmp_path / "chosen.csv").exists()


def test_an_impossible_window_opens_the_dialog_then_fails_leaving_the_folder_empty(tmp_path, capsys, monkeypatch):
    """The dialog comes before generation, so an unreachable window asks first and then fails cleanly."""
    chosen_folder = tmp_path / "chosen"
    chosen_folder.mkdir()
    calls = install_dialog(monkeypatch, str(chosen_folder / "out.csv"))
    code, error_text = run_main(["--input-json", IMPOSSIBLE_TEXT] + IMPOSSIBLE_ARGUMENTS, capsys)
    assert code == 1
    assert "attempts" in error_text
    assert len(calls) == 1
    assert folder_names(chosen_folder) == []


def test_a_replay_with_the_dialog_uses_the_chosen_path(tmp_path, capsys, monkeypatch):
    """--from-summary without --output asks through the dialog and replays to that path."""
    original = write_run(tmp_path, capsys, "original", ["--seed", "3"])
    chosen = tmp_path / "chosen.csv"
    calls = install_dialog(monkeypatch, str(chosen))
    code, _ = run_main(["--from-summary", str(summary_of(original))], capsys)
    assert code == 0
    assert len(calls) == 1
    assert chosen.read_bytes() == original.read_bytes()


def test_a_cancelled_dialog_exits_non_zero_and_writes_nothing(tmp_path, capsys, monkeypatch):
    """An empty path from the dialog is reported and nothing is written."""
    install_dialog(monkeypatch, "")
    code, error_text = run_main(["--input-json", SIMPLE_TEXT, "--seed", "1"], capsys)
    assert code == 1
    assert "no output file was chosen" in error_text
    assert folder_names(tmp_path) == []


def test_a_missing_display_exits_non_zero_and_writes_nothing(tmp_path, capsys, monkeypatch):
    """When Tk cannot open a window, the run reports it and writes nothing."""

    def failing_window():
        """Raise the error Tk raises when no display exists."""
        raise tkinter.TclError("no display name and no $DISPLAY environment variable")

    monkeypatch.setattr(tkinter, "Tk", failing_window)
    code, error_text = run_main(["--input-json", SIMPLE_TEXT, "--seed", "1"], capsys)
    assert code == 1
    assert "no display" in error_text
    assert folder_names(tmp_path) == []


def test_an_existing_csv_is_reported_before_the_sampling_error(tmp_path, capsys):
    """With an existing CSV and no --overwrite, the already-exists error wins and the bytes stay."""
    existing = tmp_path / "out.csv"
    existing.write_bytes(b"keep me")
    code, error_text = run_main(
        ["--input-json", IMPOSSIBLE_TEXT, "--output", str(existing)] + IMPOSSIBLE_ARGUMENTS, capsys
    )
    assert code == 1
    assert "already exists" in error_text
    assert "attempts" not in error_text
    assert existing.read_bytes() == b"keep me"
    assert folder_names(tmp_path) == ["out.csv"]


def test_an_existing_summary_is_reported_before_the_sampling_error(tmp_path, capsys):
    """With only the summary present, the already-exists error wins, the summary stays and no CSV appears."""
    existing = tmp_path / "out_summary.txt"
    existing.write_bytes(b"keep me")
    code, error_text = run_main(
        ["--input-json", IMPOSSIBLE_TEXT, "--output", str(tmp_path / "out.csv")] + IMPOSSIBLE_ARGUMENTS, capsys
    )
    assert code == 1
    assert "already exists" in error_text
    assert existing.read_bytes() == b"keep me"
    assert folder_names(tmp_path) == ["out_summary.txt"]


def test_a_folder_at_the_summary_path_is_reported_even_with_overwrite(tmp_path, capsys):
    """A directory at the summary path is reported before generation and no CSV is written."""
    (tmp_path / "out_summary.txt").mkdir()
    code, error_text = run_main(
        ["--input-json", IMPOSSIBLE_TEXT, "--output", str(tmp_path / "out.csv"), "--overwrite"]
        + IMPOSSIBLE_ARGUMENTS,
        capsys,
    )
    assert code == 1
    assert "is a folder" in error_text
    assert "attempts" not in error_text
    assert folder_names(tmp_path) == ["out_summary.txt"]


def make_impossible_summary(tmp_path, capsys):
    """Write a run, then edit its summary so that replaying it can never keep a particle."""
    original = write_run(tmp_path, capsys, "original", ["--seed", "3"])
    edit_block(
        summary_of(original),
        {
            "entries": json.loads(IMPOSSIBLE_TEXT),
            "threshold_lower": 500,
            "threshold_upper": 600,
            "max_attempts": 20,
        },
    )
    return summary_of(original)


def test_a_replay_opens_the_dialog_before_it_draws(tmp_path, capsys, monkeypatch):
    """A replay that cannot keep a particle still asks for the path first, then fails cleanly."""
    summary_path = make_impossible_summary(tmp_path, capsys)
    chosen_folder = tmp_path / "chosen"
    chosen_folder.mkdir()
    calls = install_dialog(monkeypatch, str(chosen_folder / "out.csv"))
    code, error_text = run_main(["--from-summary", str(summary_path)], capsys)
    assert code == 1
    assert "attempts" in error_text
    assert len(calls) == 1
    assert folder_names(chosen_folder) == []


def test_a_replay_checks_its_destination_before_it_draws(tmp_path, capsys):
    """A replay onto an existing CSV reports already-exists, not the sampling error, and keeps the bytes."""
    summary_path = make_impossible_summary(tmp_path, capsys)
    existing = tmp_path / "taken.csv"
    existing.write_bytes(b"keep me")
    code, error_text = run_main(["--from-summary", str(summary_path), "--output", str(existing)], capsys)
    assert code == 1
    assert "already exists" in error_text
    assert "attempts" not in error_text
    assert existing.read_bytes() == b"keep me"
    assert not summary_of(existing).exists()


def test_a_missing_tkinter_asks_for_output_and_writes_nothing(tmp_path, capsys, monkeypatch):
    """Without tkinter and without --output, the run reports one message naming --output and writes nothing."""
    monkeypatch.setitem(sys.modules, "tkinter", None)
    monkeypatch.setitem(sys.modules, "tkinter.filedialog", None)
    code, error_text = run_main(["--input-json", SIMPLE_TEXT, "--seed", "1"], capsys)
    assert code == 1
    assert error_text.startswith("error:")
    assert "--output" in error_text
    assert "Traceback" not in error_text
    assert folder_names(tmp_path) == []


def test_the_command_line_tests_collect_and_run_without_a_real_tkinter():
    """With tkinter blocked in a fresh process, this module still collects and its other tests pass."""
    script = (
        "import sys\n"
        "sys.modules['tkinter'] = None\n"
        "sys.modules['tkinter.filedialog'] = None\n"
        "import pytest\n"
        "sys.exit(pytest.main(['-q', '-p', 'no:cacheprovider', '-k', 'not ' + sys.argv[2], sys.argv[1]]))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script, str(Path(__file__).resolve()), THIS_TEST_NAME],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "error during collection" not in completed.stdout
