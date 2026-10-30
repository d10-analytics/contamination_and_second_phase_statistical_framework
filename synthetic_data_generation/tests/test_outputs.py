# This file tests writing a dataset to disk and reading it back: the CSV bytes
# and numbers, the readable summary and its reproduction block, the checks on
# the output paths, the file permissions and the all-or-nothing commit. Every
# test triggers the behaviour through the real functions. Only two tests patch
# a call (the second temporary write and the summary rename), because a disk
# failure cannot be caused naturally in a test. A "umask" is the permission
# mask that decides the mode of newly created files.

import csv
import dataclasses
import io
import json
import os
import re
import stat

import numpy
import pytest

from synthetic_data_generation import dataset
from synthetic_data_generation import outputs

MARKER_NAME = "--- BEGIN REPRODUCTION PARAMETERS ---"


def gaussian(class_name, count=4):
    """Return a Gaussian entry dictionary."""
    return {"class": class_name, "distribution": "gaussian", "N": count, "mean": 50.0, "stdev": 3.0}


def make_result(
    entry_list=None, input_source=None, threshold_lower=None, threshold_upper=None, max_attempts=100_000, shuffle=False
):
    """Return a small seeded dataset result, optionally with another input source."""
    if entry_list is None:
        entry_list = [gaussian("A"), gaussian('say "hi", ok'), gaussian("A", 2)]
    result = dataset.generate_dataset(
        entry_list,
        threshold_lower=threshold_lower,
        threshold_upper=threshold_upper,
        seed=11,
        max_attempts=max_attempts,
        shuffle=shuffle,
    )
    if input_source is not None:
        result = dataset_with_source(result, input_source)
    return result


def dataset_with_source(result, input_source):
    """Return a copy of the result that records a different input source."""
    return dataclasses.replace(result, input_source=input_source)


def folder_listing(folder):
    """Return the sorted names found in a folder, hidden files included."""
    return sorted(os.listdir(folder))


def summary_text_of(csv_path):
    """Return the text of the summary beside a CSV path."""
    return outputs.summary_path_for(csv_path).read_text(encoding="utf-8")


def test_write_leaves_exactly_the_two_files_with_the_shared_serializer_bytes(tmp_path):
    """Exactly run.csv and run_summary.txt appear, and the CSV equals the serializer's output."""
    result = make_result()
    outputs.write_outputs(result, tmp_path / "run.csv")
    assert folder_listing(tmp_path) == ["run.csv", "run_summary.txt"]
    assert (tmp_path / "run.csv").read_bytes() == outputs.serialize_csv(result).encode("utf-8")


def test_csv_has_exact_header_unix_line_endings_and_round_trip_numbers(tmp_path):
    """The header is fixed, lines end in \\n, and every number is repr of a float that converts back exactly."""
    result = make_result()
    outputs.write_outputs(result, tmp_path / "run.csv")
    raw = (tmp_path / "run.csv").read_bytes()
    assert b"\r" not in raw
    text = raw.decode("utf-8")
    assert text.split("\n")[0] == "feature_label,object_id,contour_area_um2,equivalent_diameter_um,volume_um3"
    assert "np.float64" not in text
    parsed = list(csv.reader(io.StringIO(text, newline="")))
    assert len(parsed) == len(result.rows) + 1
    for cells, row in zip(parsed[1:], result.rows):
        assert cells[2] == repr(float(row.contour_area_um2))
        assert float(cells[2]) == row.contour_area_um2
        assert float(cells[3]) == row.equivalent_diameter_um
        assert float(cells[4]) == row.volume_um3


def test_numpy_number_cells_are_written_as_plain_floats(tmp_path):
    """Rows holding NumPy numbers still give plain cells, because each is converted to a float first."""
    result = make_result()
    numpy_rows = []
    for row in result.rows:
        numpy_rows.append(
            dataclasses.replace(
                row,
                contour_area_um2=numpy.float64(row.contour_area_um2),
                equivalent_diameter_um=numpy.float64(row.equivalent_diameter_um),
                volume_um3=numpy.float64(row.volume_um3),
            )
        )
    text = outputs.serialize_csv(dataclasses.replace(result, rows=tuple(numpy_rows)))
    assert "np.float64" not in text
    assert text == outputs.serialize_csv(result)


def test_class_with_comma_and_quote_reads_back_verbatim(tmp_path):
    """A class name holding a comma and a double quote survives the csv module unchanged."""
    result = make_result()
    outputs.write_outputs(result, tmp_path / "run.csv")
    with open(tmp_path / "run.csv", encoding="utf-8", newline="") as handle:
        labels = {line[0] for line in csv.reader(handle)}
    assert 'say "hi", ok' in labels


def test_summary_readable_part_lists_inputs_entries_classes_and_file_names(tmp_path):
    """The readable part shows every promised section, with class lines starting with their labels."""
    result = make_result(threshold_lower=40.0, threshold_upper=60.0, max_attempts=500, shuffle=True)
    outputs.write_outputs(result, tmp_path / "run.csv")
    text = summary_text_of(tmp_path / "run.csv")
    readable = text.split(MARKER_NAME)[0]
    assert "Inputs: entries given as a Python list" in readable
    assert "Run time: " in readable
    assert "C1  entry 1  A  gaussian  N=4  mean=50.0  stdev=3.0" in readable
    assert 'C2  entry 2  say "hi", ok  gaussian' in readable
    assert "C1  entry 3  A  gaussian  N=2" in readable
    assert "Lower threshold: 40.0" in readable
    assert "Upper threshold: 60.0" in readable
    assert "Seed: 11" in readable
    assert "Maximum attempts: 500" in readable
    assert "Shuffle: True" in readable
    versions = outputs.current_library_versions()
    assert f"numpy {versions['numpy']}" in readable
    assert "C1  A  6" in readable
    assert 'C2  say "hi", ok  4' in readable
    assert "'run.csv'" in readable
    assert "'run_summary.txt'" in readable
    for line in readable.split("\n"):
        if "say \"hi\"" in line or line.endswith("  6"):
            assert re.match(r"C\d+  ", line)


def test_summary_inputs_line_for_text_and_for_a_file_path(tmp_path):
    """Entries given as text say so; a recorded path appears as its repr."""
    text_entries = json.dumps([gaussian("A")])
    text_result = dataset.generate_dataset(text_entries, seed=1)
    outputs.write_outputs(text_result, tmp_path / "text.csv")
    assert "Inputs: entries given as text" in summary_text_of(tmp_path / "text.csv")
    path_result = make_result(input_source="data/in.json")
    outputs.write_outputs(path_result, tmp_path / "path.csv")
    assert "Inputs: 'data/in.json'" in summary_text_of(tmp_path / "path.csv")


def test_reproduction_block_has_exactly_the_documented_keys(tmp_path):
    """One marker pair holds a JSON object with the seven keys and no format version."""
    result = make_result()
    outputs.write_outputs(result, tmp_path / "run.csv")
    text = summary_text_of(tmp_path / "run.csv")
    lines = text.split("\n")
    assert lines.count(MARKER_NAME) == 1
    assert lines.count("--- END REPRODUCTION PARAMETERS ---") == 1
    start = lines.index(MARKER_NAME)
    end = lines.index("--- END REPRODUCTION PARAMETERS ---")
    block = json.loads("\n".join(lines[start + 1 : end]))
    assert sorted(block) == sorted(
        ["entries", "threshold_lower", "threshold_upper", "seed", "max_attempts", "shuffle", "library_versions"]
    )
    assert sorted(block["library_versions"]) == ["numpy", "python", "scipy"]


def test_read_summary_returns_entries_equal_to_the_results_and_accepted_by_generate(tmp_path):
    """The parameters read back equal the run's settings and generate the same rows."""
    result = make_result(threshold_lower=40.0, shuffle=True)
    outputs.write_outputs(result, tmp_path / "run.csv")
    parameters = outputs.read_summary(tmp_path / "run_summary.txt")
    assert parameters.entries == list(result.entries)
    assert parameters.threshold_lower == 40.0
    assert parameters.threshold_upper is None
    assert parameters.seed == 11
    assert parameters.shuffle is True
    again = dataset.generate_dataset(
        parameters.entries,
        threshold_lower=parameters.threshold_lower,
        seed=parameters.seed,
        shuffle=parameters.shuffle,
    )
    assert again.rows == result.rows


def test_replay_gives_the_same_csv_bytes_and_records_the_summary_path(tmp_path):
    """Replaying a summary recreates the CSV bytes and names the summary as the input."""
    original = make_result(shuffle=True)
    outputs.write_outputs(original, tmp_path / "run.csv")
    summary_path = tmp_path / "run_summary.txt"
    replayed = outputs.replay_summary(summary_path)
    assert outputs.serialize_csv(replayed).encode("utf-8") == (tmp_path / "run.csv").read_bytes()
    assert replayed.input_source == str(summary_path)
    outputs.write_outputs(replayed, tmp_path / "again.csv")
    assert f"Inputs: {str(summary_path)!r}" in summary_text_of(tmp_path / "again.csv")


def test_replay_warns_on_stderr_for_another_library_version_and_keeps_the_bytes(tmp_path, capsys):
    """An edited numpy version prints a warning but the bytes stay the same."""
    outputs.write_outputs(make_result(), tmp_path / "run.csv")
    summary_path = tmp_path / "run_summary.txt"
    text = summary_path.read_text(encoding="utf-8")
    versions = outputs.current_library_versions()
    edited = text.replace(f'"numpy": "{versions["numpy"]}"', '"numpy": "0.0.0"')
    assert edited != text
    summary_path.write_text(edited, encoding="utf-8")
    capsys.readouterr()
    replayed = outputs.replay_summary(summary_path)
    captured = capsys.readouterr()
    assert "numpy 0.0.0" in captured.err
    assert captured.out == ""
    assert outputs.serialize_csv(replayed).encode("utf-8") == (tmp_path / "run.csv").read_bytes()


def test_matching_versions_print_no_warning(tmp_path, capsys):
    """An unedited summary replays silently."""
    outputs.write_outputs(make_result(), tmp_path / "run.csv")
    capsys.readouterr()
    outputs.replay_summary(tmp_path / "run_summary.txt")
    assert capsys.readouterr().err == ""


def test_class_named_like_the_begin_marker_still_reads_and_replays(tmp_path):
    """A class called exactly the begin marker cannot fake a block, because class lines have a label prefix."""
    result = make_result([gaussian(MARKER_NAME), gaussian("--- END REPRODUCTION PARAMETERS ---")])
    outputs.write_outputs(result, tmp_path / "run.csv")
    text = summary_text_of(tmp_path / "run.csv")
    assert text.split("\n").count(MARKER_NAME) == 1
    replayed = outputs.replay_summary(tmp_path / "run_summary.txt")
    assert outputs.serialize_csv(replayed) == outputs.serialize_csv(result)


def test_input_path_with_a_line_break_and_marker_still_gives_a_readable_summary(tmp_path):
    """A path holding a newline then the begin marker is written with repr, so it stays on one line."""
    nasty_path = "in.json\n" + MARKER_NAME + "\nrest"
    result = make_result(input_source=nasty_path)
    outputs.write_outputs(result, tmp_path / "run.csv")
    text = summary_text_of(tmp_path / "run.csv")
    assert text.split("\n").count(MARKER_NAME) == 1
    assert f"Inputs: {nasty_path!r}" in text
    parameters = outputs.read_summary(tmp_path / "run_summary.txt")
    assert parameters.seed == 11


def edit_block(summary_path, change):
    """Rewrite a summary after applying a change function to its parsed reproduction block."""
    lines = summary_path.read_text(encoding="utf-8").split("\n")
    start = lines.index(MARKER_NAME)
    end = lines.index("--- END REPRODUCTION PARAMETERS ---")
    block = json.loads("\n".join(lines[start + 1 : end]))
    change(block)
    lines[start + 1 : end] = [json.dumps(block, indent=2)]
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def set_value(key, value):
    """Return a function that stores a value under a key of a block."""

    def apply(block):
        """Store the value."""
        block[key] = value

    return apply


def drop_key(key):
    """Return a function that deletes a key from a block."""

    def apply(block):
        """Delete the key."""
        del block[key]

    return apply


def drop_scipy(block):
    """Remove scipy from the block's library versions."""
    del block["library_versions"]["scipy"]


def break_first_entry(block):
    """Give the first entry a negative standard deviation."""
    block["entries"][0]["stdev"] = -1.0


def test_read_summary_rejects_edited_blocks_with_specific_messages(tmp_path):
    """Each kind of damage gives its own message, and generate_dataset's wording for bad values."""
    outputs.write_outputs(make_result(), tmp_path / "run.csv")
    original = (tmp_path / "run_summary.txt").read_text(encoding="utf-8")
    summary_path = tmp_path / "edited.txt"
    cases = [
        (set_value("threshold_lower", 10), "threshold_lower must be below threshold_upper"),
        (set_value("seed", -1), "seed must be a whole number of 0 or more"),
        (set_value("max_attempts", 0), "max_attempts must be a whole number of 1 or more"),
        (set_value("shuffle", "yes"), "shuffle must be True or False"),
        (set_value("shuffle", 1), "shuffle must be True or False"),
        (set_value("format_version", 1), "unexpected key 'format_version'"),
        (set_value("extra", 1), "unexpected key 'extra'"),
        (drop_key("seed"), "missing the key 'seed'"),
        (drop_scipy, "library_versions is missing the key 'scipy'"),
        (break_first_entry, "entry 1"),
    ]
    for change, message in cases:
        summary_path.write_text(original, encoding="utf-8")
        if message.startswith("threshold_lower"):
            edit_block(summary_path, set_value("threshold_upper", 5))
        edit_block(summary_path, change)
        with pytest.raises(ValueError, match=re.escape(message)):
            outputs.read_summary(summary_path)


def test_read_summary_rejects_damaged_marker_lines_and_json(tmp_path):
    """A missing begin, missing end, two blocks and malformed JSON are each refused."""
    outputs.write_outputs(make_result(), tmp_path / "run.csv")
    original = (tmp_path / "run_summary.txt").read_text(encoding="utf-8")
    end_marker = "--- END REPRODUCTION PARAMETERS ---"
    block_start = original.index(MARKER_NAME)
    block_text = original[block_start:]
    cases = [
        (original.replace(MARKER_NAME, "gone"), "no begin line"),
        (original.replace(end_marker, "gone"), "no end line"),
        (original + block_text, "2 begin lines"),
        (original.replace('"seed": 11', '"seed": 11,,'), "not valid JSON"),
    ]
    summary_path = tmp_path / "damaged.txt"
    for text, message in cases:
        summary_path.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError, match=re.escape(message)):
            outputs.read_summary(summary_path)


def test_destination_check_and_write_reject_bad_paths_without_creating_files(tmp_path):
    """Wrong suffix, missing folder and folders at either path are refused and nothing appears."""
    result = make_result()
    (tmp_path / "out.csv").mkdir()
    (tmp_path / "dirsum").mkdir()
    (tmp_path / "dirsum_summary.txt").mkdir()
    before = folder_listing(tmp_path)
    bad_paths = [
        (tmp_path / "run.txt", "ending in '.csv'"),
        (tmp_path / "run.CSV", "ending in '.csv'"),
        (tmp_path / "missing" / "run.csv", "does not exist"),
        (tmp_path / "out.csv", "is a folder"),
        (tmp_path / "dirsum_summary.txt", "ending in '.csv'"),
    ]
    for path, message in bad_paths:
        for overwrite in (False, True):
            with pytest.raises(ValueError, match=re.escape(message)):
                outputs.check_destinations(path, overwrite)
            with pytest.raises(ValueError, match=re.escape(message)):
                outputs.write_outputs(result, path, overwrite=overwrite)
    assert folder_listing(tmp_path) == before


def test_summary_path_that_is_a_folder_blocks_the_csv_even_with_overwrite(tmp_path):
    """A folder at the summary path refuses the whole write, so the CSV is not written either."""
    result = make_result()
    (tmp_path / "dirsum_summary.txt").mkdir()
    before = folder_listing(tmp_path)
    for overwrite in (False, True):
        with pytest.raises(ValueError, match="is a folder"):
            outputs.check_destinations(tmp_path / "dirsum.csv", overwrite)
        with pytest.raises(ValueError, match="is a folder"):
            outputs.write_outputs(result, tmp_path / "dirsum.csv", overwrite=overwrite)
    assert folder_listing(tmp_path) == before


def test_existing_files_without_overwrite_keep_their_bytes(tmp_path):
    """An existing CSV or an existing summary is refused and both keep their bytes."""
    result = make_result()
    for existing in ("run.csv", "run_summary.txt"):
        folder = tmp_path / existing.replace(".", "_")
        folder.mkdir()
        (folder / existing).write_bytes(b"old " + existing.encode())
        before = folder_listing(folder)
        with pytest.raises(ValueError, match="already exists"):
            outputs.check_destinations(folder / "run.csv")
        with pytest.raises(ValueError, match="already exists"):
            outputs.write_outputs(result, folder / "run.csv")
        assert folder_listing(folder) == before
        assert (folder / existing).read_bytes() == b"old " + existing.encode()


def test_overwrite_replaces_both_existing_files(tmp_path):
    """With overwrite the old pair is replaced by the new one and no extra file remains."""
    (tmp_path / "run.csv").write_bytes(b"old csv")
    (tmp_path / "run_summary.txt").write_bytes(b"old summary")
    result = make_result()
    outputs.write_outputs(result, tmp_path / "run.csv", overwrite=True)
    assert (tmp_path / "run.csv").read_bytes() == outputs.serialize_csv(result).encode("utf-8")
    assert outputs.read_summary(tmp_path / "run_summary.txt").seed == 11
    assert folder_listing(tmp_path) == ["run.csv", "run_summary.txt"]


def test_written_files_get_the_normal_umask_mode(tmp_path):
    """Under a umask of 0o022 both files are 0o644, which a mkstemp temporary file (0o600) would fail."""
    previous = os.umask(0o022)
    try:
        outputs.write_outputs(make_result(), tmp_path / "run.csv")
    finally:
        os.umask(previous)
    for name in ("run.csv", "run_summary.txt"):
        assert stat.S_IMODE(os.stat(tmp_path / name).st_mode) == 0o644


def test_second_temporary_write_failure_leaves_the_folder_and_old_files_unchanged(tmp_path, monkeypatch):
    """When the second temporary file cannot be written, nothing is left and old bytes stay."""
    result = make_result()
    real_write = outputs.write_temporary_file
    calls = []

    def fail_second(path, text):
        """Write the first temporary file normally and fail on the second."""
        calls.append(path)
        if len(calls) == 2:
            raise OSError("disk full")
        real_write(path, text)

    monkeypatch.setattr(outputs, "write_temporary_file", fail_second)
    clean = tmp_path / "clean"
    clean.mkdir()
    with pytest.raises(OSError, match="disk full"):
        outputs.write_outputs(result, clean / "run.csv")
    assert folder_listing(clean) == []
    kept = tmp_path / "kept"
    kept.mkdir()
    (kept / "run.csv").write_bytes(b"old csv")
    (kept / "run_summary.txt").write_bytes(b"old summary")
    calls.clear()
    with pytest.raises(OSError, match="disk full"):
        outputs.write_outputs(result, kept / "run.csv", overwrite=True)
    assert folder_listing(kept) == ["run.csv", "run_summary.txt"]
    assert (kept / "run.csv").read_bytes() == b"old csv"
    assert (kept / "run_summary.txt").read_bytes() == b"old summary"


def patch_summary_rename_to_fail(monkeypatch):
    """Make os.replace fail when the destination is a summary file, and work otherwise."""
    real_replace = os.replace

    def replace_or_fail(source, destination):
        """Rename normally except for a summary destination."""
        if str(destination).endswith(outputs.SUMMARY_ENDING):
            raise OSError("rename refused")
        real_replace(source, destination)

    monkeypatch.setattr(outputs.os, "replace", replace_or_fail)


def test_summary_rename_failure_after_replacing_a_csv_says_the_csv_was_replaced(tmp_path, monkeypatch):
    """With an existing pair, the old summary survives and the error says the CSV was replaced."""
    (tmp_path / "run.csv").write_bytes(b"old csv")
    (tmp_path / "run_summary.txt").write_bytes(b"old summary")
    result = make_result()
    patch_summary_rename_to_fail(monkeypatch)
    with pytest.raises(OSError, match="was replaced and the summary was not"):
        outputs.write_outputs(result, tmp_path / "run.csv", overwrite=True)
    assert folder_listing(tmp_path) == ["run.csv", "run_summary.txt"]
    assert (tmp_path / "run_summary.txt").read_bytes() == b"old summary"
    assert (tmp_path / "run.csv").read_bytes() == outputs.serialize_csv(result).encode("utf-8")


def test_summary_rename_failure_with_no_existing_csv_removes_the_new_csv(tmp_path, monkeypatch):
    """When the CSV was new, a failed summary rename deletes it and leaves no file at all."""
    patch_summary_rename_to_fail(monkeypatch)
    with pytest.raises(OSError, match="rename refused"):
        outputs.write_outputs(make_result(), tmp_path / "run.csv")
    assert folder_listing(tmp_path) == []
    (tmp_path / "run_summary.txt").write_bytes(b"old summary")
    with pytest.raises(OSError, match="rename refused"):
        outputs.write_outputs(make_result(), tmp_path / "run.csv", overwrite=True)
    assert folder_listing(tmp_path) == ["run_summary.txt"]
    assert (tmp_path / "run_summary.txt").read_bytes() == b"old summary"
