# This file checks saving a placement run through the real writer: the exact
# placements CSV, the readable summary and its settings block, the file mode,
# the refusal to replace existing files, and the all-or-nothing behaviour when
# a temporary write or a rename fails. A failure is injected in the
# generator's outputs module, where the shared writer lives, so a private
# copy of the writer in the simulator would fail these tests. Runs use a few
# particles on a small slate in a temporary folder. A "umask" is the
# permission mask that decides the mode of newly created files.

import csv
import io
import os
import re
import stat
from pathlib import Path

import pytest

from placement_simulation import check_placement_destinations, place_particles, write_placement_outputs
from placement_simulation import saving
from synthetic_data_generation import outputs

SQUARE = {"shape": "square", "side_um": 60.0}
HEADER_LINE = (
    "placement_order,source_index,object_id,feature_label,equivalent_diameter_um,placed,"
    "x_um,y_um,tries,edge_rejections,overlap_rejections"
)
CSV_HEADER = "feature_label,object_id,contour_area_um2,equivalent_diameter_um,volume_um3\n"


def folder_listing(folder):
    """Return the sorted names found in a folder, hidden files included."""
    return sorted(os.listdir(folder))


def make_run(tmp_path):
    """Place six particles of diameter 30 on a 60 micrometre square with 5 tries each; some do not fit."""
    lines = [CSV_HEADER]
    for number in range(1, 7):
        lines.append(f"Class A,P{number},700.0,30.0,9000.0\n")
    source = tmp_path / "sizes.csv"
    source.write_bytes("".join(lines).encode("utf-8"))
    result = place_particles(source, SQUARE, max_tries=5, seed=4)
    source.unlink()
    return result


def make_folder(tmp_path):
    """Return a fresh empty output folder inside tmp_path."""
    folder = tmp_path / "out"
    folder.mkdir()
    return folder


def fail_on_second_call(path, text):
    """Fail the way a full disk would when the second temporary file is written."""
    fail_on_second_call.calls += 1
    if fail_on_second_call.calls == 2:
        raise OSError("disk full")
    real_write(path, text)


real_write = outputs.write_temporary_file


def patch_second_write_to_fail(monkeypatch):
    """Make outputs.write_temporary_file fail on its second call and work otherwise."""
    fail_on_second_call.calls = 0
    monkeypatch.setattr(outputs, "write_temporary_file", fail_on_second_call)


def patch_summary_rename_to_fail(monkeypatch):
    """Make outputs.os.replace fail when the destination is a summary file, and work otherwise."""
    real_replace = os.replace

    def replace_or_fail(source, destination):
        """Rename normally except for a summary destination."""
        if str(destination).endswith(saving.SUMMARY_ENDING):
            raise OSError("rename refused")
        real_replace(source, destination)

    monkeypatch.setattr(outputs.os, "replace", replace_or_fail)


def test_placements_file_has_the_exact_header_and_cells(tmp_path):
    """The header is exact; placed rows hold numbers, unplaced rows hold empty coordinates and the full tries."""
    result = make_run(tmp_path)
    folder = make_folder(tmp_path)
    placements_path, _ = write_placement_outputs(result, folder / "run")
    raw = placements_path.read_bytes()
    assert b"\r" not in raw
    text = raw.decode("utf-8")
    assert text.split("\n")[0] == HEADER_LINE
    assert text.endswith("\n")
    records = list(csv.reader(io.StringIO(text, newline="")))
    assert len(records) == 1 + 6
    flags = [record[5] for record in records[1:]]
    assert "true" in flags and "false" in flags
    for record, row in zip(records[1:], result.rows):
        assert record[0] == str(row.placement_order)
        assert record[1] == str(row.source_index)
        assert record[2:5] == [row.object_id, row.feature_label, repr(float(row.equivalent_diameter_um))]
        assert record[8:] == [str(row.tries), str(row.edge_rejections), str(row.overlap_rejections)]
        if row.placed:
            assert record[5:8] == ["true", repr(float(row.x_um)), repr(float(row.y_um))]
        else:
            assert record[5:8] == ["false", "", ""]
            assert record[8] == "5"
    assert records[1][0] == "1"
    assert records[1][4] == "30.0"
    assert records[1][2] == "P1"
    assert records[1][3] == "Class A"
    assert text == saving.serialize_placements(result)


def test_summary_has_the_readable_lines_and_the_exact_settings_block(tmp_path):
    """The four prescribed lines are present, and the block has exactly the nine keys and reads back."""
    result = make_run(tmp_path)
    folder = make_folder(tmp_path)
    placements_path, summary_path = write_placement_outputs(result, folder / "run")
    assert placements_path == folder / "run_placements.csv"
    assert summary_path == folder / "run_summary.txt"
    text = summary_path.read_text(encoding="utf-8")
    placed = sum(1 for row in result.rows if row.placed)
    lines = text.split("\n")
    assert f"Placed: {placed}" in lines
    assert f"Not placed: {6 - placed}" in lines
    assert f"Requested coverage: {result.requested_coverage!r}" in lines
    assert f"Achieved coverage: {result.achieved_coverage!r}" in lines
    assert 0 < placed < 6
    assert "Seed: 4" in lines
    assert "Order: as_given" in lines
    assert "Maximum tries: 5" in lines
    assert any(line.startswith("Input: measurements file 'sizes.csv', SHA-256 ") for line in lines)
    assert "Placements file: 'run_placements.csv'" in lines
    assert "Summary file: 'run_summary.txt'" in lines
    block = outputs.load_block(text)
    assert list(block) == [
        "slate",
        "mode",
        "max_tries",
        "buffer_um",
        "pixel_um",
        "order",
        "seed",
        "input",
        "library_versions",
    ]
    assert block["slate"] == SQUARE
    assert block["mode"] == "random"
    assert block["max_tries"] == 5
    assert block["buffer_um"] == 0.0
    assert block["pixel_um"] == 1.0
    assert block["seed"] == 4
    assert block["input"] == result.input
    assert block["library_versions"] == outputs.current_library_versions()
    assert outputs.BEGIN_MARKER in lines
    assert lines[-2] == outputs.END_MARKER


def test_a_run_leaves_exactly_two_files_with_mode_644(tmp_path):
    """Two files and nothing hidden remain, each with mode 0o644 under a umask of 0o022."""
    result = make_run(tmp_path)
    folder = make_folder(tmp_path)
    previous = os.umask(0o022)
    try:
        write_placement_outputs(result, folder / "run")
    finally:
        os.umask(previous)
    assert folder_listing(folder) == ["run_placements.csv", "run_summary.txt"]
    for name in folder_listing(folder):
        assert stat.S_IMODE((folder / name).stat().st_mode) == 0o644


def test_existing_files_without_overwrite_are_refused_and_kept(tmp_path):
    """An existing placements file or summary is refused with the simulator's hint, and the folder is unchanged."""
    result = make_run(tmp_path)
    folder = make_folder(tmp_path)
    for existing in ("run_placements.csv", "run_summary.txt"):
        (folder / existing).write_bytes(b"old " + existing.encode("ascii"))
        before = folder_listing(folder)
        with pytest.raises(ValueError) as caught:
            write_placement_outputs(result, folder / "run")
        message = str(caught.value)
        assert message == f"{str(folder / existing)!r} already exists; pass overwrite=True to replace it"
        assert "--overwrite" not in message
        with pytest.raises(ValueError, match="pass overwrite=True to replace it"):
            check_placement_destinations(folder / "run")
        assert folder_listing(folder) == before
        assert (folder / existing).read_bytes() == b"old " + existing.encode("ascii")
        (folder / existing).unlink()
    assert folder_listing(folder) == []


def test_overwrite_replaces_both_existing_files(tmp_path):
    """With overwrite=True both old files are replaced by the new run."""
    result = make_run(tmp_path)
    folder = make_folder(tmp_path)
    (folder / "run_placements.csv").write_bytes(b"old")
    (folder / "run_summary.txt").write_bytes(b"old")
    write_placement_outputs(result, folder / "run", overwrite=True)
    assert (folder / "run_placements.csv").read_text(encoding="utf-8") == saving.serialize_placements(result)
    assert outputs.load_block((folder / "run_summary.txt").read_text(encoding="utf-8"))["seed"] == 4


def test_destination_check_returns_the_two_paths_and_refuses_bad_names(tmp_path):
    """The check gives both paths, refuses an empty name and a missing folder, and writes nothing."""
    folder = make_folder(tmp_path)
    assert check_placement_destinations(folder / "run") == (folder / "run_placements.csv", folder / "run_summary.txt")
    with pytest.raises(ValueError, match="no file name"):
        check_placement_destinations(Path("/"))
    with pytest.raises(ValueError, match="does not exist"):
        check_placement_destinations(tmp_path / "missing" / "run")
    (folder / "run_summary.txt").mkdir()
    with pytest.raises(ValueError, match="is a folder; choose a file name instead"):
        check_placement_destinations(folder / "run", overwrite=True)
    assert folder_listing(folder) == ["run_summary.txt"]


def test_second_temporary_write_failure_changes_nothing(tmp_path, monkeypatch):
    """When the second temporary write fails, old files keep their bytes and no temporary file remains."""
    result = make_run(tmp_path)
    folder = make_folder(tmp_path)
    (folder / "run_placements.csv").write_bytes(b"old placements")
    (folder / "run_summary.txt").write_bytes(b"old summary")
    patch_second_write_to_fail(monkeypatch)
    with pytest.raises(OSError, match="disk full"):
        write_placement_outputs(result, folder / "run", overwrite=True)
    assert fail_on_second_call.calls == 2
    assert folder_listing(folder) == ["run_placements.csv", "run_summary.txt"]
    assert (folder / "run_placements.csv").read_bytes() == b"old placements"
    assert (folder / "run_summary.txt").read_bytes() == b"old summary"


def test_second_temporary_write_failure_in_a_fresh_folder_leaves_it_empty(tmp_path, monkeypatch):
    """With no earlier files, the failed run leaves the folder empty."""
    result = make_run(tmp_path)
    folder = make_folder(tmp_path)
    patch_second_write_to_fail(monkeypatch)
    with pytest.raises(OSError, match="disk full"):
        write_placement_outputs(result, folder / "run")
    assert folder_listing(folder) == []


def test_summary_rename_failure_with_an_existing_pair_names_the_replaced_placements_file(tmp_path, monkeypatch):
    """The error says the placements file was replaced, the old summary survives and no temporary file remains."""
    result = make_run(tmp_path)
    folder = make_folder(tmp_path)
    (folder / "run_placements.csv").write_bytes(b"old placements")
    (folder / "run_summary.txt").write_bytes(b"old summary")
    patch_summary_rename_to_fail(monkeypatch)
    with pytest.raises(OSError) as caught:
        write_placement_outputs(result, folder / "run", overwrite=True)
    placements_name = repr(str(folder / "run_placements.csv"))
    assert str(caught.value).endswith(f"the placements file {placements_name} was replaced and the summary was not")
    assert re.match(r"the summary '.*run_summary\.txt' could not be written \(rename refused\); ", str(caught.value))
    assert folder_listing(folder) == ["run_placements.csv", "run_summary.txt"]
    assert (folder / "run_summary.txt").read_bytes() == b"old summary"
    assert (folder / "run_placements.csv").read_text(encoding="utf-8") == saving.serialize_placements(result)


def test_summary_rename_failure_in_a_fresh_folder_ends_empty_with_the_original_error(tmp_path, monkeypatch):
    """The new placements file is deleted and the original error is raised unchanged."""
    result = make_run(tmp_path)
    folder = make_folder(tmp_path)
    patch_summary_rename_to_fail(monkeypatch)
    with pytest.raises(OSError) as caught:
        write_placement_outputs(result, folder / "run")
    assert str(caught.value) == "rename refused"
    assert folder_listing(folder) == []


def test_placing_alone_leaves_the_folder_unchanged(tmp_path):
    """place_particles writes nothing: the folder holds only the input afterwards."""
    folder = make_folder(tmp_path)
    source = folder / "sizes.csv"
    source.write_bytes((CSV_HEADER + "Class A,P1,700.0,30.0,9000.0\n").encode("utf-8"))
    before = folder_listing(folder)
    place_particles(source, SQUARE, max_tries=5, seed=1)
    assert folder_listing(folder) == before
