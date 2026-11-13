# This file checks reading a saved placement run back. Real runs are saved and
# read again, and the result read must equal the result saved, field for field,
# for random and centre placement, a CSV and a generator input, the three slate
# shapes and a crowded run with unplaced particles. It then edits copies of the
# saved files for every refusal, checks that the summary's settings check is the
# one shared function that replay also calls, and that a missing summary is a
# ValueError from both readers. Runs use 20 particles.

import csv
import io
import json

import pytest

from placement_simulation import place_particles, placement, read_placement_outputs, replay_placement, saving, write_placement_outputs
from synthetic_data_generation import dataset, outputs

SQUARE = {"shape": "square", "side_um": 120.0}
RECTANGLE = {"shape": "rectangle", "width_um": 200.0, "height_um": 90.0}
CIRCLE = {"shape": "circle", "diameter_um": 130.0}
CROWDED = {"shape": "square", "side_um": 40.0}
RANDOM_SETTINGS = {"mode": "random", "order": "shuffled", "max_tries": 5, "buffer_um": 1.0, "seed": 7}
CENTRE_SETTINGS = {"mode": "centre", "order": "shuffled", "max_tries": 1, "buffer_um": 1.0, "seed": 7}
RUNS = [
    ("generator", SQUARE, RANDOM_SETTINGS),
    ("csv", SQUARE, RANDOM_SETTINGS),
    ("generator", RECTANGLE, RANDOM_SETTINGS),
    ("csv", CIRCLE, CENTRE_SETTINGS),
    ("generator", SQUARE, CENTRE_SETTINGS),
    ("csv", CROWDED, RANDOM_SETTINGS),
    ("generator", CROWDED, CENTRE_SETTINGS),
]


def make_generated():
    """Return a seeded generator result of 20 particles in two classes."""
    entries = [
        {"class": "A", "distribution": "gaussian", "N": 12, "mean": 14.0, "stdev": 2.0},
        {"class": "B", "distribution": "gaussian", "N": 8, "mean": 9.0, "stdev": 1.0},
    ]
    return dataset.generate_dataset(entries, seed=5)


def save_run(folder, kind, slate, settings):
    """Place and save a run into the folder; return (the saved result, the base path)."""
    generated = make_generated()
    particles = generated
    if kind == "csv":
        outputs.write_outputs(generated, folder / "sizes.csv")
        particles = folder / "sizes.csv"
    result = place_particles(
        particles,
        slate,
        mode=settings["mode"],
        order=settings["order"],
        max_tries=settings["max_tries"],
        buffer_um=settings["buffer_um"],
        seed=settings["seed"],
    )
    base = folder / "run"
    write_placement_outputs(result, base)
    return result, base


def placements_file(base):
    """Return the saved placements path of a base path."""
    return base.parent / (base.name + saving.PLACEMENTS_ENDING)


def summary_file(base):
    """Return the saved summary path of a base path."""
    return base.parent / (base.name + saving.SUMMARY_ENDING)


def read_table(base):
    """Return the saved placements file as a list of cell lists, header first."""
    text = placements_file(base).read_text(encoding="utf-8")
    return list(csv.reader(io.StringIO(text, newline="")))


def write_table(base, table):
    """Overwrite the saved placements file with the cell lists."""
    buffer = io.StringIO(newline="")
    csv.writer(buffer, lineterminator="\n").writerows(table)
    placements_file(base).write_text(buffer.getvalue(), encoding="utf-8")


def edit_summary(base, edit):
    """Rewrite the summary's settings block after edit(block) changed the loaded dictionary."""
    path = summary_file(base)
    text = path.read_text(encoding="utf-8")
    block = outputs.load_block(text)
    edit(block)
    lines = text.split("\n")
    begin = lines.index(outputs.BEGIN_MARKER)
    end = lines.index(outputs.END_MARKER)
    path.write_text("\n".join(lines[: begin + 1] + [json.dumps(block, indent=2)] + lines[end:]), encoding="utf-8")


def refusal(base):
    """Return the ValueError text of reading the base path, failing if it does not raise one."""
    with pytest.raises(ValueError) as caught:
        read_placement_outputs(base)
    return str(caught.value)


def test_a_saved_run_reads_back_equal_in_every_field(tmp_path):
    """The read result equals the saved one for each input kind, mode and slate, with the exact types kept."""
    unplaced_seen = False
    for number, (kind, slate, settings) in enumerate(RUNS):
        folder = tmp_path / str(number)
        folder.mkdir()
        saved, base = save_run(folder, kind, slate, settings)
        read = read_placement_outputs(base)
        assert read == saved, (kind, slate, settings["mode"])
        for found, expected in zip(read.rows, saved.rows):
            assert type(found.tries) is int and type(found.source_index) is int
            assert type(found.x_um) is type(expected.x_um)
        unplaced_seen = unplaced_seen or any(not row.placed for row in read.rows)
    assert unplaced_seen


def test_the_coverages_are_recomputed_over_the_rows_in_source_order(tmp_path):
    """A shuffled run's requested coverage is the sum taken in source order, so another order would differ in the last digits."""
    saved, base = save_run(tmp_path, "generator", SQUARE, RANDOM_SETTINGS)
    read = read_placement_outputs(base)
    assert [row.source_index for row in read.rows] != sorted(row.source_index for row in read.rows)
    assert read.requested_coverage == saved.requested_coverage
    assert read.achieved_coverage == saved.achieved_coverage


def test_a_header_only_placements_file_reads_as_no_rows(tmp_path):
    """A file holding only the header gives a result with no rows and zero coverage."""
    _, base = save_run(tmp_path, "generator", SQUARE, RANDOM_SETTINGS)
    write_table(base, [list(saving.PLACEMENT_COLUMNS)])
    read = read_placement_outputs(base)
    assert read.rows == ()
    assert read.requested_coverage == 0.0
    assert read.achieved_coverage == 0.0


def set_cell(position, column, value):
    """Return the edit that sets one cell of the table's data line to the value."""
    index = saving.PLACEMENT_COLUMNS.index(column)

    def edit(table):
        """Set the cell."""
        table[position][index] = value

    return edit


def drop_column(table):
    """Remove the last column from every line."""
    for line in table:
        line.pop()


def add_column(table):
    """Append an extra column to every line."""
    for line in table:
        line.append("x")


def swap_columns(table):
    """Swap the first two header names."""
    table[0][0], table[0][1] = table[0][1], table[0][0]


def short_row(table):
    """Remove one cell from the second data line."""
    table[2].pop()


def long_row(table):
    """Add one cell to the second data line."""
    table[2].append("1")


def repeat_source(table):
    """Give the second data line the first line's source_index."""
    table[2][1] = table[1][1]


def empty_file(table):
    """Remove every line."""
    del table[:]


PLACEMENTS_CASES = [
    ("missing column", drop_column, "header"),
    ("extra column", add_column, "header"),
    ("reordered column", swap_columns, "header"),
    ("empty file", empty_file, "header"),
    ("short row", short_row, "cells"),
    ("long row", long_row, "cells"),
    ("order skipped", set_cell(2, "placement_order", "5"), "placement_order must be 2"),
    ("order not a number", set_cell(1, "placement_order", "one"), "placement_order must be a whole number"),
    ("source zero", set_cell(1, "source_index", "0"), "source_index must be a whole number of 1 or more"),
    ("source fraction", set_cell(1, "source_index", "1.5"), "source_index must be a whole number of 1 or more"),
    ("source negative", set_cell(1, "source_index", "-3"), "source_index must be a whole number of 1 or more"),
    ("source repeated", repeat_source, "source_index"),
    ("placed other", set_cell(1, "placed", "True"), "placed must be true or false"),
    ("placed empty", set_cell(1, "placed", ""), "placed must be true or false"),
    ("x empty when placed", set_cell(1, "x_um", ""), "must be filled"),
    ("y empty when placed", set_cell(1, "y_um", ""), "must be filled"),
    ("x not a number", set_cell(1, "x_um", "abc"), "x_um must be a finite number"),
    ("y infinite", set_cell(1, "y_um", "inf"), "y_um must be a finite number"),
    ("x not finite", set_cell(1, "x_um", "nan"), "x_um must be a finite number"),
    ("diameter zero", set_cell(1, "equivalent_diameter_um", "0.0"), "equivalent_diameter_um must be above 0"),
    ("diameter negative", set_cell(1, "equivalent_diameter_um", "-2.0"), "equivalent_diameter_um must be above 0"),
    ("diameter not finite", set_cell(1, "equivalent_diameter_um", "inf"), "equivalent_diameter_um must be a finite number"),
    ("diameter text", set_cell(1, "equivalent_diameter_um", "wide"), "equivalent_diameter_um must be a finite number"),
    ("tries negative", set_cell(1, "tries", "-1"), "tries must be a whole number of 0 or more"),
    ("tries fraction", set_cell(1, "tries", "2.0"), "tries must be a whole number of 0 or more"),
    ("edge rejections text", set_cell(1, "edge_rejections", "x"), "edge_rejections must be a whole number of 0 or more"),
    ("overlap rejections negative", set_cell(1, "overlap_rejections", "-4"), "overlap_rejections must be a whole number of 0 or more"),
]


def test_each_placements_file_refusal_names_the_file_and_the_problem(tmp_path):
    """Every edited copy of the placements file raises a ValueError naming the file and the specific problem."""
    _, base = save_run(tmp_path, "generator", SQUARE, RANDOM_SETTINGS)
    original = read_table(base)
    assert original[1][5] == "true"
    for name, edit, expected in PLACEMENTS_CASES:
        table = [list(line) for line in original]
        edit(table)
        write_table(base, table)
        message = refusal(base)
        assert str(placements_file(base)) in message, name
        assert expected in message, (name, message)


def test_a_filled_coordinate_of_an_unplaced_particle_is_refused(tmp_path):
    """A row that says not placed but holds coordinates is refused naming the file."""
    result, base = save_run(tmp_path, "csv", CROWDED, RANDOM_SETTINGS)
    table = read_table(base)
    position = 0
    for number, row in enumerate(result.rows, start=1):
        if not row.placed:
            position = number
    assert position > 0
    table[position][6] = "1.0"
    write_table(base, table)
    message = refusal(base)
    assert str(placements_file(base)) in message
    assert "must be empty" in message


def test_a_placed_flag_changed_to_false_with_coordinates_is_refused(tmp_path):
    """Flipping a placed row to false leaves its coordinates, which is refused."""
    _, base = save_run(tmp_path, "generator", SQUARE, RANDOM_SETTINGS)
    table = read_table(base)
    table[1][5] = "false"
    write_table(base, table)
    assert "must be empty" in refusal(base)


def test_file_level_refusals_name_the_file(tmp_path):
    """A missing file, a folder in its place and non-UTF-8 bytes each raise a ValueError naming that file."""
    _, base = save_run(tmp_path, "generator", SQUARE, RANDOM_SETTINGS)
    for path in (placements_file(base), summary_file(base)):
        original = path.read_bytes()
        path.unlink()
        assert str(path) in refusal(base) and "does not exist" in refusal(base)
        path.mkdir()
        assert str(path) in refusal(base) and "folder" in refusal(base)
        path.rmdir()
        path.write_bytes(b"\xff\xfe\x00bad")
        assert str(path) in refusal(base) and "UTF-8" in refusal(base)
        path.write_bytes(original)
    assert read_placement_outputs(base).rows


def test_a_refused_mode_gives_the_message_place_particles_gives(tmp_path):
    """A summary with mode spiral gives the same message as place_particles, and it names no file."""
    _, base = save_run(tmp_path, "generator", SQUARE, RANDOM_SETTINGS)
    edit_summary(base, set_block("mode", "spiral"))
    with pytest.raises(ValueError) as direct:
        place_particles(make_generated(), SQUARE, mode="spiral")
    assert refusal(base) == str(direct.value)
    assert str(summary_file(base)) not in refusal(base)


def set_block(key, value):
    """Return the edit that sets one key of the settings block."""

    def edit(block):
        """Set the key."""
        block[key] = value

    return edit


def remove_block_key(block):
    """Remove the seed key from the settings block."""
    del block["seed"]


def add_block_key(block):
    """Add an unknown key to the settings block."""
    block["extra"] = 1


SETTINGS_CASES = [
    ("missing key", remove_block_key, "missing the key 'seed'"),
    ("extra key", add_block_key, "unexpected key 'extra'"),
    ("bad slate", set_block("slate", {"shape": "triangle"}), "slate shape"),
    ("bad max_tries", set_block("max_tries", 0), "max_tries"),
    ("bad buffer", set_block("buffer_um", -1.0), "buffer_um"),
    ("bad pixel", set_block("pixel_um", 0), "pixel_um"),
    ("bad order", set_block("order", "backwards"), "order must be one of"),
    ("bad seed", set_block("seed", "x"), "seed"),
]


def test_every_settings_refusal_is_the_one_replay_gives(tmp_path):
    """Each refusal of the settings block reads the same from the reader and from replay, with no file name added."""
    _, base = save_run(tmp_path, "generator", SQUARE, RANDOM_SETTINGS)
    original = summary_file(base).read_text(encoding="utf-8")
    for name, edit, expected in SETTINGS_CASES:
        summary_file(base).write_text(original, encoding="utf-8")
        edit_summary(base, edit)
        message = refusal(base)
        assert expected in message, (name, message)
        with pytest.raises(ValueError) as replayed:
            replay_placement(summary_file(base))
        assert str(replayed.value) == message, name


def test_a_summary_without_a_block_is_refused(tmp_path):
    """A summary with no settings markers is refused by the shared block loader."""
    _, base = save_run(tmp_path, "generator", SQUARE, RANDOM_SETTINGS)
    summary_file(base).write_text("nothing here\n", encoding="utf-8")
    assert "begin line" in refusal(base)


def test_a_missing_summary_is_a_value_error_from_both_readers(tmp_path):
    """Replay and the reader both raise a ValueError naming a summary that does not exist."""
    _, base = save_run(tmp_path, "generator", SQUARE, RANDOM_SETTINGS)
    summary_file(base).unlink()
    assert str(summary_file(base)) in refusal(base)
    with pytest.raises(ValueError) as replayed:
        replay_placement(summary_file(base))
    assert str(summary_file(base)) in str(replayed.value)


class Marker(Exception):
    """Raised by the replacement summary check."""


def raise_marker(summary_path):
    """Stand in for the shared summary check by raising the marker."""
    raise Marker(str(summary_path))


def test_both_readers_call_the_one_shared_summary_check(tmp_path, monkeypatch):
    """Replacing saving.read_summary_settings makes replay and the reader raise its marker, so neither keeps its own copy."""
    _, base = save_run(tmp_path, "generator", SQUARE, RANDOM_SETTINGS)
    monkeypatch.setattr(saving, "read_summary_settings", raise_marker)
    with pytest.raises(Marker):
        replay_placement(summary_file(base))
    with pytest.raises(Marker):
        read_placement_outputs(base)


def test_the_reader_is_exported_and_uses_the_placement_coverage_functions():
    """The package exports the reader, and the coverage functions it calls are the placement module's."""
    import placement_simulation

    assert placement_simulation.read_placement_outputs is saving.read_placement_outputs
    assert saving.placement is placement


def test_a_cell_too_long_for_the_csv_reader_is_a_value_error_naming_the_file(tmp_path):
    """A cell beyond the csv module's field size limit raises a ValueError naming the file, not a csv error."""
    _, base = save_run(tmp_path, "generator", SQUARE, RANDOM_SETTINGS)
    table = read_table(base)
    table[1][2] = "x" * 200000
    write_table(base, table)
    message = refusal(base)
    assert str(placements_file(base)) in message
    assert "CSV" in message
