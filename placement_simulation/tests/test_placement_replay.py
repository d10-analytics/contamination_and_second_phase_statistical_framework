# This file checks replaying a saved placement run from its summary. A run is
# placed and saved, then replayed, and the replayed placements text must equal
# the saved placements file byte for byte, for a CSV input and a generator
# input, in random and in centre placement. It also checks every refusal: a
# changed CSV (refused before any coverage line), a generator result edited
# after it was placed, a moved CSV that is given by path, an extra settings
# key, an edited mode, an unknown input kind, and the one warning per library
# when the summary records other library versions. Runs use 20 particles.

import dataclasses
import json
import shutil
from pathlib import Path

import pytest

from placement_simulation import place_particles, replay_placement, saving, write_placement_outputs
from synthetic_data_generation import dataset, outputs

SQUARE = {"shape": "square", "side_um": 120.0}
RANDOM_SETTINGS = {"mode": "random", "order": "shuffled", "max_tries": 5, "buffer_um": 1.0, "seed": 7}
CENTRE_SETTINGS = {"mode": "centre", "order": "shuffled", "max_tries": 1, "buffer_um": 1.0, "seed": 7}
MODE_SETTINGS = [RANDOM_SETTINGS, CENTRE_SETTINGS]
KINDS = ["csv", "generator"]


def make_generated():
    """Return a seeded generator result of 20 particles in two classes."""
    entries = [
        {"class": "A", "distribution": "gaussian", "N": 12, "mean": 14.0, "stdev": 2.0},
        {"class": "B", "distribution": "gaussian", "N": 8, "mean": 9.0, "stdev": 1.0},
    ]
    return dataset.generate_dataset(entries, seed=5)


def place_with(particles, settings):
    """Place the particles on the test slate with the named settings."""
    return place_particles(
        particles,
        SQUARE,
        mode=settings["mode"],
        order=settings["order"],
        max_tries=settings["max_tries"],
        buffer_um=settings["buffer_um"],
        seed=settings["seed"],
    )


def save_run(tmp_path, kind, settings):
    """Place and save a run for the input kind; return (summary path, placements path, the CSV path or None)."""
    result = make_generated()
    source = None
    particles = result
    if kind == "csv":
        outputs.write_outputs(result, tmp_path / "sizes.csv")
        source = tmp_path / "sizes.csv"
        particles = source
    placed = place_with(particles, settings)
    placements_path, summary_path = write_placement_outputs(placed, tmp_path / "run")
    return summary_path, placements_path, source


def edit_summary(summary_path, edit):
    """Rewrite the summary's settings block after edit(block) changed the loaded dictionary."""
    text = summary_path.read_text(encoding="utf-8")
    block = outputs.load_block(text)
    edit(block)
    lines = text.split("\n")
    begin = lines.index(outputs.BEGIN_MARKER)
    end = lines.index(outputs.END_MARKER)
    kept = lines[: begin + 1] + [json.dumps(block, indent=2)] + lines[end:]
    summary_path.write_text("\n".join(kept), encoding="utf-8")


def refusal(summary_path, measurements_path=None):
    """Return the ValueError text of replaying the summary, failing if it does not raise one."""
    with pytest.raises(ValueError) as caught:
        replay_placement(summary_path, measurements_path)
    return str(caught.value)


def test_replay_equals_the_saved_placements_byte_for_byte(tmp_path):
    """Replaying a saved run gives the saved placements file's text for each input kind and placement mode."""
    for kind in KINDS:
        for settings in MODE_SETTINGS:
            folder = tmp_path / (kind + "_" + settings["mode"])
            folder.mkdir()
            summary_path, placements_path, _ = save_run(folder, kind, settings)
            replayed = replay_placement(summary_path)
            saved_text = placements_path.read_text(encoding="utf-8")
            assert saving.serialize_placements(replayed) == saved_text, (kind, settings["mode"])
            assert any(row.placed for row in replayed.rows), (kind, settings["mode"])


def test_a_changed_csv_digit_is_refused_before_any_coverage_line(tmp_path, capsys):
    """One changed diameter digit gives a checksum refusal naming the file, and no coverage line is printed."""
    summary_path, _, source = save_run(tmp_path, "csv", RANDOM_SETTINGS)
    text = source.read_text(encoding="utf-8")
    lines = text.split("\n")
    fields = lines[1].split(",")
    fields[3] = fields[3][:-1] + ("1" if fields[3][-1] != "1" else "2")
    lines[1] = ",".join(fields)
    source.write_text("\n".join(lines), encoding="utf-8")
    capsys.readouterr()
    message = refusal(summary_path)
    assert str(source) in message
    assert "SHA-256" in message
    assert capsys.readouterr().err == ""


def test_an_edited_generator_result_is_refused_at_replay(tmp_path, capsys):
    """A result changed with dataclasses.replace after its settings were recorded no longer matches the regenerated CSV."""
    result = make_generated()
    first = dataclasses.replace(result.rows[0], equivalent_diameter_um=result.rows[0].equivalent_diameter_um + 1.0)
    edited = dataclasses.replace(result, rows=(first,) + result.rows[1:])
    placed = place_with(edited, RANDOM_SETTINGS)
    _, summary_path = write_placement_outputs(placed, tmp_path / "run")
    capsys.readouterr()
    message = refusal(summary_path)
    assert "generator result is not the one that was placed" in message
    assert capsys.readouterr().err == ""


def test_a_moved_csv_replays_when_its_path_is_given(tmp_path):
    """The CSV is found by measurements_path when it is no longer beside the summary, and is refused when it is nowhere."""
    summary_path, placements_path, source = save_run(tmp_path, "csv", RANDOM_SETTINGS)
    moved_folder = tmp_path / "elsewhere"
    moved_folder.mkdir()
    moved = moved_folder / "renamed.csv"
    shutil.move(str(source), str(moved))
    assert "does not exist" in refusal(summary_path)
    replayed = replay_placement(summary_path, moved)
    assert saving.serialize_placements(replayed) == placements_path.read_text(encoding="utf-8")


def test_measurements_path_is_refused_with_generator_input(tmp_path):
    """A generator summary has no CSV to replace, so a measurements_path is refused."""
    summary_path, _, _ = save_run(tmp_path, "generator", RANDOM_SETTINGS)
    other = tmp_path / "other.csv"
    outputs.write_outputs(make_generated(), other)
    assert "measurements_path cannot be used" in refusal(summary_path, other)


def add_extra_key(block):
    """Add a key the settings block does not have."""
    block["extra"] = 1


def set_other_mode(block):
    """Change the mode to a value no run accepts."""
    block["mode"] = "spiral"


def set_unknown_kind(block):
    """Change the input kind to one that does not exist."""
    block["input"]["kind"] = "database"


def add_input_key(block):
    """Add a key the CSV input object does not have."""
    block["input"]["extra"] = 1


CHANGED_BLOCKS = [
    (add_extra_key, "has an unexpected key 'extra'"),
    (set_unknown_kind, "input kind must be 'csv' or 'generator'"),
    (add_input_key, "has an unexpected key 'extra'"),
]


def test_a_changed_block_is_refused(tmp_path):
    """An extra key, an unknown input kind or an extra input key is refused with a message naming it."""
    for number, (edit, expected) in enumerate(CHANGED_BLOCKS):
        folder = tmp_path / str(number)
        folder.mkdir()
        summary_path, _, _ = save_run(folder, "csv", RANDOM_SETTINGS)
        edit_summary(summary_path, edit)
        assert expected in refusal(summary_path)


def test_an_edited_mode_gets_the_message_of_a_fresh_run(tmp_path):
    """The refusal of a mode edited in the summary is the message place_particles gives for the same value."""
    summary_path, _, source = save_run(tmp_path, "csv", RANDOM_SETTINGS)
    with pytest.raises(ValueError) as fresh:
        place_particles(source, SQUARE, mode="spiral")
    edit_summary(summary_path, set_other_mode)
    assert refusal(summary_path) == str(fresh.value)


def change_all_versions(block):
    """Record other library versions in the settings block and in the generator's own block."""
    other = {"python": "0.0.1", "numpy": "0.0.2", "scipy": "0.0.3"}
    block["library_versions"] = dict(other)
    block["input"]["generator_parameters"]["library_versions"] = dict(other)


def change_all_versions_csv(block):
    """Record other library versions in the settings block of a CSV summary."""
    block["library_versions"] = {"python": "0.0.1", "numpy": "0.0.2", "scipy": "0.0.3"}


def test_other_library_versions_warn_once_per_library(tmp_path, capsys):
    """A summary with other versions prints one warning for each of the three libraries, also for generator input."""
    for kind in KINDS:
        folder = tmp_path / kind
        folder.mkdir()
        summary_path, placements_path, _ = save_run(folder, kind, RANDOM_SETTINGS)
        if kind == "generator":
            edit_summary(summary_path, change_all_versions)
        else:
            edit_summary(summary_path, change_all_versions_csv)
        capsys.readouterr()
        replayed = replay_placement(summary_path)
        warnings = [line for line in capsys.readouterr().err.split("\n") if line.startswith("warning:")]
        assert len(warnings) == 3, kind
        assert "python 0.0.1" in warnings[0]
        assert "numpy 0.0.2" in warnings[1]
        assert "scipy 0.0.3" in warnings[2]
        assert saving.serialize_placements(replayed) == placements_path.read_text(encoding="utf-8")
