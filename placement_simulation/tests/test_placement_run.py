# This file checks one random placement run: how particles are read (from a
# generator result or a measurements CSV) and identified, the settings checks,
# the two independent random streams (one for the order, one for the
# locations), the try and rejection counts, the geometry of the placed
# particles, and the one coverage line printed on stderr before placing. Runs
# use a few to 40 particles on slates of at most 300 micrometres.

import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from placement_simulation import geometry
from placement_simulation import placement
from placement_simulation import place_particles
from synthetic_data_generation import dataset, outputs

SQUARE_200 = {"shape": "square", "side_um": 200.0}
HEADER = "feature_label,object_id,contour_area_um2,equivalent_diameter_um,volume_um3\n"
COVERAGE_LINE = "requested coverage: {!r} (total particle area / slate area)\n"
ROW_COLUMNS = [
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
SETTING_NAMES = ["slate", "mode", "max_tries", "buffer_um", "pixel_um", "order", "seed", "input"]


class Marker(Exception):
    """Raised by a replacement function so a test can see that the replaced function was reached."""


def write_csv(path, diameters, extra_column=True):
    """Write a measurements CSV with the given diameters (object ids P1, P2, ...) and return its path."""
    lines = [HEADER if extra_column else "object_id,feature_label,equivalent_diameter_um\n"]
    for number, diameter in enumerate(diameters, start=1):
        if extra_column:
            lines.append(f"Class A,P{number},1.5,{diameter!r},2.5\n")
        else:
            lines.append(f"P{number},Class A,{diameter!r}\n")
    Path(path).write_bytes("".join(lines).encode("utf-8"))
    return path


def make_generated(counts=(3, 2), seed=5):
    """Return a small seeded generator result with two classes."""
    entries = [
        {"class": "A", "distribution": "gaussian", "N": counts[0], "mean": 20.0, "stdev": 2.0},
        {"class": "B", "distribution": "gaussian", "N": counts[1], "mean": 12.0, "stdev": 1.0},
    ]
    return dataset.generate_dataset(entries, seed=seed)


def row_key(row):
    """Return the placement-order-independent outcome of a row: placed, coordinates, tries and both rejection counts."""
    return (row.placed, row.x_um, row.y_um, row.tries, row.edge_rejections, row.overlap_rejections)


def refusal(function, first, second, settings=None):
    """Return the ValueError text of function(first, second, ...settings), failing if it does not raise one."""
    with pytest.raises(ValueError) as caught:
        call_with(function, first, second, settings)
    return str(caught.value)


def refusal_of_reader(path):
    """Return the ValueError text outputs.read_measurements gives for a path, failing if it does not raise one."""
    with pytest.raises(ValueError) as caught:
        outputs.read_measurements(path)
    return str(caught.value)


def call_with(function, first, second, settings=None):
    """Call function(first, second) with the named placement settings from a dictionary, or none."""
    if settings is None:
        return function(first, second)
    return function(
        first,
        second,
        mode=settings.get("mode", "random"),
        max_tries=settings.get("max_tries", 1),
        buffer_um=settings.get("buffer_um", 0.0),
        pixel_um=settings.get("pixel_um", 1.0),
        order=settings.get("order", "as_given"),
        seed=settings.get("seed"),
    )


def dense_run(path, slate=None, buffer_um=0.0, pixel_um=1.0):
    """Run 40 particles of diameter 30 on a 150 micrometre square (or the given slate) with 60 tries each."""
    write_csv(path, [30.0] * 40)
    if slate is None:
        slate = {"shape": "square", "side_um": 150.0}
    return place_particles(path, slate, max_tries=60, seed=3, buffer_um=buffer_um, pixel_um=pixel_um)


# ---------------------------------------------------------------- input


def test_generator_result_and_its_csv_give_equal_records(tmp_path):
    """The same particles from a DatasetResult and from its written CSV (text path and Path) give equal records."""
    result = make_generated()
    outputs.write_outputs(result, tmp_path / "data.csv")
    from_result, _ = placement.read_particles(result)
    from_text, _ = placement.read_particles(str(tmp_path / "data.csv"))
    from_path, _ = placement.read_particles(tmp_path / "data.csv")
    assert from_result == from_text == from_path
    assert len(from_result) == 5
    for position, row in enumerate(result.rows):
        particle = from_result[position]
        assert particle.source_index == position + 1
        assert particle.object_id == row.object_id
        assert particle.feature_label == row.feature_label
        assert particle.equivalent_diameter_um == row.equivalent_diameter_um
        assert particle.radius_um == row.equivalent_diameter_um / 2.0


def test_csv_identity_is_the_digest_of_the_file_bytes(tmp_path):
    """A CSV's identity holds its file name and the SHA-256 of its bytes, whatever the extra columns."""
    path = write_csv(tmp_path / "sizes.csv", [10.0, 20.0])
    _, identity = placement.read_particles(path)
    assert identity == {
        "kind": "csv",
        "file_name": "sizes.csv",
        "measurements_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    _, text_identity = placement.read_particles(os.fspath(path))
    assert text_identity == identity
    path.write_bytes(path.read_bytes() + b"Class A,P3,1.0,5.0,1.0\n")
    _, changed = placement.read_particles(path)
    assert changed["measurements_sha256"] != identity["measurements_sha256"]


def test_generator_identity_holds_its_parameters_and_the_csv_digest():
    """A generator result's identity holds its reproduction parameters and the digest of serialize_csv."""
    result = make_generated()
    _, identity = placement.read_particles(result)
    assert identity == {
        "kind": "generator",
        "generator_parameters": outputs.reproduction_parameters(result),
        "measurements_sha256": hashlib.sha256(outputs.serialize_csv(result).encode("utf-8")).hexdigest(),
    }
    assert identity["generator_parameters"]["seed"] == 5


def test_reproduction_block_is_built_from_the_extracted_parameters():
    """The summary's reproduction block holds exactly the dictionary reproduction_parameters returns."""
    result = make_generated()
    block = outputs.build_reproduction_block(result)
    assert block[0] == outputs.BEGIN_MARKER
    assert block[2] == outputs.END_MARKER
    assert json.loads(block[1]) == json.loads(json.dumps(outputs.reproduction_parameters(result)))
    assert set(outputs.reproduction_parameters(result)) == set(outputs.PARAMETER_KEYS)
    assert outputs.reproduction_parameters(result)["library_versions"] == outputs.current_library_versions()


def test_csv_parsing_belongs_to_the_shared_reader(tmp_path, monkeypatch):
    """Replacing outputs.parse_measurements with a function that raises makes a CSV run raise it."""
    path = write_csv(tmp_path / "sizes.csv", [10.0])

    def raising(text):
        """Raise the marker instead of parsing."""
        raise Marker("parse reached")

    monkeypatch.setattr(outputs, "parse_measurements", raising)
    with pytest.raises(Marker):
        place_particles(path, SQUARE_200)


def test_a_csv_refusal_is_the_readers_own_message(tmp_path):
    """A missing column, a missing file and a bad diameter give the text read_measurements gives for that path."""
    path = tmp_path / "bad.csv"
    path.write_text("object_id,feature_label\nP1,A\n", encoding="utf-8")
    expected = refusal_of_reader(path)
    assert "equivalent_diameter_um" in expected
    assert refusal(place_particles, path, SQUARE_200) == expected
    missing = tmp_path / "absent.csv"
    assert refusal(place_particles, str(missing), SQUARE_200) == refusal_of_reader(str(missing))
    bad_value = write_csv(tmp_path / "zero.csv", [0.0])
    assert refusal(place_particles, bad_value, SQUARE_200) == refusal_of_reader(bad_value)


def test_other_particle_inputs_are_refused(tmp_path):
    """A list, a number, None and a bytes path are refused with a ValueError."""
    for bad in ([10.0, 20.0], [{"equivalent_diameter_um": 10.0}], 5, 2.5, None, b"sizes.csv"):
        assert "particles" in refusal(place_particles, bad, SQUARE_200)


def test_header_only_input_gives_no_rows_and_zero_coverage(tmp_path, capsys):
    """A header-only CSV runs, places nothing and requests a coverage of 0.0."""
    path = tmp_path / "empty.csv"
    path.write_text(HEADER, encoding="utf-8")
    result = place_particles(path, SQUARE_200, seed=1)
    assert result.rows == ()
    assert result.requested_coverage == 0.0
    assert result.achieved_coverage == 0.0
    assert capsys.readouterr().err == COVERAGE_LINE.format(0.0)


def test_a_run_returns_a_frozen_result_and_writes_nothing(tmp_path, monkeypatch):
    """The result is frozen, its fields are the settings-block and placements-file names, and no file appears."""
    monkeypatch.chdir(tmp_path)
    result = place_particles(make_generated(), SQUARE_200, max_tries=3, buffer_um=2, seed=9)
    assert os.listdir(tmp_path) == []
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.seed = 1
    names = [field.name for field in dataclasses.fields(result)]
    assert names[: len(SETTING_NAMES)] == SETTING_NAMES
    assert [field.name for field in dataclasses.fields(result.rows[0])] == ROW_COLUMNS
    assert result.slate == SQUARE_200
    assert result.mode == "random"
    assert result.max_tries == 3
    assert result.buffer_um == 2.0
    assert result.pixel_um == 1.0
    assert result.order == "as_given"
    assert result.seed == 9
    assert result.input["kind"] == "generator"
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.rows[0].x_um = 0.0


# ---------------------------------------------------------------- order and streams


def test_order_and_location_streams_are_independent(tmp_path, capsys):
    """30 equal particles give identical outcomes by placement order for as_given and shuffled; only the source indices differ."""
    path = write_csv(tmp_path / "equal.csv", [20.0] * 30)
    given = place_particles(path, SQUARE_200, max_tries=20, order="as_given", seed=17)
    shuffled = place_particles(path, SQUARE_200, max_tries=20, order="shuffled", seed=17)
    assert [row_key(row) for row in given.rows] == [row_key(row) for row in shuffled.rows]
    assert sum(row.edge_rejections + row.overlap_rejections for row in given.rows) > 0
    assert [row.source_index for row in given.rows] == list(range(1, 31))
    shuffled_sources = [row.source_index for row in shuffled.rows]
    assert shuffled_sources != list(range(1, 31))
    assert sorted(shuffled_sources) == list(range(1, 31))
    assert [row.placement_order for row in shuffled.rows] == list(range(1, 31))
    assert shuffled.achieved_coverage == given.achieved_coverage
    assert shuffled.requested_coverage == given.requested_coverage


def test_the_shuffle_is_child_zero_and_locations_are_child_one(tmp_path):
    """The order is the permutation of child 0 and the first-try locations are the draws of child 1."""
    path = write_csv(tmp_path / "sparse.csv", [2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    slate = {"shape": "square", "side_um": 1000.0}
    children = np.random.SeedSequence(44).spawn(2)
    result = place_particles(path, slate, order="shuffled", seed=44)
    expected_order = [int(value) + 1 for value in np.random.default_rng(children[0]).permutation(6)]
    assert [row.source_index for row in result.rows] == expected_order
    location_stream = np.random.default_rng(children[1])
    for row in result.rows:
        x, y = geometry.random_point(slate, location_stream)
        assert row.tries == 1
        assert (row.x_um, row.y_um) == (x, y)


def test_as_given_places_in_input_order(tmp_path):
    """With as_given placement order k takes source index k and the object ids follow the file."""
    path = write_csv(tmp_path / "three.csv", [10.0, 20.0, 30.0])
    result = place_particles(path, SQUARE_200, seed=2)
    assert [row.source_index for row in result.rows] == [1, 2, 3]
    assert [row.object_id for row in result.rows] == ["P1", "P2", "P3"]
    assert [row.equivalent_diameter_um for row in result.rows] == [10.0, 20.0, 30.0]
    assert [row.feature_label for row in result.rows] == ["Class A"] * 3


# ---------------------------------------------------------------- settings


def test_each_refused_setting_raises_before_anything_is_printed(tmp_path, capsys):
    """Every refusal is a ValueError naming the setting, raised before the coverage line is printed."""
    path = write_csv(tmp_path / "sizes.csv", [10.0])
    cases = [
        ("slate", {"slate": {"shape": "hexagon"}}),
        ("slate", {"slate": [1, 2]}),
        ("mode", {"mode": "settle"}),
        ("mode", {"mode": 3}),
        ("max_tries", {"max_tries": 0}),
        ("max_tries", {"max_tries": -2}),
        ("max_tries", {"max_tries": True}),
        ("max_tries", {"max_tries": 2.0}),
        ("max_tries", {"max_tries": "3"}),
        ("buffer_um", {"buffer_um": -0.5}),
        ("buffer_um", {"buffer_um": math.nan}),
        ("buffer_um", {"buffer_um": math.inf}),
        ("buffer_um", {"buffer_um": True}),
        ("buffer_um", {"buffer_um": "1"}),
        ("buffer_um", {"buffer_um": 10**400}),
        ("pixel_um", {"pixel_um": 0}),
        ("pixel_um", {"pixel_um": -1.0}),
        ("pixel_um", {"pixel_um": math.nan}),
        ("pixel_um", {"pixel_um": False}),
        ("order", {"order": "largest_first"}),
        ("order", {"order": "smallest_first"}),
        ("order", {"order": "sorted"}),
        ("order", {"order": None}),
        ("seed", {"seed": -1}),
        ("seed", {"seed": True}),
        ("seed", {"seed": 1.5}),
        ("seed", {"seed": "7"}),
    ]
    for name, override in cases:
        slate = override.pop("slate", SQUARE_200)
        message = refusal(place_particles, path, slate, override)
        assert name in message, (override, message)
        assert capsys.readouterr().err == ""
    # A refused setting is reported even when the input is unreadable, so settings are checked first.
    assert "max_tries" in refusal(place_particles, tmp_path / "missing.csv", SQUARE_200, {"max_tries": 0})
    assert capsys.readouterr().err == ""


def test_check_settings_returns_checked_floats_and_keeps_a_missing_seed():
    """The shared check returns float numbers, the checked slate and None for no seed."""
    checked = placement.check_settings({"shape": "square", "side_um": 10}, "random", 4, 3, 2, "shuffled", None)
    assert checked == {
        "slate": {"shape": "square", "side_um": 10.0},
        "mode": "random",
        "max_tries": 4,
        "buffer_um": 3.0,
        "pixel_um": 2.0,
        "order": "shuffled",
        "seed": None,
    }
    assert isinstance(checked["buffer_um"], float)
    assert placement.check_settings(SQUARE_200, "random", 1, 0, 1, "as_given", 0)["seed"] == 0


# ---------------------------------------------------------------- seeds


def test_equal_seeds_give_equal_rows_and_another_seed_other_locations(tmp_path):
    """The same seed repeats every row; another seed changes the random locations."""
    path = write_csv(tmp_path / "sizes.csv", [20.0] * 10)
    first = place_particles(path, SQUARE_200, max_tries=5, seed=101)
    again = place_particles(path, SQUARE_200, max_tries=5, seed=101)
    other = place_particles(path, SQUARE_200, max_tries=5, seed=102)
    assert first == again
    assert [row.x_um for row in first.rows] != [row.x_um for row in other.rows]
    assert [row.y_um for row in first.rows] != [row.y_um for row in other.rows]


def test_an_unseeded_run_draws_and_records_a_seed_that_repeats_it(tmp_path, monkeypatch):
    """With no seed one comes from secrets.randbits(SEED_BITS), is recorded, and reproduces the rows."""
    path = write_csv(tmp_path / "sizes.csv", [20.0] * 10)
    asked = []

    def fixed_bits(bits):
        """Record the requested width and return a fixed number."""
        asked.append(bits)
        return 123456789

    monkeypatch.setattr(placement.secrets, "randbits", fixed_bits)
    unseeded = place_particles(path, SQUARE_200, max_tries=5)
    assert asked == [dataset.SEED_BITS]
    assert unseeded.seed == 123456789
    repeated = place_particles(path, SQUARE_200, max_tries=5, seed=unseeded.seed)
    assert repeated.rows == unseeded.rows
    monkeypatch.undo()
    first = place_particles(path, SQUARE_200)
    second = place_particles(path, SQUARE_200)
    assert 0 <= first.seed < 2**dataset.SEED_BITS
    assert first.seed != second.seed


# ---------------------------------------------------------------- placement


def test_tries_and_rejection_counts_follow_the_outcome(tmp_path):
    """A placed particle has tries = rejections + 1; an unplaced one has tries = max_tries = rejections and no coordinates."""
    result = dense_run(tmp_path / "dense.csv")
    placed = [row for row in result.rows if row.placed]
    unplaced = [row for row in result.rows if not row.placed]
    assert placed and unplaced
    for row in placed:
        assert row.tries == row.edge_rejections + row.overlap_rejections + 1
        assert isinstance(row.x_um, float) and isinstance(row.y_um, float)
        assert 1 <= row.tries <= 60
    for row in unplaced:
        assert row.tries == 60 == row.edge_rejections + row.overlap_rejections
        assert row.x_um is None and row.y_um is None
    assert sum(row.edge_rejections for row in result.rows) > 0
    assert sum(row.overlap_rejections for row in result.rows) > 0


def test_placed_particles_are_inside_and_clear_of_each_other(tmp_path):
    """Every placed pair is at least r1 + r2 + buffer apart and every placed particle is inside the slate."""
    slates = [
        {"shape": "square", "side_um": 150.0},
        {"shape": "rectangle", "width_um": 200.0, "height_um": 120.0},
        {"shape": "circle", "diameter_um": 160.0},
    ]
    for slate in slates:
        for buffer_um in (0.0, 4.0):
            result = dense_run(tmp_path / "dense.csv", slate=slate, buffer_um=buffer_um)
            placed = [row for row in result.rows if row.placed]
            assert len(placed) >= 5
            for row in placed:
                assert geometry.edge_distance(result.slate, row.x_um, row.y_um) >= row.equivalent_diameter_um / 2.0
            for first_number, first in enumerate(placed):
                for second in placed[first_number + 1 :]:
                    distance = math.hypot(first.x_um - second.x_um, first.y_um - second.y_um)
                    radii = (first.equivalent_diameter_um + second.equivalent_diameter_um) / 2.0
                    assert distance >= radii + buffer_um


def test_a_crowded_slate_gives_overlap_rejections_and_an_unplaced_second_particle(tmp_path):
    """Two particles that cannot both fit: the second is rejected for overlap on every candidate that is inside."""
    path = write_csv(tmp_path / "pair.csv", [90.0, 90.0])
    result = place_particles(path, {"shape": "circle", "diameter_um": 100.0}, max_tries=3000, seed=8)
    first, second = result.rows
    assert first.placed
    assert not second.placed
    assert second.tries == 3000 == second.edge_rejections + second.overlap_rejections
    assert second.overlap_rejections > 0
    assert second.edge_rejections > 0


def test_a_particle_wider_than_the_slate_never_stops_the_run(tmp_path):
    """A 500 um particle on a 300 um slate gets max_tries edge rejections and a later small particle is placed."""
    path = write_csv(tmp_path / "wide.csv", [500.0, 20.0])
    result = place_particles(path, {"shape": "square", "side_um": 300.0}, max_tries=7, seed=4)
    wide, small = result.rows
    assert not wide.placed
    assert (wide.tries, wide.edge_rejections, wide.overlap_rejections) == (7, 7, 0)
    assert small.placed
    assert small.tries >= 1
    assert result.achieved_coverage == math.pi * 20.0 * 20.0 / 4.0 / 90000.0


def test_one_try_is_one_shot_placement_through_the_same_path(tmp_path):
    """The default max_tries of 1 gives each particle one candidate; the first try either places it or rejects it."""
    path = write_csv(tmp_path / "shot.csv", [20.0] * 30)
    result = place_particles(path, SQUARE_200, seed=6)
    assert result.max_tries == 1
    for row in result.rows:
        assert row.tries == 1
        if row.placed:
            assert row.edge_rejections == 0 and row.overlap_rejections == 0
        else:
            assert row.edge_rejections + row.overlap_rejections == 1
    assert any(row.placed for row in result.rows)
    assert not all(row.placed for row in result.rows)


def test_pixel_size_does_not_change_random_placement(tmp_path):
    """pixel_um 1 and 7 give identical rows."""
    first = dense_run(tmp_path / "dense.csv", pixel_um=1.0)
    coarse = dense_run(tmp_path / "dense.csv", pixel_um=7.0)
    assert first.rows == coarse.rows
    assert first.pixel_um == 1.0 and coarse.pixel_um == 7.0


def test_a_buffer_never_changes_diameters_or_coverage(tmp_path):
    """The buffer keeps recorded diameters and both coverages equal to the true-size values."""
    path = write_csv(tmp_path / "sparse.csv", [20.0, 30.0, 40.0])
    slate = {"shape": "square", "side_um": 10000.0}
    plain = place_particles(path, slate, seed=12)
    buffered = place_particles(path, slate, buffer_um=500.0, seed=12)
    expected = (math.pi * 20.0 * 20.0 / 4.0 + math.pi * 30.0 * 30.0 / 4.0 + math.pi * 40.0 * 40.0 / 4.0) / 10000.0**2
    assert all(row.placed for row in buffered.rows)
    assert [row.equivalent_diameter_um for row in buffered.rows] == [20.0, 30.0, 40.0]
    assert plain.requested_coverage == buffered.requested_coverage == pytest.approx(expected, rel=1e-12)
    assert plain.achieved_coverage == buffered.achieved_coverage == pytest.approx(expected, rel=1e-12)
    # With a buffer that crowds the slate, achieved coverage is still the placed true area.
    crowded = dense_run(tmp_path / "dense.csv", buffer_um=12.0)
    placed_count = sum(1 for row in crowded.rows if row.placed)
    assert crowded.achieved_coverage == pytest.approx(placed_count * math.pi * 30.0 * 30.0 / 4.0 / 150.0**2, rel=1e-12)
    assert all(row.equivalent_diameter_um == 30.0 for row in crowded.rows)


# ---------------------------------------------------------------- coverage line


def test_the_coverage_line_is_printed_once_before_placing(tmp_path, capsys):
    """Exactly one stderr line gives the requested coverage, equal to the kept value."""
    path = write_csv(tmp_path / "two.csv", [20.0, 30.0])
    slate = {"shape": "square", "side_um": 100.0}
    result = place_particles(path, slate, seed=1)
    expected = (math.pi * 20.0 * 20.0 / 4.0 + math.pi * 30.0 * 30.0 / 4.0) / 10000.0
    assert capsys.readouterr().err == COVERAGE_LINE.format(expected)
    assert result.requested_coverage == expected


def test_the_line_is_already_printed_when_the_first_candidate_is_judged(tmp_path, monkeypatch, capsys):
    """Replacing geometry.classify_candidate with a raising function leaves the exact coverage line on stderr."""
    path = write_csv(tmp_path / "two.csv", [20.0, 30.0])
    slate = {"shape": "square", "side_um": 100.0}

    def raising(slate, x, y, radius, placed_x, placed_y, placed_radius, buffer_um):
        """Raise the marker instead of judging the candidate."""
        raise Marker("judge reached")

    monkeypatch.setattr(geometry, "classify_candidate", raising)
    with pytest.raises(Marker):
        place_particles(path, slate, seed=1)
    expected = (math.pi * 20.0 * 20.0 / 4.0 + math.pi * 30.0 * 30.0 / 4.0) / 10000.0
    assert capsys.readouterr().err == COVERAGE_LINE.format(expected)


def test_a_requested_coverage_above_one_is_not_refused(tmp_path, capsys):
    """Twelve 40 um particles on a 100 um square request about 1.5 and the run still completes."""
    path = write_csv(tmp_path / "many.csv", [40.0] * 12)
    result = place_particles(path, {"shape": "square", "side_um": 100.0}, max_tries=3, seed=2)
    assert 1.4 < result.requested_coverage < 1.6
    assert len(result.rows) == 12
    assert result.achieved_coverage < 1.0
    assert capsys.readouterr().err == COVERAGE_LINE.format(result.requested_coverage)


# ---------------------------------------------------------------- reviewer additions


def test_the_hashed_bytes_are_the_parsed_bytes_even_if_the_file_changes_during_parsing(tmp_path, monkeypatch):
    """A file replaced while it is being parsed leaves an identity that matches the text that was parsed."""
    path = write_csv(tmp_path / "sizes.csv", [10.0, 20.0])
    original_bytes = path.read_bytes()
    real_parse = outputs.parse_measurements
    parsed_texts = []

    def parse_then_replace_file(text):
        """Parse the given text, then change the file on disk as a concurrent writer would."""
        parsed_texts.append(text)
        rows = real_parse(text)
        path.write_bytes(original_bytes + b"Class A,P3,1.0,5.0,1.0\n")
        return rows

    monkeypatch.setattr(outputs, "parse_measurements", parse_then_replace_file)
    _, identity = placement.read_particles(path)
    assert len(parsed_texts) == 1
    assert identity["measurements_sha256"] == hashlib.sha256(parsed_texts[0].encode("utf-8")).hexdigest()
    assert identity["measurements_sha256"] == hashlib.sha256(original_bytes).hexdigest()


def test_a_refused_csv_prints_no_coverage_line(tmp_path, capsys):
    """An unreadable or malformed CSV is refused before the coverage line is printed."""
    bad = tmp_path / "bad.csv"
    bad.write_text("object_id,feature_label\nP1,A\n", encoding="utf-8")
    for path in (bad, tmp_path / "absent.csv", tmp_path):
        with pytest.raises(ValueError):
            place_particles(path, SQUARE_200, seed=1)
        assert capsys.readouterr().err == ""


def test_a_recorded_seed_repeats_a_shuffled_run_with_its_source_order(tmp_path, monkeypatch):
    """With no seed and order shuffled, the recorded seed reproduces the source indices and every outcome."""
    path = write_csv(tmp_path / "mixed.csv", [12.0, 18.0, 24.0, 30.0, 36.0, 42.0, 48.0, 54.0])
    monkeypatch.setattr(placement.secrets, "randbits", lambda bits: 987654321)
    unseeded = place_particles(path, SQUARE_200, max_tries=15, order="shuffled")
    repeated = place_particles(path, SQUARE_200, max_tries=15, order="shuffled", seed=unseeded.seed)
    assert [row.source_index for row in unseeded.rows] != list(range(1, 9))
    assert repeated == unseeded
