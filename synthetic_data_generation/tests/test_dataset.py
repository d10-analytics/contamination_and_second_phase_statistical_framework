# This file tests the dataset assembly: repeatable seeds, one independent random
# stream per entry, up-front checking of every setting, the derived area and
# volume columns, the object ids and the optional shuffle. Each test triggers the
# behaviour through generate_dataset itself rather than calling a helper. A
# "seed" is a whole number that makes the random draws repeat exactly.

import json
import math

import pytest

from synthetic_data_generation import dataset
from synthetic_data_generation import entries as entries_module
from synthetic_data_generation import sampling


def gaussian(class_name, count, mean=100.0, stdev=1.0):
    """Return a Gaussian entry dictionary."""
    return {"class": class_name, "distribution": "gaussian", "N": count, "mean": mean, "stdev": stdev}


def three_entries(first_mean):
    """Return the Gaussian, gamma and gev entries used to prove stream independence."""
    return [
        gaussian("A", 200, first_mean, 5.0),
        {"class": "B", "distribution": "gamma", "N": 30, "shape": 2.0, "scale": 3.0},
        {"class": "C", "distribution": "gev", "N": 30, "shape": 0.1, "loc": 50.0, "scale": 5.0},
    ]


def diameters_of(result, class_name):
    """Return the diameters of one class's rows, ordered by their object id counter."""
    pairs = []
    for row in result.rows:
        if row.feature_label == class_name:
            pairs.append((row.object_id, row.equivalent_diameter_um))
    pairs.sort()
    return [diameter for _identifier, diameter in pairs]


def object_ids(result):
    """Return the object ids in row order."""
    return [row.object_id for row in result.rows]


def abc_entries():
    """Return entries A(3), B(2), A(2) as dictionaries."""
    return [gaussian("A", 3), gaussian("B", 2), gaussian("A", 2)]


def test_same_seed_gives_equal_rows_and_other_seed_other_diameters():
    """Two runs with one seed match exactly; a different seed changes the diameters."""
    first = dataset.generate_dataset(three_entries(1), seed=7, shuffle=True)
    second = dataset.generate_dataset(three_entries(1), seed=7, shuffle=True)
    other = dataset.generate_dataset(three_entries(1), seed=8, shuffle=True)
    assert first.rows == second.rows
    assert diameters_of(first, "A") != diameters_of(other, "A")


def test_changing_entry_zero_leaves_other_entries_and_shuffle_order_unchanged():
    """Mean 1 to 2 on entry 0 moves only entry 0's diameters; the id order of the shuffle stays."""
    before = dataset.generate_dataset(three_entries(1), seed=42, shuffle=True)
    after = dataset.generate_dataset(three_entries(2), seed=42, shuffle=True)
    assert diameters_of(before, "A") != diameters_of(after, "A")
    assert diameters_of(before, "B") == diameters_of(after, "B")
    assert diameters_of(before, "C") == diameters_of(after, "C")
    assert object_ids(before) == object_ids(after)


def test_shuffle_flag_does_not_move_entry_streams():
    """Turning the shuffle on or off keeps every entry's diameters the same."""
    plain = dataset.generate_dataset(three_entries(1), seed=5, shuffle=False)
    shuffled = dataset.generate_dataset(three_entries(1), seed=5, shuffle=True)
    for class_name in ("A", "B", "C"):
        assert diameters_of(plain, class_name) == diameters_of(shuffled, class_name)


def test_no_seed_records_a_seed_that_replays_the_run():
    """Without a seed the result holds a 64-bit seed, and passing it back gives the same rows."""
    result = dataset.generate_dataset(three_entries(1))
    assert type(result.seed) is int
    assert 0 <= result.seed < 2**64
    replay = dataset.generate_dataset(three_entries(1), seed=result.seed)
    assert replay.rows == result.rows


def test_defaults_come_from_the_module_constants():
    """Omitted settings use the constants, and the impossible window reports the 100,000 cap."""
    result = dataset.generate_dataset([gaussian("A", 2)], seed=1)
    assert result.max_attempts == dataset.DEFAULT_MAX_ATTEMPTS == 100000
    assert result.shuffle is False
    assert result.threshold_lower is None
    assert result.threshold_upper is None
    with pytest.raises(sampling.SamplingError) as error:
        dataset.generate_dataset([gaussian("A", 1, 10.0, 1.0)], threshold_lower=500, threshold_upper=600, seed=1)
    assert "max_attempts 100000" in str(error.value)


def test_result_records_inputs_counts_and_source():
    """The result carries the settings used, per-class counts and how the entries were given."""
    entries = abc_entries()
    from_list = dataset.generate_dataset(entries, threshold_lower=1, threshold_upper=1000.5, seed=3, max_attempts=50)
    assert from_list.input_source == "list"
    assert from_list.threshold_lower == 1.0
    assert from_list.threshold_upper == 1000.5
    assert from_list.seed == 3
    assert from_list.max_attempts == 50
    counts = [(item.label, item.class_name, item.kept_count) for item in from_list.class_counts]
    assert counts == [("C1", "A", 5), ("C2", "B", 2)]
    assert len(from_list.entries) == 3
    from_text = dataset.generate_dataset(json.dumps(entries), seed=3)
    assert from_text.input_source == "text"
    assert from_text.rows == dataset.generate_dataset(entries, seed=3).rows
    with pytest.raises(AttributeError):
        from_text.seed = 4


def test_seed_is_rejected_with_the_validator_message():
    """Negative, boolean, float and string seeds fail with the shared message and no result."""
    for bad_seed in (-1, True, 1.5, "1"):
        with pytest.raises(ValueError) as error:
            dataset.generate_dataset([gaussian("A", 1)], seed=bad_seed)
        with pytest.raises(ValueError) as direct:
            dataset.validate_seed(bad_seed)
        assert str(error.value) == str(direct.value)
        assert "seed must be a whole number of 0 or more" in str(error.value)


def test_max_attempts_is_rejected_with_the_validator_message():
    """Zero, boolean and float attempt caps fail before any draw."""
    for bad_cap in (0, True, 2.5):
        with pytest.raises(ValueError) as error:
            dataset.generate_dataset([gaussian("A", 1)], max_attempts=bad_cap)
        with pytest.raises(ValueError) as direct:
            dataset.validate_max_attempts(bad_cap)
        assert str(error.value) == str(direct.value)
        assert "max_attempts must be a whole number of 1 or more" in str(error.value)


def test_shuffle_must_be_exactly_true_or_false():
    """The numbers 0 and 1 and strings are not accepted as shuffle values."""
    for bad_shuffle in (1, 0, "yes"):
        with pytest.raises(ValueError) as error:
            dataset.generate_dataset([gaussian("A", 1)], shuffle=bad_shuffle)
        with pytest.raises(ValueError) as direct:
            dataset.validate_shuffle(bad_shuffle)
        assert str(error.value) == str(direct.value)
        assert "shuffle must be True or False" in str(error.value)
    assert dataset.validate_shuffle(True) is True
    assert dataset.validate_shuffle(False) is False


def test_boolean_threshold_is_rejected():
    """A boolean threshold is refused with the sampling module's threshold message."""
    with pytest.raises(ValueError) as error:
        dataset.generate_dataset([gaussian("A", 1)], threshold_lower=True)
    with pytest.raises(ValueError) as direct:
        sampling.validate_thresholds(True, None)
    assert str(error.value) == str(direct.value)


def test_invalid_entry_message_matches_parse_entries():
    """A list with a bad second entry gives the same message as the equal JSON text."""
    bad_list = [gaussian("A", 1), gaussian("B", 0)]
    with pytest.raises(ValueError) as from_text:
        entries_module.parse_entries(json.dumps(bad_list))
    with pytest.raises(ValueError) as from_list:
        dataset.generate_dataset(bad_list, seed=1)
    assert str(from_list.value) == str(from_text.value)
    assert "entry 2" in str(from_list.value)


def test_inverted_and_bad_thresholds_raise_the_threshold_message_not_the_sampling_error():
    """Bad windows fail up front; an inverted window would otherwise exhaust the attempt cap."""
    cases = [(10, 5), (None, float("inf")), (-1, None)]
    for lower, upper in cases:
        with pytest.raises(ValueError) as error:
            dataset.generate_dataset(
                [gaussian("A", 5, 7.0, 1.0)],
                threshold_lower=lower,
                threshold_upper=upper,
                seed=1,
                max_attempts=3,
            )
        assert not isinstance(error.value, sampling.SamplingError)
        with pytest.raises(ValueError) as direct:
            sampling.validate_thresholds(lower, upper)
        assert str(error.value) == str(direct.value)


def test_validation_happens_before_any_stream_is_spawned(monkeypatch):
    """A bad setting must fail before the seed sequence is created or the sampler runs."""
    calls = []

    def record_draw(entry, rng, lower, upper, max_attempts, entry_number):
        """Record that a draw happened and return a placeholder."""
        calls.append(entry_number)
        return [1.0]

    monkeypatch.setattr(sampling, "draw_entry_values", record_draw)
    with pytest.raises(ValueError):
        dataset.generate_dataset([gaussian("A", 1)], shuffle="yes")
    with pytest.raises(ValueError):
        dataset.generate_dataset([gaussian("A", 1)], threshold_lower=3, threshold_upper=2)
    assert calls == []


def test_columns_labels_and_derived_values():
    """Columns are exactly the five names; area and volume are recomputed from each diameter."""
    entries = [gaussian("IN738", 20, 50.0, 5.0), gaussian('Contaminant "A", é', 20, 20.0, 2.0)]
    result = dataset.generate_dataset(entries, seed=9)
    assert result.columns == (
        "feature_label",
        "object_id",
        "contour_area_um2",
        "equivalent_diameter_um",
        "volume_um3",
    )
    labels = [row.feature_label for row in result.rows]
    assert labels == ["IN738"] * 20 + ['Contaminant "A", é'] * 20
    for row in result.rows:
        diameter = row.equivalent_diameter_um
        assert row.contour_area_um2 == pytest.approx(math.pi * diameter**2 / 4, rel=1e-12)
        assert row.volume_um3 == pytest.approx(math.pi * diameter**3 / 6, rel=1e-12)
        for value in (row.contour_area_um2, row.equivalent_diameter_um, row.volume_um3):
            assert type(value) is float
    assert not hasattr(result.rows[0], "lot")
    assert not hasattr(result.rows[0], "source_feature_id")


def test_derivation_matches_the_worked_example():
    """A diameter of 10 gives an area of 78.5398 and a volume of 523.5988."""
    row = dataset.derive_row("A", "C1_o00001", 10.0)
    assert round(row.contour_area_um2, 4) == 78.5398
    assert round(row.volume_um3, 4) == 523.5988


def test_ids_continue_the_class_counter_across_entries():
    """Entries A(3), B(2), A(2) without shuffle give the ids in entry order, with no lot prefix."""
    result = dataset.generate_dataset(abc_entries(), seed=1)
    assert object_ids(result) == [
        "C1_o00001",
        "C1_o00002",
        "C1_o00003",
        "C2_o00001",
        "C2_o00002",
        "C1_o00004",
        "C1_o00005",
    ]


def test_shuffle_permutes_rows_but_keeps_ids_with_their_diameters():
    """Shuffled rows change order, repeat per seed, and keep each id's diameter."""
    plain = dataset.generate_dataset(abc_entries(), seed=11, shuffle=False)
    shuffled = dataset.generate_dataset(abc_entries(), seed=11, shuffle=True)
    again = dataset.generate_dataset(abc_entries(), seed=11, shuffle=True)
    assert object_ids(shuffled) != object_ids(plain)
    assert shuffled.rows == again.rows
    plain_pairs = {(row.object_id, row.equivalent_diameter_um) for row in plain.rows}
    shuffled_pairs = {(row.object_id, row.equivalent_diameter_um) for row in shuffled.rows}
    assert shuffled_pairs == plain_pairs
    assert len(shuffled.rows) == 7


def test_counter_widens_past_five_digits():
    """One class of 100,000 particles ends at C1_o100000 with every id distinct."""
    result = dataset.generate_dataset([gaussian("A", 100000, 100.0, 1.0)], seed=2)
    identifiers = object_ids(result)
    assert identifiers[0] == "C1_o00001"
    assert identifiers[99998] == "C1_o99999"
    assert identifiers[-1] == "C1_o100000"
    assert len(set(identifiers)) == 100000
