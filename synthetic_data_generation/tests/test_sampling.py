# This file tests the particle size sampling. The statistical tests use fixed
# seeds (a seed makes the random generator repeat the same numbers every run) and
# tolerances wide enough to be stable but narrow enough to fail when a parameter
# is misread, for example a gamma scale used as a rate or a flipped GEV sign.
# The sieve window and the threshold checks are called directly, because
# sampling alone can never land exactly on a limit.

import math
import statistics

import numpy
import pytest

from synthetic_data_generation import sampling

DEFAULT_CAP = 100000


def make_gaussian(mean, stdev, count):
    """Return a checked-style Gaussian entry dictionary."""
    return {"class": "A", "distribution": "gaussian", "N": count, "mean": mean, "stdev": stdev}


def make_gamma(shape, scale, count):
    """Return a checked-style gamma entry dictionary."""
    return {"class": "A", "distribution": "gamma", "N": count, "shape": shape, "scale": scale}


def make_gev(shape, loc, scale, count):
    """Return a checked-style GEV entry dictionary."""
    return {"class": "A", "distribution": "gev", "N": count, "shape": shape, "loc": loc, "scale": scale}


def draw(entry, seed, lower=None, upper=None, cap=DEFAULT_CAP):
    """Draw an entry's values with a generator built from a fixed seed."""
    rng = numpy.random.default_rng(seed)
    return sampling.draw_entry_values(entry, rng, lower, upper, cap, 1)


def test_values_are_python_floats_finite_and_exactly_n():
    """Each distribution returns exactly N built-in floats, all finite and above zero."""
    entries_to_draw = [make_gaussian(100.0, 10.0, 50), make_gamma(2.0, 3.0, 50), make_gev(0.1, 50.0, 5.0, 50)]
    for entry in entries_to_draw:
        values = draw(entry, 3)
        assert len(values) == 50
        for value in values:
            assert type(value) is float
            assert math.isfinite(value)
            assert value > 0


def test_gaussian_with_mostly_negative_mass_is_all_positive():
    """A bare draw of mean 1, stdev 5 is about 42% negative, so the > 0 check must remove them."""
    values = draw(make_gaussian(1.0, 5.0, 5000), 11)
    assert len(values) == 5000
    assert min(values) > 0


def test_gaussian_sample_mean_and_stdev():
    """Gaussian mean 100, stdev 10 gives a sample mean near 100 and a sample stdev near 10."""
    values = draw(make_gaussian(100.0, 10.0, 5000), 12)
    assert abs(statistics.fmean(values) - 100.0) < 1.0
    assert abs(statistics.stdev(values) - 10.0) < 0.7


def test_gamma_scale_is_a_scale_not_a_rate():
    """Gamma shape 2, scale 3 has mean 6; reading scale as a rate would give about 0.67."""
    values = draw(make_gamma(2.0, 3.0, 5000), 13)
    assert abs(statistics.fmean(values) - 6.0) < 0.4


def test_gev_mean_matches_textbook_formula():
    """GEV shape 0.1, loc 50, scale 5 has the textbook mean near 53.43; a flipped sign gives about 52.43."""
    shape, loc, scale = 0.1, 50.0, 5.0
    expected = loc + scale * (math.gamma(1 - shape) - 1) / shape
    values = draw(make_gev(shape, loc, scale, 5000), 14)
    assert abs(statistics.fmean(values) - expected) < 0.6


def test_gev_sign_is_pinned_by_the_support_bound():
    """GEV shape 0.5 has a lower bound of loc - scale/shape = 8 and no upper bound at 12."""
    values = draw(make_gev(0.5, 10.0, 1.0, 2000), 15)
    assert min(values) >= 8.0
    assert max(values) > 12.0


def test_window_accepts_limits_and_rejects_the_next_float_outside():
    """The window [8, 12] keeps 8 and 12 exactly and rejects the floats just outside."""
    assert sampling.is_inside_window(8.0, 8.0, 12.0)
    assert sampling.is_inside_window(12.0, 8.0, 12.0)
    assert not sampling.is_inside_window(math.nextafter(8, 0), 8.0, 12.0)
    assert not sampling.is_inside_window(math.nextafter(12, 13), 8.0, 12.0)


def test_window_with_one_or_no_limit():
    """A lone lower limit keeps everything at or above it, a lone upper everything at or below, none keeps all."""
    for value in (8.0, 9.5, 1e300):
        assert sampling.is_inside_window(value, 8.0, None)
    assert not sampling.is_inside_window(math.nextafter(8, 0), 8.0, None)
    for value in (12.0, 9.5, 1e-300):
        assert sampling.is_inside_window(value, None, 12.0)
    assert not sampling.is_inside_window(math.nextafter(12, 13), None, 12.0)
    for value in (1e-300, 5.0, 1e300):
        assert sampling.is_inside_window(value, None, None)


def test_windowed_draw_returns_n_values_inside_the_window():
    """A Gaussian of mean 10, stdev 4 with window [8, 12] returns exactly 2000 values inside it."""
    values = draw(make_gaussian(10.0, 4.0, 2000), 16, lower=8.0, upper=12.0)
    assert len(values) == 2000
    for value in values:
        assert 8.0 <= value <= 12.0


def test_valid_thresholds_are_accepted():
    """A lower limit of 0, either limit omitted, and both given are accepted and returned as floats."""
    assert sampling.validate_thresholds(0, None) == (0.0, None)
    assert sampling.validate_thresholds(None, None) == (None, None)
    assert sampling.validate_thresholds(None, 5) == (None, 5.0)
    assert sampling.validate_thresholds(8, 12.5) == (8.0, 12.5)


def test_invalid_thresholds_are_rejected_with_a_message():
    """Each bad pair raises ValueError with a message naming the problem."""
    bad_pairs = [
        ((5, 5), "below threshold_upper"),
        ((6, 5), "below threshold_upper"),
        ((-1, None), "threshold_lower must be 0 or more"),
        ((None, 0), "threshold_upper must be above 0"),
        ((None, -2), "threshold_upper must be above 0"),
    ]
    for name in ("threshold_lower", "threshold_upper"):
        for bad in (math.nan, math.inf, -math.inf, True, "5"):
            pair = (bad, None) if name == "threshold_lower" else (None, bad)
            bad_pairs.append((pair, name))
    for pair, expected in bad_pairs:
        with pytest.raises(ValueError) as caught:
            sampling.validate_thresholds(pair[0], pair[1])
        assert expected in str(caught.value), pair


def test_unreachable_window_raises_sampling_error_naming_the_entry():
    """A window far from the distribution exhausts the cap and the error names every detail."""
    entry = make_gaussian(10, 1, 3)
    rng = numpy.random.default_rng(17)
    with pytest.raises(sampling.SamplingError) as caught:
        sampling.draw_entry_values(entry, rng, 500, 600, 50, 7)
    message = str(caught.value)
    for part in ("entry 7", "gaussian", "'A'", "mean 10", "stdev 1", "500", "600", "50"):
        assert part in message, part


def test_cap_counts_consecutive_draws_for_one_point():
    """About 38% of draws are kept here; 2000 points need far more than 50 draws in total but each fits."""
    values = draw(make_gaussian(10.0, 4.0, 2000), 18, lower=8.0, upper=12.0, cap=50)
    assert len(values) == 2000
