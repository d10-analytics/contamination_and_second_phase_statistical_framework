# This file draws the random particle sizes (equivalent diameters in micrometres)
# for one entry. It builds the entry's distribution (Gaussian, gamma or GEV, the
# generalised extreme value distribution) from SciPy, draws one size at a time
# and keeps a size only when it is a finite number above zero and inside the
# optional sieve window. A "sieve" is a pair of size limits (lower and upper,
# both inclusive) that every entry shares. Drawing one size at a time keeps the
# rule easy to read; speed is not a goal. If the window leaves almost no chance
# of a kept size, the loop gives up after a fixed number of consecutive tries
# and raises SamplingError instead of running forever.

import math

from scipy import stats


class SamplingError(RuntimeError):
    """Raised when a particle size cannot be drawn within the allowed number of attempts."""


def convert_threshold(name, value):
    """Return a sieve limit as a finite float, or raise ValueError naming the limit."""
    # Booleans are rejected first because Python treats True as the integer 1.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number, not {value!r}")
    try:
        number = float(value)
    except OverflowError:
        raise ValueError(f"{name} is too large to use as a number")
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number, not {value!r}")
    return number


def validate_thresholds(threshold_lower, threshold_upper):
    """Check the optional sieve limits and return them as (lower, upper), each a float or None."""
    lower = None
    upper = None
    if threshold_lower is not None:
        lower = convert_threshold("threshold_lower", threshold_lower)
        if lower < 0:
            raise ValueError(f"threshold_lower must be 0 or more, not {threshold_lower!r}")
    if threshold_upper is not None:
        upper = convert_threshold("threshold_upper", threshold_upper)
        if upper <= 0:
            raise ValueError(f"threshold_upper must be above 0, not {threshold_upper!r}")
    if lower is not None and upper is not None and lower >= upper:
        raise ValueError(
            f"threshold_lower must be below threshold_upper, got {threshold_lower!r} and {threshold_upper!r}"
        )
    return lower, upper


def is_inside_window(value, lower, upper):
    """Return True when value lies in the sieve window; both limits are inclusive and either may be None."""
    if lower is not None and value < lower:
        return False
    if upper is not None and value > upper:
        return False
    return True


def build_distribution(entry):
    """Return the frozen SciPy distribution (fixed parameters) described by a checked entry."""
    distribution = entry["distribution"]
    if distribution == "gaussian":
        return stats.norm(loc=entry["mean"], scale=entry["stdev"])
    if distribution == "gamma":
        # SciPy's "scale" is the spread (1 / rate), the same meaning as the entry's "scale".
        return stats.gamma(a=entry["shape"], scale=entry["scale"])
    if distribution == "gev":
        # SciPy's shape c has the opposite sign to the textbook shape used in entries.
        return stats.genextreme(c=-entry["shape"], loc=entry["loc"], scale=entry["scale"])
    raise ValueError(f"unknown distribution {distribution!r}")


def describe_window(lower, upper):
    """Return the sieve window as text for error messages."""
    return f"lower limit {lower!r}, upper limit {upper!r}"


def describe_entry(entry, entry_number, lower, upper, max_attempts):
    """Return the entry's number, class, distribution, parameters, window and cap as text."""
    parts = [f"entry {entry_number}", f"class {entry['class']!r}", f"distribution {entry['distribution']}"]
    for key, value in entry.items():
        if key not in ("class", "distribution", "N"):
            parts.append(f"{key} {value!r}")
    parts.append(describe_window(lower, upper))
    parts.append(f"max_attempts {max_attempts}")
    return ", ".join(parts)


def draw_one_value(frozen_distribution, rng, lower, upper, max_attempts):
    """Draw until one value is finite, above zero and inside the window; return None if the cap is hit."""
    for _attempt in range(max_attempts):
        value = float(frozen_distribution.rvs(random_state=rng))
        if math.isfinite(value) and value > 0 and is_inside_window(value, lower, upper):
            return value
    return None


def draw_entry_values(entry, rng, lower, upper, max_attempts, entry_number):
    """Return exactly entry['N'] kept sizes as Python floats, drawn with the given random generator.

    The attempt cap counts consecutive draws for one particle, so it restarts
    after each kept value. Exhausting it raises SamplingError naming the entry.
    """
    frozen_distribution = build_distribution(entry)
    values = []
    for _particle in range(entry["N"]):
        value = draw_one_value(frozen_distribution, rng, lower, upper, max_attempts)
        if value is None:
            raise SamplingError(
                "could not draw a particle size within the attempt cap: "
                + describe_entry(entry, entry_number, lower, upper, max_attempts)
            )
        values.append(value)
    return values
