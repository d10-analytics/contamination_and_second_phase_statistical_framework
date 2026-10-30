# This file builds a whole synthetic dataset in memory. It takes the checked
# entries (one per particle class), draws every entry's particle diameters,
# derives the area and volume columns, gives each particle an id and, when asked,
# shuffles the rows. Nothing is printed and no file is touched here; writing
# files and the command line live elsewhere. The default settings are defined
# once, below, so the function signature, the summary and the command line all
# agree. A "seed" is a whole number that makes the random draws repeat exactly.

import math
import secrets
from dataclasses import dataclass
from types import MappingProxyType

import numpy

from synthetic_data_generation import entries as entries_module
from synthetic_data_generation import sampling

# The one set of default settings. Every other place that needs a default reads
# these constants instead of repeating the numbers.
DEFAULT_MAX_ATTEMPTS = 100_000
DEFAULT_SHUFFLE = False
DEFAULT_THRESHOLD_LOWER = None
DEFAULT_THRESHOLD_UPPER = None

# The columns of every row, in the order they are written.
COLUMN_NAMES = (
    "feature_label",
    "object_id",
    "contour_area_um2",
    "equivalent_diameter_um",
    "volume_um3",
)

# The two ways the entries can be supplied to generate_dataset.
SOURCE_TEXT = "text"
SOURCE_LIST = "list"

# A random seed is drawn from this many bits when the caller gives none.
SEED_BITS = 64

# Object ids use at least this many digits for the particle counter.
ID_DIGITS = 5


@dataclass(frozen=True)
class Row:
    """One synthetic particle measurement; the field names are the column names."""

    feature_label: str
    object_id: str
    contour_area_um2: float
    equivalent_diameter_um: float
    volume_um3: float


@dataclass(frozen=True)
class ClassCount:
    """The count of one class: the N particles generated for it and written to the CSV, with its label such as C1."""

    label: str
    class_name: str
    count: int


@dataclass(frozen=True)
class DatasetResult:
    """The generated rows plus every setting used, so a run can be described and repeated."""

    columns: tuple
    rows: tuple
    entries: tuple
    threshold_lower: object
    threshold_upper: object
    seed: int
    max_attempts: int
    shuffle: bool
    class_counts: tuple
    input_source: str


def validate_seed(seed):
    """Return the seed when it is a non-negative whole number (not a boolean); otherwise raise ValueError."""
    # Booleans are refused first because Python treats True as the integer 1.
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"seed must be a whole number of 0 or more, not {seed!r}")
    if seed < 0:
        raise ValueError(f"seed must be a whole number of 0 or more, not {seed!r}")
    return seed


def validate_max_attempts(max_attempts):
    """Return max_attempts when it is a whole number of at least 1 (not a boolean); otherwise raise ValueError."""
    if isinstance(max_attempts, bool) or not isinstance(max_attempts, int):
        raise ValueError(f"max_attempts must be a whole number of 1 or more, not {max_attempts!r}")
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be a whole number of 1 or more, not {max_attempts!r}")
    return max_attempts


def validate_shuffle(shuffle):
    """Return shuffle when it is exactly True or False; 0, 1 and strings are refused with ValueError."""
    if shuffle is not True and shuffle is not False:
        raise ValueError(f"shuffle must be True or False, not {shuffle!r}")
    return shuffle


def prepare_entries(entries):
    """Return (checked entries, source) for entry text or for a list of entry dictionaries."""
    if isinstance(entries, str):
        return entries_module.parse_entries(entries), SOURCE_TEXT
    return entries_module.validate_entries(entries), SOURCE_LIST


def freeze_entries(checked_entries):
    """Return the entries as read-only mappings, so a finished run's record cannot be edited."""
    frozen = []
    for entry in checked_entries:
        frozen.append(MappingProxyType(entry))
    return tuple(frozen)


def format_object_id(class_number, particle_number):
    """Return an id such as C1_o00001; the counter is padded to five digits and widens beyond them."""
    return f"C{class_number}_o{particle_number:0{ID_DIGITS}d}"


def derive_row(class_name, object_id, diameter):
    """Return a Row whose area and volume are computed from the drawn diameter."""
    return Row(
        feature_label=class_name,
        object_id=object_id,
        contour_area_um2=math.pi * diameter**2 / 4,
        equivalent_diameter_um=diameter,
        volume_um3=math.pi * diameter**3 / 6,
    )


def derive_checked_row(class_name, object_id, diameter, entry_number):
    """Return derive_row's Row, or raise SamplingError when the area or volume does not fit in a float."""
    try:
        row = derive_row(class_name, object_id, diameter)
    except OverflowError:
        row = None
    # A product can also overflow silently to infinity, so finiteness is checked too.
    if row is None or not (math.isfinite(row.contour_area_um2) and math.isfinite(row.volume_um3)):
        raise sampling.SamplingError(
            f"entry {entry_number}, class {class_name!r}: the drawn diameter {diameter!r} is too large "
            "for its area or volume to be calculated; check the entry's distribution parameters"
        )
    return row


def build_rows(checked_entries, entry_streams, lower, upper, max_attempts):
    """Draw every entry and return (rows in entry order, per-class counts) with ids assigned."""
    class_numbers = {}
    particle_counters = {}
    rows = []
    for entry_index, entry in enumerate(checked_entries):
        class_name = entry["class"]
        if class_name not in class_numbers:
            class_numbers[class_name] = len(class_numbers) + 1
            particle_counters[class_name] = 0
        values = sampling.draw_entry_values(
            entry, entry_streams[entry_index], lower, upper, max_attempts, entry_index + 1
        )
        for diameter in values:
            particle_counters[class_name] += 1
            object_id = format_object_id(class_numbers[class_name], particle_counters[class_name])
            rows.append(derive_checked_row(class_name, object_id, diameter, entry_index + 1))
    class_counts = []
    for class_name, class_number in class_numbers.items():
        class_counts.append(ClassCount(f"C{class_number}", class_name, particle_counters[class_name]))
    return rows, class_counts


def generate_dataset(
    entries,
    *,
    threshold_lower=DEFAULT_THRESHOLD_LOWER,
    threshold_upper=DEFAULT_THRESHOLD_UPPER,
    seed=None,
    max_attempts=DEFAULT_MAX_ATTEMPTS,
    shuffle=DEFAULT_SHUFFLE,
):
    """Return a DatasetResult for the entries (JSON text or a list of dictionaries); no printing, no files."""
    # Everything is checked before any random stream exists or any value is drawn.
    checked_entries, source = prepare_entries(entries)
    lower, upper = sampling.validate_thresholds(threshold_lower, threshold_upper)
    if seed is not None:
        validate_seed(seed)
    validate_max_attempts(max_attempts)
    validate_shuffle(shuffle)
    if seed is None:
        seed = secrets.randbits(SEED_BITS)
    # One independent stream per entry plus a last one for the shuffle, so a
    # changed entry or the shuffle flag never moves another entry's draws.
    children = numpy.random.SeedSequence(seed).spawn(len(checked_entries) + 1)
    entry_streams = []
    for child in children[:-1]:
        entry_streams.append(numpy.random.default_rng(child))
    shuffle_stream = numpy.random.default_rng(children[-1])
    rows, class_counts = build_rows(checked_entries, entry_streams, lower, upper, max_attempts)
    if shuffle:
        order = shuffle_stream.permutation(len(rows))
        rows = [rows[int(position)] for position in order]
    return DatasetResult(
        columns=COLUMN_NAMES,
        rows=tuple(rows),
        entries=freeze_entries(checked_entries),
        threshold_lower=lower,
        threshold_upper=upper,
        seed=seed,
        max_attempts=max_attempts,
        shuffle=shuffle,
        class_counts=tuple(class_counts),
        input_source=source,
    )
