# This file writes a finished dataset to disk and reads it back. A run produces
# two files side by side: a measurements CSV and a text summary that records
# every setting, including a block of "reproduction parameters" (a small JSON
# object, which is plain text describing the settings). Feeding that summary
# back in recreates the CSV byte for byte. Both files are built fully in memory
# first and only then put in place, so a failed run leaves no half-written
# result. The one CSV writer and the one replay function live here so that the
# command line and the example both use exactly the same code.

import csv
import datetime
import io
import json
import math
import os
import platform
import sys
import uuid
from dataclasses import dataclass, replace
from pathlib import Path

import numpy
import scipy

from synthetic_data_generation import dataset
from synthetic_data_generation import entries as entries_module
from synthetic_data_generation import sampling

BEGIN_MARKER = "--- BEGIN REPRODUCTION PARAMETERS ---"
END_MARKER = "--- END REPRODUCTION PARAMETERS ---"

# The exact keys of the reproduction block, and of its library versions.
PARAMETER_KEYS = (
    "entries",
    "threshold_lower",
    "threshold_upper",
    "seed",
    "max_attempts",
    "shuffle",
    "library_versions",
)
LIBRARY_KEYS = ("python", "numpy", "scipy")

CSV_SUFFIX = ".csv"
SUMMARY_ENDING = "_summary.txt"


@dataclass(frozen=True)
class SummaryParameters:
    """The settings read back from a summary's reproduction block, already checked."""

    entries: list
    threshold_lower: object
    threshold_upper: object
    seed: int
    max_attempts: int
    shuffle: bool
    library_versions: dict


def current_library_versions():
    """Return the versions of Python, NumPy and SciPy that are running now."""
    return {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
    }


def serialize_csv(result):
    """Return the measurements CSV text: the header, then one line per row, with numbers as plain floats."""
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(result.columns)
    for row in result.rows:
        # float() first, because repr of a NumPy number would print "np.float64(...)".
        writer.writerow(
            [
                row.feature_label,
                row.object_id,
                repr(float(row.contour_area_um2)),
                repr(float(row.equivalent_diameter_um)),
                repr(float(row.volume_um3)),
            ]
        )
    return buffer.getvalue()


MEASUREMENT_COLUMNS = ("feature_label", "object_id", "equivalent_diameter_um")


def parse_measurements(text):
    """Return one dict per data row of a measurements CSV: feature_label, object_id (text) and equivalent_diameter_um (float).

    Other columns are ignored and column order does not matter. A missing
    required column, or a diameter that is not a finite number above zero,
    raises ValueError naming the column or the row's 1-based line in the file.
    """
    reader = csv.DictReader(io.StringIO(text, newline=""))
    header = reader.fieldnames or []
    for column in MEASUREMENT_COLUMNS:
        if column not in header:
            raise ValueError(f"the measurements header has no '{column}' column")
    rows = []
    for record in reader:
        line_number = reader.line_num
        for column in MEASUREMENT_COLUMNS:
            if record[column] is None:
                raise ValueError(f"line {line_number}: the row has no '{column}' value")
        cell = record["equivalent_diameter_um"]
        try:
            diameter = float(cell)
        except ValueError:
            diameter = float("nan")
        if not (math.isfinite(diameter) and diameter > 0):
            raise ValueError(f"line {line_number}: equivalent_diameter_um {cell!r} is not a finite number above 0")
        rows.append(
            {
                "feature_label": record["feature_label"],
                "object_id": record["object_id"],
                "equivalent_diameter_um": diameter,
            }
        )
    return rows


def load_measurements(path):
    """Check and read a measurements CSV once; return (its bytes, the rows parsed from exactly those bytes).

    Refusals name the path, as read_entries does. A caller that hashes the file
    uses these bytes so that it hashes what it parsed.
    """
    file_path = Path(path)
    if file_path.is_dir():
        raise ValueError(f"cannot read measurements: '{path}' is a directory, not a file")
    if not file_path.exists():
        raise ValueError(f"cannot read measurements: '{path}' does not exist")
    try:
        data = file_path.read_bytes()
        text = data.decode("utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError(f"cannot read measurements from '{path}' as UTF-8 text: {error}")
    try:
        return data, parse_measurements(text)
    except ValueError as error:
        raise ValueError(f"{path}: {error}")


def read_measurements(path):
    """Read a UTF-8 measurements CSV file and return the same rows that parse_measurements gives."""
    return load_measurements(path)[1]


def summary_path_for(csv_path):
    """Return the summary path that sits beside the CSV path: <stem>_summary.txt."""
    csv_path = Path(csv_path)
    return csv_path.with_name(csv_path.stem + SUMMARY_ENDING)


def describe_inputs(input_source):
    """Return the inputs line: text, a Python list, or the repr of an input file path."""
    if input_source == dataset.SOURCE_TEXT:
        return "Inputs: entries given as text"
    if input_source == dataset.SOURCE_LIST:
        return "Inputs: entries given as a Python list"
    # repr() keeps a line break in a path from starting a new line of the summary.
    return f"Inputs: {input_source!r}"


def describe_entry_line(entry_number, label, entry):
    """Return one line of the entries table; it starts with the class label."""
    parts = [f"{label}", f"entry {entry_number}", entry["class"], entry["distribution"], f"N={entry['N']}"]
    for key in entries_module.PARAMETER_KEYS[entry["distribution"]]:
        parts.append(f"{key}={entry[key]!r}")
    return "  ".join(parts)


def reproduction_parameters(result):
    """Return the settings dictionary that the reproduction block holds, library versions included."""
    plain_entries = []
    for entry in result.entries:
        # Entries on a result are read-only views; json needs ordinary dictionaries.
        plain_entries.append(dict(entry))
    return {
        "entries": plain_entries,
        "threshold_lower": result.threshold_lower,
        "threshold_upper": result.threshold_upper,
        "seed": result.seed,
        "max_attempts": result.max_attempts,
        "shuffle": result.shuffle,
        "library_versions": current_library_versions(),
    }


def build_reproduction_block(result):
    """Return the lines from the begin marker to the end marker, holding the settings as JSON."""
    return [BEGIN_MARKER, json.dumps(reproduction_parameters(result), indent=2), END_MARKER]


def build_summary_text(result, csv_name, summary_name):
    """Return the whole summary: a readable part followed by the reproduction block."""
    labels = {}
    for class_count in result.class_counts:
        labels[class_count.class_name] = class_count.label
    run_time = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    lines = [
        "Synthetic particle measurements: run summary",
        f"Run time: {run_time}",
        describe_inputs(result.input_source),
        "",
        "Entries:",
    ]
    for entry_number, entry in enumerate(result.entries, start=1):
        lines.append(describe_entry_line(entry_number, labels[entry["class"]], entry))
    versions = current_library_versions()
    lines += [
        "",
        f"Lower threshold: {result.threshold_lower!r}",
        f"Upper threshold: {result.threshold_upper!r}",
        f"Seed: {result.seed}",
        f"Maximum attempts: {result.max_attempts}",
        f"Shuffle: {result.shuffle}",
        f"Library versions: python {versions['python']}, numpy {versions['numpy']}, scipy {versions['scipy']}",
        "",
        "Particles per class (the count N of each class, as written to the CSV):",
    ]
    for class_count in result.class_counts:
        lines.append(f"{class_count.label}  {class_count.class_name}  {class_count.count}")
    lines += [
        "",
        f"Measurements file: {csv_name!r}",
        f"Summary file: {summary_name!r}",
        "",
    ]
    lines += build_reproduction_block(result)
    return "\n".join(lines) + "\n"


def check_paths(paths, overwrite, overwrite_hint):
    """Check each destination in order and raise ValueError for a folder, a non-regular file or an existing file.

    An existing regular file is refused unless overwrite is true; the message
    ends with "pass {overwrite_hint} to replace it".
    """
    for destination in paths:
        destination = Path(destination)
        if destination.is_dir():
            raise ValueError(f"{str(destination)!r} is a folder; choose a file name instead")
        if not os.path.lexists(destination):
            continue
        if not destination.is_file():
            raise ValueError(f"{str(destination)!r} exists and is not a regular file")
        if not overwrite:
            raise ValueError(f"{str(destination)!r} already exists; pass {overwrite_hint} to replace it")


def check_destinations(csv_path, overwrite=False):
    """Check the output paths before anything is generated or written; return the summary path or raise ValueError."""
    csv_path = Path(csv_path)
    if csv_path.suffix != CSV_SUFFIX:
        raise ValueError(f"the measurements file must have a name ending in '.csv', not {str(csv_path)!r}")
    if not csv_path.parent.is_dir():
        raise ValueError(f"the folder {str(csv_path.parent)!r} does not exist")
    summary_path = summary_path_for(csv_path)
    check_paths([csv_path, summary_path], overwrite, "overwrite=True (command line: --overwrite)")
    return summary_path


def temporary_path_for(destination):
    """Return a unique hidden name in the destination's folder for a file written before it is renamed."""
    return destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")


def write_temporary_file(path, text):
    """Create a new file at path (failing if it exists) and write the text with \\n line endings."""
    # An ordinary open gives the file the user's normal permissions after the rename.
    with open(path, "x", encoding="utf-8", newline="") as handle:
        handle.write(text)


def remove_files(paths):
    """Delete each file that exists; a missing file is fine."""
    for path in paths:
        path.unlink(missing_ok=True)


def describe_replaced(replaced):
    """Return the text that names each replaced (description, path) pair, in order, with the right verb."""
    names = []
    for description, path in replaced:
        names.append(f"the {description} {str(path)!r}")
    verb = "was" if len(names) == 1 else "were"
    return " and ".join(names) + " " + verb


def rename_into_place(planned, temporaries, existed):
    """Rename each temporary file onto its destination in order; undo or report as the contract says on a failure."""
    replaced = []
    created = []
    for position, (path, _text, description) in enumerate(planned):
        try:
            os.replace(temporaries[position], path)
        except OSError as error:
            remove_files(created)
            if len(replaced) == 0:
                raise
            raise OSError(
                f"the {description} {str(path)!r} could not be written ({error}); "
                f"{describe_replaced(replaced)} replaced and the {description} was not"
            ) from error
        if existed[position]:
            replaced.append((description, path))
        else:
            created.append(path)


def write_files_together(planned):
    """Write every (path, text, description) entry, all or nothing as far as a failure can be detected.

    Which destinations already exist is recorded first, every temporary file
    is written before any rename, and the temporary files are always removed.
    """
    located = []
    for path, text, description in planned:
        located.append((Path(path), text, description))
    existed = []
    temporaries = []
    for path, _text, _description in located:
        existed.append(path.exists())
        temporaries.append(temporary_path_for(path))
    try:
        for position, (_path, text, _description) in enumerate(located):
            write_temporary_file(temporaries[position], text)
        rename_into_place(located, temporaries, existed)
    finally:
        remove_files(temporaries)


def write_outputs(result, csv_path, overwrite=False):
    """Write the CSV and its summary beside it, all or nothing as far as a failure can be detected."""
    csv_path = Path(csv_path)
    summary_path = check_destinations(csv_path, overwrite)
    csv_text = serialize_csv(result)
    summary_text = build_summary_text(result, csv_path.name, summary_path.name)
    write_files_together([(csv_path, csv_text, "measurements file"), (summary_path, summary_text, "summary")])


def find_single_line(lines, marker, description):
    """Return the position of the one line equal to marker; raise ValueError for none or several."""
    positions = []
    for position, line in enumerate(lines):
        if line == marker:
            positions.append(position)
    if len(positions) == 0:
        raise ValueError(f"the summary has no {description} line {marker!r}")
    if len(positions) > 1:
        raise ValueError(f"the summary has {len(positions)} {description} lines {marker!r}; expected one block")
    return positions[0]


def load_block(text):
    """Return the JSON object between the one marker pair of a summary's text."""
    lines = text.split("\n")
    begin = find_single_line(lines, BEGIN_MARKER, "begin")
    end = find_single_line(lines, END_MARKER, "end")
    if end < begin:
        raise ValueError("the summary's end line comes before its begin line")
    block_text = "\n".join(lines[begin + 1 : end])
    try:
        block = json.loads(block_text)
    except json.JSONDecodeError as error:
        raise ValueError(f"the summary's reproduction parameters are not valid JSON: {error}") from error
    if not isinstance(block, dict):
        raise ValueError("the summary's reproduction parameters must be a JSON object")
    return block


def check_keys(found, expected, description):
    """Raise ValueError naming any missing or extra key of a mapping."""
    for key in expected:
        if key not in found:
            raise ValueError(f"{description} is missing the key {key!r}")
    for key in found:
        if key not in expected:
            raise ValueError(f"{description} has an unexpected key {key!r}")


def check_library_versions(versions):
    """Check the library_versions object and return it."""
    if not isinstance(versions, dict):
        raise ValueError("library_versions must be a JSON object")
    check_keys(versions, LIBRARY_KEYS, "library_versions")
    for key in LIBRARY_KEYS:
        if not isinstance(versions[key], str):
            raise ValueError(f"library_versions {key!r} must be text, not {versions[key]!r}")
    return versions


def check_reproduction_parameters(block):
    """Check a reproduction block (already loaded from JSON) and return its settings as SummaryParameters."""
    check_keys(block, PARAMETER_KEYS, "the summary's reproduction parameters")
    checked_entries = entries_module.validate_entries(block["entries"])
    lower, upper = sampling.validate_thresholds(block["threshold_lower"], block["threshold_upper"])
    seed = dataset.validate_seed(block["seed"])
    max_attempts = dataset.validate_max_attempts(block["max_attempts"])
    shuffle = dataset.validate_shuffle(block["shuffle"])
    versions = check_library_versions(block["library_versions"])
    return SummaryParameters(
        entries=checked_entries,
        threshold_lower=lower,
        threshold_upper=upper,
        seed=seed,
        max_attempts=max_attempts,
        shuffle=shuffle,
        library_versions=versions,
    )


def read_summary(path):
    """Read a summary file and return its checked reproduction parameters as SummaryParameters."""
    text = Path(path).read_text(encoding="utf-8")
    return check_reproduction_parameters(load_block(text))


def generate_from_parameters(parameters):
    """Generate the dataset that checked SummaryParameters describe."""
    return dataset.generate_dataset(
        parameters.entries,
        threshold_lower=parameters.threshold_lower,
        threshold_upper=parameters.threshold_upper,
        seed=parameters.seed,
        max_attempts=parameters.max_attempts,
        shuffle=parameters.shuffle,
    )


def warn_about_versions(recorded):
    """Print a warning on stderr for each library whose recorded version differs from the running one."""
    running = current_library_versions()
    for key in LIBRARY_KEYS:
        if recorded[key] != running[key]:
            print(
                f"warning: the summary was written with {key} {recorded[key]} but {key} {running[key]} "
                "is running; the output may differ from the original run",
                file=sys.stderr,
            )


def replay_summary(summary_path):
    """Recreate a run from its summary file and return the result, with the summary as its input source."""
    parameters = read_summary(summary_path)
    warn_about_versions(parameters.library_versions)
    result = generate_from_parameters(parameters)
    # An absolute path can never equal the plain words "text" or "list" that mark
    # entries given as text or as a list, even when the summary file is named so.
    return replace(result, input_source=str(Path(summary_path).resolve()))
