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


def build_reproduction_block(result):
    """Return the lines from the begin marker to the end marker, holding the settings as JSON."""
    plain_entries = []
    for entry in result.entries:
        # Entries on a result are read-only views; json needs ordinary dictionaries.
        plain_entries.append(dict(entry))
    parameters = {
        "entries": plain_entries,
        "threshold_lower": result.threshold_lower,
        "threshold_upper": result.threshold_upper,
        "seed": result.seed,
        "max_attempts": result.max_attempts,
        "shuffle": result.shuffle,
        "library_versions": current_library_versions(),
    }
    return [BEGIN_MARKER, json.dumps(parameters, indent=2), END_MARKER]


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
        "Particles kept per class:",
    ]
    for class_count in result.class_counts:
        lines.append(f"{class_count.label}  {class_count.class_name}  {class_count.kept_count}")
    lines += [
        "",
        f"Measurements file: {csv_name!r}",
        f"Summary file: {summary_name!r}",
        "",
    ]
    lines += build_reproduction_block(result)
    return "\n".join(lines) + "\n"


def check_destinations(csv_path, overwrite=False):
    """Check the output paths before anything is generated or written; return the summary path or raise ValueError."""
    csv_path = Path(csv_path)
    if csv_path.suffix != CSV_SUFFIX:
        raise ValueError(f"the measurements file must have a name ending in '.csv', not {str(csv_path)!r}")
    if not csv_path.parent.is_dir():
        raise ValueError(f"the folder {str(csv_path.parent)!r} does not exist")
    summary_path = summary_path_for(csv_path)
    for destination in (csv_path, summary_path):
        if destination.is_dir():
            raise ValueError(f"{str(destination)!r} is a folder; choose a file name instead")
        if not os.path.lexists(destination):
            continue
        if not destination.is_file():
            raise ValueError(f"{str(destination)!r} exists and is not a regular file")
        if not overwrite:
            raise ValueError(f"{str(destination)!r} already exists; pass overwrite=True to replace it")
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


def write_outputs(result, csv_path, overwrite=False):
    """Write the CSV and its summary beside it, all or nothing as far as a failure can be detected."""
    csv_path = Path(csv_path)
    summary_path = check_destinations(csv_path, overwrite)
    csv_text = serialize_csv(result)
    summary_text = build_summary_text(result, csv_path.name, summary_path.name)
    csv_existed = csv_path.exists()
    csv_temporary = temporary_path_for(csv_path)
    summary_temporary = temporary_path_for(summary_path)
    try:
        write_temporary_file(csv_temporary, csv_text)
        write_temporary_file(summary_temporary, summary_text)
        os.replace(csv_temporary, csv_path)
        try:
            os.replace(summary_temporary, summary_path)
        except OSError as error:
            if csv_existed:
                raise OSError(
                    f"the summary {str(summary_path)!r} could not be written ({error}); "
                    f"the measurements file {str(csv_path)!r} was replaced and the summary was not"
                ) from error
            csv_path.unlink(missing_ok=True)
            raise
    finally:
        remove_files([csv_temporary, summary_temporary])


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


def read_summary(path):
    """Read a summary file and return its checked reproduction parameters as SummaryParameters."""
    text = Path(path).read_text(encoding="utf-8")
    block = load_block(text)
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
    result = dataset.generate_dataset(
        parameters.entries,
        threshold_lower=parameters.threshold_lower,
        threshold_upper=parameters.threshold_upper,
        seed=parameters.seed,
        max_attempts=parameters.max_attempts,
        shuffle=parameters.shuffle,
    )
    # An absolute path can never equal the plain words "text" or "list" that mark
    # entries given as text or as a list, even when the summary file is named so.
    return replace(result, input_source=str(Path(summary_path).resolve()))
