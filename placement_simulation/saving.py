# This file saves a finished placement run. A run produces two files side by
# side: a placements CSV (one row per particle, in the order taken) and a text
# summary with a readable part and a machine-readable settings block (a small
# JSON object, which is plain text describing the settings). The destination
# checks and the all-or-nothing writing are the generator's own functions in
# its outputs module, so both packages refuse, write and clean up identically.
# Coordinates are in micrometres, the origin at the slate centre, x to the
# right and y up. Reading goes through the same module: the summary's settings
# check is written once here and used by replay and by the placements reader.

import csv
import io
import json
import math
import operator
import re
from pathlib import Path

from placement_simulation import placement
from synthetic_data_generation import outputs

PLACEMENTS_ENDING = "_placements.csv"
SUMMARY_ENDING = "_summary.txt"
OVERWRITE_HINT = "overwrite=True"

PLACEMENT_COLUMNS = (
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
)
# The exact keys of the summary's settings block, in the order they are written.
BLOCK_KEYS = ("slate", "mode", "max_tries", "buffer_um", "pixel_um", "order", "seed", "input", "library_versions")


def format_float(value):
    """Return a number as the text repr of a plain Python float."""
    return repr(float(value))


def serialize_placements(result):
    """Return the placements CSV text: the header, then one line per particle in placement order."""
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(PLACEMENT_COLUMNS)
    for row in result.rows:
        writer.writerow(
            [
                int(row.placement_order),
                int(row.source_index),
                row.object_id,
                row.feature_label,
                format_float(row.equivalent_diameter_um),
                "true" if row.placed else "false",
                format_float(row.x_um) if row.placed else "",
                format_float(row.y_um) if row.placed else "",
                int(row.tries),
                int(row.edge_rejections),
                int(row.overlap_rejections),
            ]
        )
    return buffer.getvalue()


def describe_input(identity):
    """Return the readable input line: the CSV file name and checksum, or a note that the generator settings were used."""
    checksum = identity["measurements_sha256"]
    if identity["kind"] == "csv":
        return f"Input: measurements file {identity['file_name']!r}, SHA-256 {checksum}"
    return f"Input: generator result with the generator settings in the block below, SHA-256 of its measurements {checksum}"


def settings_block(result):
    """Return the dictionary the summary's settings block holds, library versions included."""
    values = {
        "slate": result.slate,
        "mode": result.mode,
        "max_tries": result.max_tries,
        "buffer_um": result.buffer_um,
        "pixel_um": result.pixel_um,
        "order": result.order,
        "seed": result.seed,
        "input": result.input,
        "library_versions": outputs.current_library_versions(),
    }
    block = {}
    for key in BLOCK_KEYS:
        block[key] = values[key]
    return block


def build_summary_text(result, placements_name, summary_name):
    """Return the whole summary: a readable part followed by the settings block."""
    placed = 0
    for row in result.rows:
        if row.placed:
            placed += 1
    versions = outputs.current_library_versions()
    lines = [
        "Particle placement: run summary",
        f"Slate: {result.slate!r}",
        f"Mode: {result.mode}",
        f"Maximum tries: {result.max_tries}",
        f"Buffer (um): {format_float(result.buffer_um)}",
        f"Pixel size (um): {format_float(result.pixel_um)}",
        f"Order: {result.order}",
        f"Seed: {result.seed}",
        describe_input(result.input),
        f"Library versions: python {versions['python']}, numpy {versions['numpy']}, scipy {versions['scipy']}",
        "",
        f"Placed: {placed}",
        f"Not placed: {len(result.rows) - placed}",
        f"Requested coverage: {float(result.requested_coverage)!r}",
        f"Achieved coverage: {float(result.achieved_coverage)!r}",
        "",
        f"Placements file: {placements_name!r}",
        f"Summary file: {summary_name!r}",
        "",
        outputs.BEGIN_MARKER,
        json.dumps(settings_block(result), indent=2),
        outputs.END_MARKER,
    ]
    return "\n".join(lines) + "\n"


def check_placement_destinations(base_path, overwrite=False):
    """Check the two output paths before a long run; return (placements path, summary path) or raise ValueError."""
    base_path = Path(base_path)
    if base_path.name == "":
        raise ValueError(f"the base path {str(base_path)!r} has no file name to build the output names from")
    folder = base_path.parent
    if not folder.is_dir():
        raise ValueError(f"the folder {str(folder)!r} does not exist")
    placements_path = folder / (base_path.name + PLACEMENTS_ENDING)
    summary_path = folder / (base_path.name + SUMMARY_ENDING)
    outputs.check_paths([placements_path, summary_path], overwrite, OVERWRITE_HINT)
    return placements_path, summary_path


def write_placement_outputs(result, base_path, overwrite=False):
    """Write the placements CSV and the summary beside base_path, all or nothing; return both paths."""
    placements_path, summary_path = check_placement_destinations(base_path, overwrite)
    placements_text = serialize_placements(result)
    summary_text = build_summary_text(result, placements_path.name, summary_path.name)
    outputs.write_files_together(
        [(placements_path, placements_text, "placements file"), (summary_path, summary_text, "summary")]
    )
    return placements_path, summary_path


WHOLE_NUMBER = re.compile("[0-9]+")


def read_saved_text(path, description):
    """Return the UTF-8 text of a saved file; raise ValueError naming the file when it is missing, a folder or not UTF-8."""
    file_path = Path(path)
    if file_path.is_dir():
        raise ValueError(f"cannot read the {description} {str(file_path)!r}: it is a folder, not a file")
    if not file_path.exists():
        raise ValueError(f"cannot read the {description} {str(file_path)!r}: it does not exist")
    try:
        return file_path.read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError(f"cannot read the {description} {str(file_path)!r} as UTF-8 text: {error}") from error


def read_summary_settings(summary_path):
    """Read a summary's settings block and check it; return (the loaded block, the checked settings).

    The text is read, the block is loaded, its keys are checked against
    BLOCK_KEYS and its settings are checked with the placement settings check,
    in that order. Every refusal is a ValueError.
    """
    block = outputs.load_block(read_saved_text(summary_path, "summary"))
    outputs.check_keys(block, BLOCK_KEYS, "the summary's settings")
    settings = placement.check_settings(
        block["slate"], block["mode"], block["max_tries"], block["buffer_um"], block["pixel_um"], block["order"], block["seed"]
    )
    return block, settings


def whole_number(text, name, minimum, where):
    """Return text as an integer when it is a whole number of at least minimum; else raise ValueError naming where."""
    if WHOLE_NUMBER.fullmatch(text) is None or int(text) < minimum:
        raise ValueError(f"{where}: {name} must be a whole number of {minimum} or more, got {text!r}")
    return int(text)


def finite_number(text, name, where):
    """Return text as a float when it is a finite number; else raise ValueError naming where."""
    try:
        number = float(text)
    except ValueError:
        raise ValueError(f"{where}: {name} must be a finite number, got {text!r}") from None
    if not math.isfinite(number):
        raise ValueError(f"{where}: {name} must be a finite number, got {text!r}")
    return number


def parse_placement_row(cells, position, seen, where):
    """Return the PlacementRow for one placements-file line (its 1-based position given); raise ValueError naming where."""
    here = f"{where} line {position + 1}"
    if len(cells) != len(PLACEMENT_COLUMNS):
        raise ValueError(f"{here}: expected {len(PLACEMENT_COLUMNS)} cells, got {len(cells)}")
    values = dict(zip(PLACEMENT_COLUMNS, cells))
    order = whole_number(values["placement_order"], "placement_order", 1, here)
    if order != position:
        raise ValueError(f"{here}: placement_order must be {position}, got {values['placement_order']!r}")
    source = whole_number(values["source_index"], "source_index", 1, here)
    if source in seen:
        raise ValueError(f"{here}: source_index {source} is repeated")
    seen.add(source)
    diameter = finite_number(values["equivalent_diameter_um"], "equivalent_diameter_um", here)
    if diameter <= 0:
        raise ValueError(f"{here}: equivalent_diameter_um must be above 0, got {values['equivalent_diameter_um']!r}")
    if values["placed"] not in ("true", "false"):
        raise ValueError(f"{here}: placed must be true or false, got {values['placed']!r}")
    placed = values["placed"] == "true"
    x = None
    y = None
    if placed:
        if values["x_um"] == "" or values["y_um"] == "":
            raise ValueError(f"{here}: x_um and y_um must be filled for a placed particle")
        x = finite_number(values["x_um"], "x_um", here)
        y = finite_number(values["y_um"], "y_um", here)
    elif values["x_um"] != "" or values["y_um"] != "":
        raise ValueError(f"{here}: x_um and y_um must be empty for a particle that was not placed")
    return placement.PlacementRow(
        placement_order=order,
        source_index=source,
        object_id=values["object_id"],
        feature_label=values["feature_label"],
        equivalent_diameter_um=diameter,
        placed=placed,
        x_um=x,
        y_um=y,
        tries=whole_number(values["tries"], "tries", 0, here),
        edge_rejections=whole_number(values["edge_rejections"], "edge_rejections", 0, here),
        overlap_rejections=whole_number(values["overlap_rejections"], "overlap_rejections", 0, here),
    )


def parse_placements(text, where):
    """Return the PlacementRow tuple of a placements file's text; raise ValueError naming where for any refusal."""
    try:
        lines = list(csv.reader(io.StringIO(text, newline="")))
    except csv.Error as error:
        raise ValueError(f"{where}: the file is not readable as CSV: {error}") from error
    if len(lines) == 0 or tuple(lines[0]) != PLACEMENT_COLUMNS:
        found = lines[0] if len(lines) > 0 else []
        raise ValueError(f"{where}: the header must be {','.join(PLACEMENT_COLUMNS)}, got {','.join(found)!r}")
    rows = []
    seen = set()
    for position, cells in enumerate(lines[1:], start=1):
        rows.append(parse_placement_row(cells, position, seen, where))
    return tuple(rows)


def read_placement_outputs(base_path):
    """Read the two files write_placement_outputs saved at base_path and return the PlacementResult they record.

    The settings come from the summary's block, the rows from the placements
    file, and both coverages are recomputed the way a run computes them.
    Every refusal is a ValueError raised before anything is returned.
    """
    base_path = Path(base_path)
    placements_path = base_path.parent / (base_path.name + PLACEMENTS_ENDING)
    summary_path = base_path.parent / (base_path.name + SUMMARY_ENDING)
    block, settings = read_summary_settings(summary_path)
    rows = parse_placements(read_saved_text(placements_path, "placements file"), f"the placements file {str(placements_path)!r}")
    by_source = sorted(rows, key=operator.attrgetter("source_index"))
    return placement.PlacementResult(
        slate=settings["slate"],
        mode=settings["mode"],
        max_tries=settings["max_tries"],
        buffer_um=settings["buffer_um"],
        pixel_um=settings["pixel_um"],
        order=settings["order"],
        seed=settings["seed"],
        input=block["input"],
        requested_coverage=placement.requested_coverage(by_source, settings["slate"]),
        achieved_coverage=placement.achieved_coverage(rows, settings["slate"]),
        rows=rows,
    )
