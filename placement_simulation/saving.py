# This file saves a finished placement run. A run produces two files side by
# side: a placements CSV (one row per particle, in the order taken) and a text
# summary with a readable part and a machine-readable settings block (a small
# JSON object, which is plain text describing the settings). The destination
# checks and the all-or-nothing writing are the generator's own functions in
# its outputs module, so both packages refuse, write and clean up identically.
# Coordinates are in micrometres, the origin at the slate centre, x to the
# right and y up.

import csv
import io
import json
from pathlib import Path

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
