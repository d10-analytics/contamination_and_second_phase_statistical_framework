# This file repeats a saved placement run from its summary. It reads the
# summary's settings block, checks every key and every setting with the same
# functions that a fresh run and the generator's own replay use, and then
# gets the particle sizes again: from the measurements CSV, whose bytes must
# hash to the recorded checksum, or by regenerating the generator result from
# its recorded settings, whose CSV text must hash to the recorded checksum.
# Only then are the particles placed, and the records that were hashed are the
# records that are placed. The library versions recorded in the summary are
# compared with the running ones once, for either kind of input.

import hashlib
from pathlib import Path

from placement_simulation import placement, saving
from synthetic_data_generation import outputs

CSV_INPUT_KEYS = ("kind", "file_name", "measurements_sha256")
GENERATOR_INPUT_KEYS = ("kind", "generator_parameters", "measurements_sha256")


def check_input_block(identity):
    """Check the summary's input object for its kind and return its kind."""
    if not isinstance(identity, dict):
        raise ValueError("the summary's input must be a JSON object")
    kind = identity.get("kind")
    if kind == "csv":
        outputs.check_keys(identity, CSV_INPUT_KEYS, "the summary's input")
        if not isinstance(identity["file_name"], str):
            raise ValueError(f"the summary's input file_name must be text, not {identity['file_name']!r}")
    elif kind == "generator":
        outputs.check_keys(identity, GENERATOR_INPUT_KEYS, "the summary's input")
        if not isinstance(identity["generator_parameters"], dict):
            raise ValueError("the summary's generator_parameters must be a JSON object")
    else:
        raise ValueError(f"the summary's input kind must be 'csv' or 'generator', got {kind!r}")
    if not isinstance(identity["measurements_sha256"], str):
        raise ValueError(f"the summary's measurements_sha256 must be text, not {identity['measurements_sha256']!r}")
    return kind


def records_from_csv(identity, summary_path, measurements_path):
    """Read the recorded CSV once, check its checksum and return the particle records parsed from those bytes."""
    if measurements_path is None:
        csv_path = Path(summary_path).parent / identity["file_name"]
    else:
        csv_path = Path(measurements_path)
    data, rows = outputs.load_measurements(csv_path)
    found = hashlib.sha256(data).hexdigest()
    if found != identity["measurements_sha256"]:
        raise ValueError(
            f"the measurements file {str(csv_path)!r} has SHA-256 {found}, not the recorded {identity['measurements_sha256']}; "
            "it is not the file that was placed"
        )
    return placement.make_particles(rows)


def records_from_generator(identity):
    """Regenerate the recorded generator result, check the checksum of its CSV text and return its particle records."""
    parameters = outputs.check_reproduction_parameters(identity["generator_parameters"])
    result = outputs.generate_from_parameters(parameters)
    found = hashlib.sha256(outputs.serialize_csv(result).encode("utf-8")).hexdigest()
    if found != identity["measurements_sha256"]:
        raise ValueError(
            f"the regenerated measurements have SHA-256 {found}, not the recorded {identity['measurements_sha256']}; "
            "the generator result is not the one that was placed"
        )
    return placement.read_particles(result)[0]


def replay_placement(summary_path, measurements_path=None):
    """Repeat the run a placement summary records and return its PlacementResult; nothing is written to disk.

    A CSV input is read from measurements_path, or else from the recorded file
    name beside the summary. Every refusal is a ValueError raised before any
    particle is placed.
    """
    block = outputs.load_block(Path(summary_path).read_text(encoding="utf-8"))
    outputs.check_keys(block, saving.BLOCK_KEYS, "the summary's settings")
    settings = placement.check_settings(
        block["slate"], block["mode"], block["max_tries"], block["buffer_um"], block["pixel_um"], block["order"], block["seed"]
    )
    if settings["seed"] is None:
        raise ValueError("the summary's seed must be a whole number, not null")
    kind = check_input_block(block["input"])
    versions = outputs.check_library_versions(block["library_versions"])
    if kind == "generator" and measurements_path is not None:
        raise ValueError("measurements_path cannot be used: the summary records a generator result, not a CSV file")
    outputs.warn_about_versions(versions)
    if kind == "csv":
        records = records_from_csv(block["input"], summary_path, measurements_path)
    else:
        records = records_from_generator(block["input"])
    return placement.place_records(records, block["input"], settings)
