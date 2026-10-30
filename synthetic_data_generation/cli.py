# This file is the command line of the generator. It reads the arguments, checks
# everything that can be checked, picks the output file, and only then draws the
# particles and writes the two files. The order matters: a typo in the input or
# a settings mistake is reported before a save dialog ever opens, and a
# destination that cannot be written is reported before any drawing starts.
# Every generation setting defaults to None here so that the real defaults live
# in one place only (dataset.py); a None setting takes the constant from there.
# The tkinter toolkit is imported only inside choose_output_path, so a run that
# is given --output never loads it.

import argparse
import dataclasses
import sys
from pathlib import Path

from synthetic_data_generation import dataset
from synthetic_data_generation import entries as entries_module
from synthetic_data_generation import outputs
from synthetic_data_generation import sampling

PROGRAM_NAME = "python -m synthetic_data_generation"

# Every flag that describes a generation run. A summary already holds all of
# them, so none may be combined with --from-summary.
GENERATION_FLAGS = (
    ("input_json", "--input-json"),
    ("input_file", "--input-file"),
    ("seed", "--seed"),
    ("max_attempts", "--max-attempts"),
    ("threshold_lower", "--threshold-lower"),
    ("threshold_upper", "--threshold-upper"),
    ("shuffle", "--shuffle"),
)


def build_parser():
    """Return the argument parser; every generation setting stays None when omitted."""
    parser = argparse.ArgumentParser(
        prog=PROGRAM_NAME,
        description="Generate synthetic particle measurements, or repeat an earlier run from its summary.",
    )
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--input-json", default=None, help="entries as JSON or JSONL text")
    sources.add_argument("--input-file", default=None, help="path of a JSON or JSONL file of entries")
    sources.add_argument("--from-summary", default=None, help="path of a summary file to repeat exactly")
    parser.add_argument("--seed", type=int, default=None, help="whole number that makes the run repeatable")
    parser.add_argument("--max-attempts", type=int, default=None, help="redraw limit for each particle")
    parser.add_argument("--threshold-lower", type=float, default=None, help="smallest diameter kept (um)")
    parser.add_argument("--threshold-upper", type=float, default=None, help="largest diameter kept (um)")
    parser.add_argument("--shuffle", action="store_true", default=None, help="shuffle the rows")
    parser.add_argument("--output", default=None, help="path of the measurements CSV; asked for when omitted")
    parser.add_argument("--overwrite", action="store_true", help="replace existing output files")
    return parser


def reject_generation_flags_with_summary(parser, args):
    """Stop with an error when --from-summary is combined with any generation flag."""
    if args.from_summary is None:
        return
    for attribute, flag in GENERATION_FLAGS:
        if getattr(args, attribute) is not None:
            parser.error(f"{flag} cannot be combined with --from-summary; the summary already holds every setting")


def choose_output_path():
    """Ask the user where to save the CSV in a save dialog and return the chosen path."""
    # Imported here, and only here, so a run given --output never loads tkinter.
    import tkinter
    from tkinter import filedialog

    try:
        root = tkinter.Tk()
    except tkinter.TclError as error:
        raise ValueError(f"no save dialog is available ({error}); pass --output instead")
    try:
        root.withdraw()
        # The tool, not the dialog, decides whether an existing file may be replaced.
        chosen = filedialog.asksaveasfilename(
            title="Save the measurements CSV",
            confirmoverwrite=False,
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
        )
    except tkinter.TclError as error:
        raise ValueError(f"the save dialog failed ({error}); pass --output instead")
    finally:
        root.destroy()
    if not chosen:
        raise ValueError("no output file was chosen, so nothing was written")
    return str(chosen)


def check_settings(seed, max_attempts, threshold_lower, threshold_upper, shuffle):
    """Validate the settings with the shared validators and return them with defaults filled in."""
    if max_attempts is None:
        max_attempts = dataset.DEFAULT_MAX_ATTEMPTS
    if shuffle is None:
        shuffle = dataset.DEFAULT_SHUFFLE
    if threshold_lower is None:
        threshold_lower = dataset.DEFAULT_THRESHOLD_LOWER
    if threshold_upper is None:
        threshold_upper = dataset.DEFAULT_THRESHOLD_UPPER
    sampling.validate_thresholds(threshold_lower, threshold_upper)
    if seed is not None:
        dataset.validate_seed(seed)
    dataset.validate_max_attempts(max_attempts)
    dataset.validate_shuffle(shuffle)
    return seed, max_attempts, threshold_lower, threshold_upper, shuffle


def check_input(args):
    """Read and validate the entries and settings for a new run; return the generate_dataset arguments."""
    if args.input_json is not None:
        entries_module.parse_entries(args.input_json)
        entries = args.input_json
    else:
        entries = entries_module.read_entries(args.input_file)
    seed, max_attempts, threshold_lower, threshold_upper, shuffle = check_settings(
        args.seed, args.max_attempts, args.threshold_lower, args.threshold_upper, args.shuffle
    )
    return {
        "entries": entries,
        "seed": seed,
        "max_attempts": max_attempts,
        "threshold_lower": threshold_lower,
        "threshold_upper": threshold_upper,
        "shuffle": shuffle,
    }


def make_result(args, settings):
    """Generate a new run's dataset and record an input file's absolute path as its source."""
    result = dataset.generate_dataset(
        settings["entries"],
        threshold_lower=settings["threshold_lower"],
        threshold_upper=settings["threshold_upper"],
        seed=settings["seed"],
        max_attempts=settings["max_attempts"],
        shuffle=settings["shuffle"],
    )
    if args.input_file is not None:
        # An absolute path can never equal the words "text" or "list" that mark API sources.
        result = dataclasses.replace(result, input_source=str(Path(args.input_file).resolve()))
    if args.seed is None:
        print(f"seed used: {result.seed}", file=sys.stderr)
    return result


def run(args):
    """Run in the order: validate input, choose the path, check destinations, generate, write."""
    if args.from_summary is not None:
        outputs.read_summary(args.from_summary)
        settings = None
    else:
        settings = check_input(args)
    csv_path = args.output
    if csv_path is None:
        csv_path = choose_output_path()
    outputs.check_destinations(csv_path, args.overwrite)
    if settings is None:
        result = outputs.replay_summary(args.from_summary)
    else:
        result = make_result(args, settings)
    outputs.write_outputs(result, csv_path, args.overwrite)


def main(argv=None):
    """Run the command line on the argument list and return the exit code (0 for success)."""
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        reject_generation_flags_with_summary(parser, args)
    except SystemExit as stop:
        return stop.code if isinstance(stop.code, int) else 2
    try:
        run(args)
    except (ValueError, OSError, sampling.SamplingError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0
