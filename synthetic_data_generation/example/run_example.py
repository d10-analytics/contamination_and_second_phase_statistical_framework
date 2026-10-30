# This script is a worked example of the whole generator. It is written to be
# read from top to bottom like a notebook: each numbered step has a short
# explanation of what is done and why, followed by the few lines of code that
# do it. Nearly all of the work happens in main() so that it can be followed in
# order.
#
# The example proceeds as follows:
#   1. Read a small input file describing three particle classes, and generate
#      a synthetic dataset of particle diameters from it.
#   2. Save the dataset as a CSV file together with a text summary of the
#      settings used.
#   3. Read the CSV back from disk, as an analyst receiving the file would.
#   4. Plot the diameters of each class.
#   5. Verify that the summary alone is sufficient to regenerate the CSV
#      byte for byte.
#
# Run it from the repository root with:
#   python3 -m synthetic_data_generation.example.run_example
# It requires no internet connection, no display and no interaction.

import csv
from pathlib import Path

# "Agg" is a non-interactive matplotlib backend that renders figures straight to
# image files without opening a window. It must be selected before pyplot is
# imported; otherwise matplotlib may try to connect to a display.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as pyplot

from synthetic_data_generation import generate_dataset, read_entries, write_outputs
from synthetic_data_generation import outputs

# Locations of the input file and the default output folder. Both are resolved
# relative to this script, so the example runs identically from any working
# directory.
EXAMPLE_FOLDER = Path(__file__).resolve().parent
INPUT_PATH = EXAMPLE_FOLDER / "example_input.jsonl"
DEFAULT_OUTPUT_FOLDER = EXAMPLE_FOLDER / "output"

CSV_NAME = "example_particles.csv"

# The seed initialises the pseudo-random number generator. Fixing it makes the
# simulation deterministic: the same seed, entries and limits always produce
# exactly the same dataset. Changing it yields a different, equally valid
# realisation of the same distributions.
SEED = 20261029

# Size limits (in micrometres) of the measurement window. Real instruments and
# analysis pipelines only report particles within a finite size range, so the
# generator imposes the same restriction on the simulated data. Each diameter is
# drawn from the class's distribution and rejected if it falls outside
# [SIEVE_LOWER_UM, SIEVE_UPPER_UM] (both limits inclusive); drawing continues
# until the class holds exactly N diameters. The window therefore never changes
# a class's count: it changes the distribution, which becomes the class's
# distribution truncated to this window rather than the original untruncated one. The lower limit is named "threshold_lower"
# and the upper "threshold_upper" in the generator's own interface.
SIEVE_LOWER_UM = 10.0
SIEVE_UPPER_UM = 120.0


def main(output_folder=DEFAULT_OUTPUT_FOLDER):
    """Run the five steps, writing into output_folder, and print "identical: True" or "identical: False" last."""
    output_folder = Path(output_folder)

    # ------------------------------------------------------------------
    # Step 1: describe the particles and generate the dataset in memory
    # ------------------------------------------------------------------
    # Each line of example_input.jsonl is one "entry": the name of a particle
    # class, the family of distribution its equivalent diameters follow, the
    # count N (the exact number of particles generated for the class and written
    # to the CSV) and the distribution's parameters. This example defines three
    # classes:
    #   corrosion pits - generalised extreme value (GEV). Pit sizes are
    #                    right-skewed, with most pits moderate in size and a
    #                    few much larger, the kind of tail behaviour that
    #                    extreme-value distributions describe.
    #   oxide          - Gaussian, i.e. symmetric about a single typical size.
    #   inclusion      - GEV, but small and tightly grouped.
    # read_entries validates the file (a missing key or non-numeric parameter
    # raises an error that names the offending line) and returns a list of
    # plain dictionaries.
    entries = read_entries(INPUT_PATH)

    # generate_dataset draws the diameters of every class, applies the size
    # window defined above, and derives the area and volume columns from each
    # equivalent diameter. Nothing is written to disk at this point. The result
    # is an in-memory object that also records every setting used, so that the
    # run can be repeated exactly.
    result = generate_dataset(
        entries,
        threshold_lower=SIEVE_LOWER_UM,
        threshold_upper=SIEVE_UPPER_UM,
        seed=SEED,
    )

    # ------------------------------------------------------------------
    # Step 2: save the dataset
    # ------------------------------------------------------------------
    # write_outputs saves two files side by side: the CSV of particle
    # measurements and a text summary of every setting used. The summary is what
    # makes the CSV reproducible, and step 5 depends on it. The output folder is
    # created first if it does not exist. With overwrite=True, rerunning the
    # example replaces the earlier files instead of raising an error.
    output_folder.mkdir(parents=True, exist_ok=True)
    csv_path = output_folder / CSV_NAME
    write_outputs(result, csv_path, overwrite=True)
    summary_path = outputs.summary_path_for(csv_path)

    # ------------------------------------------------------------------
    # Step 3: read the CSV back from disk
    # ------------------------------------------------------------------
    # From this point only the saved file is used, not the in-memory result,
    # because the CSV is all that a collaborator would receive. The diameters
    # are collected into a dictionary mapping each class name to the list of its
    # equivalent diameters (in micrometres).
    diameters_by_class = {}
    with open(csv_path, "r", encoding="utf-8", newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            class_name = row["feature_label"]
            if class_name not in diameters_by_class:
                diameters_by_class[class_name] = []
            diameters_by_class[class_name].append(float(row["equivalent_diameter_um"]))

    # The count of each class, i.e. the number of its rows in the CSV. Because
    # sampling continues until each class has N diameters, these equal the N
    # values in the input file whatever the size window is; the window truncates
    # the distributions but cannot reduce the counts.
    counts = {class_name: len(diameters) for class_name, diameters in diameters_by_class.items()}

    # ------------------------------------------------------------------
    # Step 4: plot the CSV
    # ------------------------------------------------------------------
    # The figure has two panels. The left panel overlays a histogram of the
    # diameters of each class, showing the shape of each size distribution and
    # where the classes overlap. The right panel is a bar chart of the
    # count per class, a quick consistency check against the input file. The
    # figure is saved next to the CSV.
    plot_path = csv_path.with_name(csv_path.stem + "_plot.png")
    figure, (histogram_axes, bar_axes) = pyplot.subplots(1, 2, figsize=(11, 4))
    for class_name, diameters in diameters_by_class.items():
        histogram_axes.hist(diameters, bins=20, alpha=0.6, label=class_name)
    histogram_axes.set_xlabel("equivalent diameter (um)")
    histogram_axes.set_ylabel("particles")
    histogram_axes.set_title("Diameter histogram per class")
    histogram_axes.legend()
    bar_axes.bar(list(counts.keys()), list(counts.values()))
    bar_axes.set_xlabel("class")
    bar_axes.set_ylabel("count (particles)")
    bar_axes.set_title("Count per class")
    figure.tight_layout()
    figure.savefig(plot_path)
    # Close the figure to release its memory; otherwise repeated runs within a
    # single Python session would accumulate open figures.
    pyplot.close(figure)

    # ------------------------------------------------------------------
    # Step 5: recreate the CSV from the summary and compare
    # ------------------------------------------------------------------
    # The summary records the entries, the size limits and the seed.
    # replay_summary reads it and reruns the generator from those settings
    # alone, returning a new in-memory dataset. serialize_csv converts that
    # dataset into the exact CSV text the generator would write. If this text
    # matches the file on disk byte for byte, the summary is a complete record
    # of how the data were produced: anyone holding it can regenerate the
    # dataset.
    replayed = outputs.replay_summary(summary_path)
    replayed_bytes = outputs.serialize_csv(replayed).encode("utf-8")
    identical = replayed_bytes == csv_path.read_bytes()

    print(f"counts per class: {counts}")
    print(f"files written to {output_folder}")
    print(f"identical: {identical}")


if __name__ == "__main__":
    main()
