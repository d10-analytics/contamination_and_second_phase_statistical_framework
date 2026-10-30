# This script is a worked example of the whole generator in five numbered steps:
# generate a small dataset from example_input.jsonl, save it, read the CSV back,
# draw a plot of it, and finally check that the summary file can recreate the
# CSV exactly. Run it from the repository root with:
#   python3 -m synthetic_data_generation.example.run_example
# It needs no internet connection, no screen and no typing while it runs.

import csv
from pathlib import Path

# "Agg" is a drawing mode that writes picture files without opening a window.
# It must be chosen before pyplot is imported, or matplotlib may look for a screen.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as pyplot

from synthetic_data_generation import generate_dataset, read_entries, write_outputs
from synthetic_data_generation import outputs

EXAMPLE_FOLDER = Path(__file__).resolve().parent
INPUT_PATH = EXAMPLE_FOLDER / "example_input.jsonl"
DEFAULT_OUTPUT_FOLDER = EXAMPLE_FOLDER / "output"

CSV_NAME = "example_particles.csv"

# The seed is a whole number that fixes the random draws, so every run gives the
# same particles. The sieve limits (in micrometres) keep only diameters that
# would pass through a sieve: nothing below the lower limit, nothing above the upper.
SEED = 20261029
SIEVE_LOWER_UM = 10.0
SIEVE_UPPER_UM = 120.0


def generate_example_dataset():
    """Step 1: read the three entries from the input file and generate the dataset in memory."""
    entries = read_entries(INPUT_PATH)
    return generate_dataset(
        entries,
        threshold_lower=SIEVE_LOWER_UM,
        threshold_upper=SIEVE_UPPER_UM,
        seed=SEED,
    )


def read_csv_back(csv_path):
    """Step 3: read the CSV file from disk and return a dictionary (a lookup table that pairs each class name with its list of diameters)."""
    diameters_by_class = {}
    with open(csv_path, "r", encoding="utf-8", newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            class_name = row["feature_label"]
            if class_name not in diameters_by_class:
                diameters_by_class[class_name] = []
            diameters_by_class[class_name].append(float(row["equivalent_diameter_um"]))
    return diameters_by_class


def plot_csv(csv_path):
    """Step 4: plot the CSV at csv_path, save <name>_plot.png beside it, and return the count of rows per class."""
    diameters_by_class = read_csv_back(csv_path)
    csv_path = Path(csv_path)
    plot_path = csv_path.with_name(csv_path.stem + "_plot.png")
    counts = {}
    figure, (histogram_axes, bar_axes) = pyplot.subplots(1, 2, figsize=(11, 4))
    for class_name, diameters in diameters_by_class.items():
        counts[class_name] = len(diameters)
        histogram_axes.hist(diameters, bins=20, alpha=0.6, label=class_name)
    histogram_axes.set_xlabel("equivalent diameter (um)")
    histogram_axes.set_ylabel("particles")
    histogram_axes.set_title("Diameter histogram per class")
    histogram_axes.legend()
    bar_axes.bar(list(counts.keys()), list(counts.values()))
    bar_axes.set_xlabel("class")
    bar_axes.set_ylabel("kept particles")
    bar_axes.set_title("Kept count per class")
    figure.tight_layout()
    figure.savefig(plot_path)
    pyplot.close(figure)
    return counts


def replay_matches(csv_path, summary_path):
    """Step 5: recreate the CSV text from the summary with the shared replay and compare it with the file on disk."""
    replayed = outputs.replay_summary(summary_path)
    replayed_bytes = outputs.serialize_csv(replayed).encode("utf-8")
    return replayed_bytes == Path(csv_path).read_bytes()


def main(output_folder=DEFAULT_OUTPUT_FOLDER):
    """Run the five steps, writing into output_folder, and print "identical: True" or "identical: False" last."""
    output_folder = Path(output_folder)
    result = generate_example_dataset()
    # Step 2: make the folder if it is missing and save the two files; overwrite
    # is on so running the example again simply replaces the earlier files.
    output_folder.mkdir(parents=True, exist_ok=True)
    csv_path = output_folder / CSV_NAME
    write_outputs(result, csv_path, overwrite=True)
    summary_path = outputs.summary_path_for(csv_path)
    counts = plot_csv(csv_path)
    print(f"kept particles per class: {counts}")
    print(f"files written to {output_folder}")
    print(f"identical: {replay_matches(csv_path, summary_path)}")


if __name__ == "__main__":
    main()
