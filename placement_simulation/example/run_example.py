# This script is a worked example of the particle placement simulator. It is
# written to be read from top to bottom like a notebook: each numbered step has
# a short explanation of what is done and why, followed by the few lines of
# code that do it. Nearly all of the work happens in main() so that it can be
# followed in order.
#
# The example proceeds as follows:
#   1. Generate the generator example's dataset of particle diameters.
#   2. Save the dataset as a CSV file and read it back, as the placement
#      simulator would receive it.
#   3. Place the particles on a slate at random, save the run and replay it.
#   4. Place the same particles from the slate centre outwards, save the run
#      and replay it.
#   5. Report whether both replays matched their saved placements files.
#
# Run it from the repository root with:
#   python3 -m placement_simulation.example.run_example
# It requires no internet connection, no display and no interaction. It uses
# only the standard library, NumPy and the two packages of this repository, so
# it runs where matplotlib is not installed.

from pathlib import Path

from synthetic_data_generation import generate_dataset, read_entries, read_measurements, write_outputs

from placement_simulation import place_particles, replay, saving, write_placement_outputs

# Locations of the input file and the default output folder. The input file is
# the generator example's own file, so both examples describe the same
# particles. Both are resolved relative to this script, so the example runs
# identically from any working directory.
EXAMPLE_FOLDER = Path(__file__).resolve().parent
INPUT_PATH = EXAMPLE_FOLDER.parent.parent / "synthetic_data_generation" / "example" / "example_input.jsonl"
DEFAULT_OUTPUT_FOLDER = EXAMPLE_FOLDER / "output"

CSV_NAME = "example_particles.csv"
RANDOM_BASE_NAME = "random"
CENTRE_BASE_NAME = "centre"

# The seed and the size limits (in micrometres) of the measurement window are
# the same as in the generator's own example, so the dataset is the same. The
# seed fixes every random draw: the same seed always gives the same particles
# and the same placements.
SEED = 20261029
SIEVE_LOWER_UM = 10.0
SIEVE_UPPER_UM = 120.0

# The slate is a square, 1800 micrometres on each side. The 400 particles of
# the example have a total area of about 1.33 million square micrometres, so
# they request a coverage of about 0.41: the particles would fill about 41% of
# the slate if all fitted. That is crowded enough for random placement to
# reject tries that come too close to a placed particle.
SLATE = {"shape": "square", "side_um": 1800.0}

# Random placement makes up to RANDOM_MAX_TRIES tries per particle. Centre
# placement makes one search per particle, so it uses CENTRE_MAX_TRIES, which
# is 1. BUFFER_UM is the minimum free gap between the surfaces of two
# particles, counted once per pair; it is not applied at the slate edge, which
# a particle may touch. PIXEL_UM is the cell size of the clearance map that
# centre placement searches. The larger cell keeps the example quick.
RANDOM_MAX_TRIES = 20
CENTRE_MAX_TRIES = 1
BUFFER_UM = 1.0
PIXEL_UM = 4.0
ORDER = "shuffled"


def save_and_replay(result, output_folder, base_name):
    """Save a placement run beside base_name, replay it from its summary and say whether the replay matches the saved file."""
    placements_path, summary_path = write_placement_outputs(result, output_folder / base_name, overwrite=True)
    # The replay reads the summary and the CSV it names, checks that the CSV is
    # the one that was placed, and places the particles again. Nothing is
    # written. serialize_placements turns the replayed run into the exact text
    # that would be saved, which is compared with the saved file.
    replayed = replay.replay_placement(summary_path)
    replayed_text = saving.serialize_placements(replayed)
    return replayed_text.encode("utf-8") == placements_path.read_bytes()


def main(output_folder=DEFAULT_OUTPUT_FOLDER):
    """Run the steps, writing into output_folder, and print "identical: True" or "identical: False" last."""
    output_folder = Path(output_folder)

    # ------------------------------------------------------------------
    # Step 1: generate the dataset in memory
    # ------------------------------------------------------------------
    # read_entries reads the three particle classes of the example input file,
    # and generate_dataset draws every diameter, keeping only those inside the
    # size window, and derives the area and volume of each particle.
    entries = read_entries(INPUT_PATH)
    result = generate_dataset(
        entries,
        threshold_lower=SIEVE_LOWER_UM,
        threshold_upper=SIEVE_UPPER_UM,
        seed=SEED,
    )

    # ------------------------------------------------------------------
    # Step 2: save the dataset and read it back
    # ------------------------------------------------------------------
    # write_outputs saves the CSV and its summary. With overwrite=True a rerun
    # replaces the earlier files. read_measurements reads the CSV back as one
    # dictionary per particle, so the count printed below comes from the saved
    # file, which is all that the placement runs below use.
    output_folder.mkdir(parents=True, exist_ok=True)
    csv_path = output_folder / CSV_NAME
    write_outputs(result, csv_path, overwrite=True)
    measurements = read_measurements(csv_path)
    print(f"particles read from {csv_path.name}: {len(measurements)}")

    # ------------------------------------------------------------------
    # Step 3: random placement, saved and replayed
    # ------------------------------------------------------------------
    # Each particle in turn is tried at random locations until one fits inside
    # the slate and keeps at least BUFFER_UM from every placed particle, or
    # until RANDOM_MAX_TRIES tries have failed. The requested coverage is printed first. Every rejected try
    # is counted, so the run shows how crowded the slate was. The particles are
    # given by the CSV's path, so the summary records the file name and its
    # checksum. The saved run is then replayed from its summary alone.
    random_result = place_particles(
        csv_path,
        SLATE,
        mode="random",
        max_tries=RANDOM_MAX_TRIES,
        buffer_um=BUFFER_UM,
        pixel_um=PIXEL_UM,
        order=ORDER,
        seed=SEED,
    )
    rejected = 0
    for row in random_result.rows:
        rejected += row.edge_rejections + row.overlap_rejections
    print(f"random placement: rejected tries {rejected}")
    random_identical = save_and_replay(random_result, output_folder, RANDOM_BASE_NAME)

    # ------------------------------------------------------------------
    # Step 4: centre placement, saved and replayed
    # ------------------------------------------------------------------
    # Each particle is put at the free spot closest to the slate centre, so the
    # cluster grows outwards from the middle. One search is made per particle,
    # and nothing is rejected until the cluster reaches the slate edge. The run
    # is saved and replayed in the same way.
    centre_result = place_particles(
        csv_path,
        SLATE,
        mode="centre",
        max_tries=CENTRE_MAX_TRIES,
        buffer_um=BUFFER_UM,
        pixel_um=PIXEL_UM,
        order=ORDER,
        seed=SEED,
    )
    centre_identical = save_and_replay(centre_result, output_folder, CENTRE_BASE_NAME)

    # ------------------------------------------------------------------
    # Step 5: report
    # ------------------------------------------------------------------
    # identical is True only when both replays matched their saved placements
    # files byte for byte.
    identical = random_identical and centre_identical
    print(f"files written to {output_folder}")
    print(f"identical: {identical}")


if __name__ == "__main__":
    main()
