# This script is a worked example of the placement drawing functions. It is
# written to be read from top to bottom like a notebook: each numbered step has
# a short explanation of what is done and why, followed by the few lines of
# code that do it. Particles are placed once in each placement mode, and every
# picture is drawn for both runs, so the two modes can be compared file by
# file. The centre run uses twice as many particles, so its cluster grows
# until it reaches the slate edge.
#
# The example proceeds as follows:
#   1. Generate the generator example's dataset of particle diameters (400
#      particles), and a second dataset with every class twice as large (800).
#   2. Place the 400 particles at random and the 800 from the slate centre, and
#      save each run in its own folder: random placement in output_random,
#      centre placement in output_center.
#   3. For each run, draw still images of the slate as it fills.
#   4. For each run, draw a time-lapse video with the dashboard layout.
#   5. For each run, draw the packing map and the series of packing maps at
#      several blur widths.
#   6. Report the files that were written.
#
# Run it from the repository root with:
#   python3 -m placement_visualization.example.run_example
# It needs matplotlib and ffmpeg (see the guide beside this folder), no
# internet connection, no display and no interaction. The videos are drawn at
# a fixed size of 1920 x 1080 pixels; the pictures use a small size so that
# the example runs quickly.

from pathlib import Path

from synthetic_data_generation import generate_dataset, read_entries

from placement_simulation import place_particles, write_placement_outputs
from placement_visualization import write_packing_map, write_packing_map_series, write_still_images, write_video

# Locations of the input file and the two output folders. The input file is
# the generator example's own file. All are resolved relative to this script,
# so the example runs identically from any working directory.
EXAMPLE_FOLDER = Path(__file__).resolve().parent
INPUT_PATH = EXAMPLE_FOLDER.parent.parent / "synthetic_data_generation" / "example" / "example_input.jsonl"
RANDOM_FOLDER_NAME = "output_random"
CENTRE_FOLDER_NAME = "output_center"

# Every file of a run starts with this name, so the two saved files of each run
# are example_placements.csv and example_summary.txt, in that run's folder.
BASE_NAME = "example"

# The seed and the size limits (in micrometres) of the measurement window are
# the same as in the generator's own example, so the random run's dataset is
# that example's 400 particles. The centre run multiplies every class's N by
# CENTRE_COUNT_FACTOR, which keeps the mix of classes and gives 800 particles,
# drawn afresh with the same seed. The seed fixes every random draw, including
# the shuffled order in which each run takes its particles.
SEED = 20261029
SIEVE_LOWER_UM = 10.0
SIEVE_UPPER_UM = 120.0
CENTRE_COUNT_FACTOR = 2

# The slate is a square, 1800 micrometres on each side, so the 400 particles
# request a coverage of about 0.41 and the 800 about 0.82. That is more than a
# cluster of circles can fill, so the centre cluster reaches the slate edge and
# the particles that no longer fit are left unplaced. Both runs keep BUFFER_UM between the
# surfaces of two particles and take the particles in the same shuffled order.
# Random placement makes up to RANDOM_MAX_TRIES random tries per particle.
# Centre placement puts each particle at the free spot closest to the slate
# centre, found by one search per particle, so it uses CENTRE_MAX_TRIES, which
# must be 1. PIXEL_UM is the cell size of the grid that centre placement
# searches; the same cell as the placement example keeps it quick.
SLATE = {"shape": "square", "side_um": 1800.0}
BUFFER_UM = 1.0
ORDER = "shuffled"
RANDOM_MAX_TRIES = 20
CENTRE_MAX_TRIES = 1
PIXEL_UM = 4.0

# A still image is drawn after every 100th particle attempted, so the 400
# particles give four pictures and the 800 give eight. The video shows a step
# after every 5th particle, which is 80 steps for 400 particles and 160 for
# 800, and lasts 20 seconds at 30 frames a second: 600 frames, so each step is
# held for a quarter of a second (400 particles) or an eighth (800). A still image's slate
# is 400 pixels on its long side, with the legend added below it, and the
# packing maps have 400 cells along the slate's long side.
STILL_EVERY = 100
VIDEO_EVERY = 5
VIDEO_SECONDS = 20
VIDEO_FPS = 30
LONG_SIDE_PX = 400

# The packing map series uses these blur widths in micrometres, smallest
# first. None stands for the default, the median particle diameter (about 60
# micrometres for both datasets), so the default map has its place in the
# series. The smallest widths show single particles; the largest blur almost
# everything into the slate's overall coverage.
SERIES_SIGMAS_UM = [5.0, 15.0, 30.0, None, 120.0, 240.0]


def main(example_folder=EXAMPLE_FOLDER):
    """Run the steps, writing each run into its folder inside example_folder, and print the names of the files written."""
    example_folder = Path(example_folder)

    # ------------------------------------------------------------------
    # Step 1: generate the dataset in memory
    # ------------------------------------------------------------------
    # read_entries reads the three particle classes of the example input file,
    # and generate_dataset draws every diameter, keeping only those inside the
    # size window. Each entry is a dictionary, so the centre run's entries are
    # copies with N multiplied by CENTRE_COUNT_FACTOR.
    entries = read_entries(INPUT_PATH)
    dataset = generate_dataset(
        entries,
        threshold_lower=SIEVE_LOWER_UM,
        threshold_upper=SIEVE_UPPER_UM,
        seed=SEED,
    )
    centre_entries = [dict(entry, N=CENTRE_COUNT_FACTOR * entry["N"]) for entry in entries]
    centre_dataset = generate_dataset(
        centre_entries,
        threshold_lower=SIEVE_LOWER_UM,
        threshold_upper=SIEVE_UPPER_UM,
        seed=SEED,
    )

    # ------------------------------------------------------------------
    # Step 2: place the particles in each mode and save each run
    # ------------------------------------------------------------------
    # Random placement tries random locations for each particle in turn until
    # one fits, and leaves the particle unplaced after RANDOM_MAX_TRIES
    # rejected tries. Centre placement grows one cluster outwards from the
    # slate centre, and leaves a particle unplaced once no free spot on the
    # slate is large enough. Each run is saved before anything is drawn, because the
    # drawing functions read the two saved files and nothing else. With
    # overwrite=True a rerun replaces the earlier files.
    random_run = place_particles(
        dataset,
        SLATE,
        mode="random",
        max_tries=RANDOM_MAX_TRIES,
        buffer_um=BUFFER_UM,
        order=ORDER,
        seed=SEED,
    )
    centre_run = place_particles(
        centre_dataset,
        SLATE,
        mode="centre",
        max_tries=CENTRE_MAX_TRIES,
        buffer_um=BUFFER_UM,
        pixel_um=PIXEL_UM,
        order=ORDER,
        seed=SEED,
    )
    base_paths = []
    for run, folder_name in ((random_run, RANDOM_FOLDER_NAME), (centre_run, CENTRE_FOLDER_NAME)):
        output_folder = example_folder / folder_name
        output_folder.mkdir(parents=True, exist_ok=True)
        base_path = output_folder / BASE_NAME
        write_placement_outputs(run, base_path, overwrite=True)
        base_paths.append(base_path)

    # Steps 3 to 5 draw every picture of one saved run. They run once for each
    # folder, so both runs get the same set of files.
    for base_path in base_paths:
        # --------------------------------------------------------------
        # Step 3: still images
        # --------------------------------------------------------------
        # One picture is written for every STILL_EVERY-th particle attempted,
        # and always one for the last. Each shows the particles placed so far,
        # coloured by class. The function returns the paths it wrote.
        still_paths = write_still_images(base_path, STILL_EVERY, long_side_px=LONG_SIDE_PX, overwrite=True)

        # --------------------------------------------------------------
        # Step 4: time-lapse video
        # --------------------------------------------------------------
        # The video shows many more steps than the still images, so the slate
        # visibly fills. The dashboard layout adds two size histograms, a line
        # graph and a bar chart beside the slate.
        video_path = write_video(
            base_path,
            VIDEO_EVERY,
            VIDEO_SECONDS,
            fps=VIDEO_FPS,
            layout="dashboard",
            overwrite=True,
        )

        # --------------------------------------------------------------
        # Step 5: packing map and packing map series
        # --------------------------------------------------------------
        # The packing map shows how much of the slate the placed particles
        # cover, smoothed over a neighbourhood. Without sigma_um the width of
        # the neighbourhood is the median particle diameter. The series draws
        # the same map at each width in SERIES_SIGMAS_UM and a grid of them
        # all, to show how the width changes what the map shows.
        map_path = write_packing_map(base_path, long_side_px=LONG_SIDE_PX, overwrite=True)
        series_paths, grid_path = write_packing_map_series(base_path, SERIES_SIGMAS_UM, long_side_px=LONG_SIDE_PX, overwrite=True)

        # --------------------------------------------------------------
        # Step 6: report
        # --------------------------------------------------------------
        print(f"files written to {base_path.parent}")
        print(f"  still images: {len(still_paths)}")
        print(f"  video: {video_path.name}")
        print(f"  packing map: {map_path.name}")
        print(f"  packing map series: {len(series_paths)} maps and {grid_path.name}")


if __name__ == "__main__":
    main()
