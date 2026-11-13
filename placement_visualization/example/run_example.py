# This script is a worked example of the placement drawing functions. It is
# written to be read from top to bottom like a notebook: each numbered step has
# a short explanation of what is done and why, followed by the few lines of
# code that do it. Nearly all of the work happens in main() so that it can be
# followed in order.
#
# The example proceeds as follows:
#   1. Generate the generator example's dataset of particle diameters.
#   2. Place the particles on a slate at random and save the run.
#   3. Draw still images of the slate as it fills, from the saved files.
#   4. Draw a time-lapse video with the dashboard layout, from the saved files.
#   5. Draw the packing map, from the saved files.
#   6. Report the files that were written.
#
# Run it from the repository root with:
#   python3 -m placement_visualization.example.run_example
# It needs matplotlib and ffmpeg (see the guide beside this folder), no
# internet connection, no display and no interaction. The video is drawn at
# a fixed size of 1920 x 1080 pixels; the pictures use a small size so that
# the example runs quickly.

from pathlib import Path

from synthetic_data_generation import generate_dataset, read_entries

from placement_simulation import place_particles, write_placement_outputs
from placement_visualization import write_packing_map, write_still_images, write_video

# Locations of the input file and the default output folder. The input file is
# the generator example's own file. Both are resolved relative to this script,
# so the example runs identically from any working directory.
EXAMPLE_FOLDER = Path(__file__).resolve().parent
INPUT_PATH = EXAMPLE_FOLDER.parent.parent / "synthetic_data_generation" / "example" / "example_input.jsonl"
DEFAULT_OUTPUT_FOLDER = EXAMPLE_FOLDER / "output"

# Every file of the run starts with this name, so the two saved files are
# example_placements.csv and example_summary.txt.
BASE_NAME = "example"

# The seed and the size limits (in micrometres) of the measurement window are
# the same as in the generator's own example, so the dataset is the same: 400
# particles. The seed fixes every random draw.
SEED = 20261029
SIEVE_LOWER_UM = 10.0
SIEVE_UPPER_UM = 120.0

# The slate is a square, 1800 micrometres on each side, so the particles
# request a coverage of about 0.41. Random placement makes up to MAX_TRIES
# tries per particle and keeps BUFFER_UM between the surfaces of two particles.
SLATE = {"shape": "square", "side_um": 1800.0}
MAX_TRIES = 20
BUFFER_UM = 1.0

# A still image is drawn after every 100th particle taken, so the 400 particles
# give four pictures. The video lasts 2 seconds at 10 frames a second, which is
# 20 frames for those four steps. The pictures are 400 pixels on their long side.
STILL_EVERY = 100
VIDEO_SECONDS = 2
VIDEO_FPS = 10
LONG_SIDE_PX = 400


def main(output_folder=DEFAULT_OUTPUT_FOLDER):
    """Run the steps, writing into output_folder, and print the names of the files written last."""
    output_folder = Path(output_folder)

    # ------------------------------------------------------------------
    # Step 1: generate the dataset in memory
    # ------------------------------------------------------------------
    # read_entries reads the three particle classes of the example input file,
    # and generate_dataset draws every diameter, keeping only those inside the
    # size window.
    entries = read_entries(INPUT_PATH)
    dataset = generate_dataset(
        entries,
        threshold_lower=SIEVE_LOWER_UM,
        threshold_upper=SIEVE_UPPER_UM,
        seed=SEED,
    )

    # ------------------------------------------------------------------
    # Step 2: place the particles and save the run
    # ------------------------------------------------------------------
    # Each particle in turn is tried at random locations until one fits. The
    # run is saved before anything is drawn, because the drawing functions read
    # the two saved files and nothing else. With overwrite=True a rerun
    # replaces the earlier files.
    placed = place_particles(
        dataset,
        SLATE,
        mode="random",
        max_tries=MAX_TRIES,
        buffer_um=BUFFER_UM,
        order="shuffled",
        seed=SEED,
    )
    output_folder.mkdir(parents=True, exist_ok=True)
    base_path = output_folder / BASE_NAME
    write_placement_outputs(placed, base_path, overwrite=True)

    # ------------------------------------------------------------------
    # Step 3: still images
    # ------------------------------------------------------------------
    # One picture is written for every STILL_EVERY-th particle taken, and
    # always one for the last. Each shows the particles placed so far, coloured
    # by class. The function returns the paths it wrote.
    still_paths = write_still_images(base_path, STILL_EVERY, long_side_px=LONG_SIDE_PX, overwrite=True)

    # ------------------------------------------------------------------
    # Step 4: time-lapse video
    # ------------------------------------------------------------------
    # The video shows the same steps as the still images. The dashboard layout
    # adds a size histogram, a line graph and a bar chart beside the slate.
    video_path = write_video(
        base_path,
        STILL_EVERY,
        VIDEO_SECONDS,
        fps=VIDEO_FPS,
        layout="dashboard",
        overwrite=True,
    )

    # ------------------------------------------------------------------
    # Step 5: packing map
    # ------------------------------------------------------------------
    # The packing map shows how much of the slate the placed particles cover,
    # smoothed over a neighbourhood. Without sigma_um the function chooses the
    # width of the neighbourhood from the particle sizes.
    map_path = write_packing_map(base_path, long_side_px=LONG_SIDE_PX, overwrite=True)

    # ------------------------------------------------------------------
    # Step 6: report
    # ------------------------------------------------------------------
    print(f"still images: {len(still_paths)}")
    print(f"video: {video_path.name}")
    print(f"packing map: {map_path.name}")
    print(f"files written to {output_folder}")


if __name__ == "__main__":
    main()
