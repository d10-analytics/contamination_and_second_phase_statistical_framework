# Particle placement simulator

This tool takes a list of particle sizes, such as the CSV the synthetic data generator writes, and places the particles as circles on a flat surface called the slate, so that no two overlap and none leaves the slate. It writes two files: a CSV with the position of every particle, and a text summary that records every setting, so the same placement can be repeated exactly. The steps below assume you have never used Python. Commands are typed into a terminal; they are written for Linux and macOS.

## 1. Install Python and the packages

Follow steps 1 to 3 of the [synthetic data guide](../synthetic_data_generation/README.md): check the Python version, create and activate a virtual environment, and install the packages from `synthetic_data_generation/requirements.txt`. The simulator needs no further packages, and its example needs no internet connection afterwards. (The simulator itself does not draw anything, so it does not use matplotlib, although the generator's requirements install it.)

## 2. Run the example

From the repository root folder, run:

```
python3 -m placement_simulation.example.run_example
```

The script `placement_simulation/example/run_example.py` is written to be read top to bottom like a notebook, with commentary before each of its five numbered steps. It generates the generator example's dataset (400 particles in three classes, from `synthetic_data_generation/example/example_input.jsonl`), saves it as a CSV and reads it back; places the particles at random on a square slate and saves the run; places them again from the slate centre outwards and saves that run; and replays each saved run from its summary alone. The last line it prints is `identical: True`, which means both replays matched their saved placements files byte for byte. It takes about a second.

It writes six files into the folder `placement_simulation/example/output/`, which is created if missing and is not tracked by git:

- `placement_simulation/example/output/example_particles.csv`: the particle sizes, in the generator's format, that both runs place.
- `placement_simulation/example/output/example_particles_summary.txt`: the generator's summary of that CSV.
- `placement_simulation/example/output/random_placements.csv`: the placements of the random run.
- `placement_simulation/example/output/random_summary.txt`: the settings and results of the random run.
- `placement_simulation/example/output/centre_placements.csv`: the placements of the centre run.
- `placement_simulation/example/output/centre_summary.txt`: the settings and results of the centre run.

Running the example again replaces these files. The run prints the requested coverage (explained below) once for each placement and replay.

## 3. The placements file

Each placements file is a CSV with one row per particle, in the order the particles were taken. Distances are in micrometres (um); the origin is the slate centre, x points to the right and y points up. The columns are:

- `placement_order`: 1 for the first particle taken, 2 for the next, and so on.
- `source_index`: the particle's row number in the input CSV, counting from 1.
- `object_id` and `feature_label`: copied from the input CSV (the particle's identifier and its class name).
- `equivalent_diameter_um`: the particle's diameter, copied from the input.
- `placed`: `true` if the particle was placed, `false` if it was not.
- `x_um` and `y_um`: the centre of the particle on the slate; empty when it was not placed.
- `tries`: how many locations were tried for this particle (always 1 in centre placement).
- `edge_rejections`: tries rejected because the particle would have crossed the slate edge.
- `overlap_rejections`: tries rejected because the particle would have come closer to a placed particle than the allowed gap (touching is allowed when `buffer_um` is 0).

## 4. The summary file

The summary starts with a readable part: the settings, the numbers of particles placed and not placed, the requested and achieved coverage, and the names of the two files. It ends with a block of settings between `--- BEGIN REPRODUCTION PARAMETERS ---` and `--- END REPRODUCTION PARAMETERS ---`, written as JSON (a plain-text format of `"name": value` pairs inside curly braces). That block holds `slate`, `mode`, `max_tries`, `buffer_um`, `pixel_um`, `order`, `seed`, `input` and `library_versions`. The `input` object names the CSV and holds the SHA-256 checksum of its contents (a long code that changes if even one byte of the file changes), so a replay refuses a CSV that is not the one that was placed. When the particles came from a generator result in memory rather than a CSV, `input` holds the generator's own settings instead.

## 5. What each setting means

You set these in Python by calling `place_particles` with the particles (a generator result or the path of a measurements CSV) and the slate, then `write_placement_outputs` to save the result and `replay_placement` to repeat it from a summary.

- **Slate** (`slate`): the surface, a dictionary with a `shape`. A `square` has `side_um`, the length of one side. A `rectangle` has `width_um` and `height_um`. A `circle` has `diameter_um`. The slate is centred on the origin, and a particle must lie wholly inside it, though it may touch the edge.
- **Mode** (`mode`): `random` takes each particle in turn and tries random locations until one fits. `centre` puts each particle at the free spot closest to the slate centre, so a cluster grows outwards from the middle; it never rejects a spot until the cluster reaches the slate edge.
- **`max_tries`**: in random mode, how many random locations are tried for each particle before it is left unplaced. In centre mode it must be 1, because each particle gets one search.
- **`buffer_um`**: the smallest gap kept between the surfaces of two particles, counted once per pair. It is not applied at the slate edge, which a particle may touch. The default is 0, so particles may touch.
- **`pixel_um`**: the cell size, in micrometres, of the grid that centre placement searches. Smaller cells place particles more precisely but take longer; larger cells are quicker. Random placement does not use it. In centre mode every coordinate is a whole multiple of `pixel_um`.
- **`order`**: the order the particles are taken in. `as_given` takes them in the order of the input file. `shuffled` takes them in a random order fixed by the seed. To place the largest particles first or the smallest first, sort the input file by diameter and use `as_given`.
- **`seed`**: a whole number that fixes every random choice, so the same seed with the same settings and the same input always gives the same placements. With no seed, one is drawn and recorded in the summary.

### Requested coverage

Before anything is placed, the run reports the requested coverage on the terminal and in the summary: the total area of all the particles divided by the area of the slate. It is the fraction of the slate the particles would fill if all of them fitted. The summary also records the achieved coverage, which counts only the placed particles. With the generator example's particle mix (roughly 3,300 square micrometres per particle, an estimate), 10,000 particles on a slate 1 cm on each side request about 33% coverage, and on a slate 2 cm on each side about 8%. In random placement, rejected tries happen at any coverage, because a random try can land too near the slate edge or a placed particle. They become more frequent as the coverage increases. The run never refuses to start because of the coverage. It only reports the coverage, so a run with a low coverage is visible at the start.

## Good to know

- Centre placement time grows with the square of the number of particles: about two minutes for 2,000 particles on a 1 cm slate at 1 um cells. Raise `pixel_um` to make it quicker.
- A failed run writes no output files. The tool saves the placements file and then the summary; if the second save fails after the first replaced an old file (with `overwrite=True`), the error message says so.
- To repeat a saved run, call `replay_placement` with the summary file; it reads the CSV recorded beside it, or the path you give as a second argument.
