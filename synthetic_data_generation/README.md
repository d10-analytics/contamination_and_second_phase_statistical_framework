# Synthetic particle measurement generator

This tool makes made-up (synthetic) particle measurements, such as the diameters of powder particles in several classes, for trying out the statistical framework. It writes two files: a CSV (a plain table of numbers that Excel and other programs open) and a text summary that records every setting, so the same CSV can be recreated exactly. The steps below assume you have never used Python. Commands are typed into a terminal (a window where you type commands); they are written for Linux and macOS.

## 1. Check that Python 3.10 or newer is installed

Python is the programming language the tool is written in. In a terminal, run:

```
python3 --version
```

It should print `Python 3.10` or a higher number. If the number is lower (for example `Python 3.9.6`), or the command is not found, install a newer Python from https://www.python.org/downloads/ first and open a new terminal. The packages in the next steps cannot be installed on Python 3.9 or older.

Windows (untested): use `py --version` instead of `python3 --version`.

## 2. Create and activate a virtual environment outside the repository

A virtual environment is a private folder of installed packages, so this tool's packages do not mix with other Python work on your computer. Create it outside this repository (here, in your home folder) so it does not clutter the repository:

```
python3 -m venv ~/synthetic-data-venv
source ~/synthetic-data-venv/bin/activate
```

After activation the terminal prompt starts with `(synthetic-data-venv)`. Do the activation again in every new terminal.

Windows (untested): `py -m venv %USERPROFILE%\synthetic-data-venv` and then `%USERPROFILE%\synthetic-data-venv\Scripts\activate`.

## 3. Install the packages

Change into the repository root folder (the top folder of this project, the one that also holds `LICENSE` and the folder `synthetic_data_generation`), then install the packages listed in `synthetic_data_generation/requirements.txt` (NumPy and SciPy draw the random numbers, matplotlib draws the plot). This step needs an internet connection; running the tool afterwards does not.

```
pip install -r synthetic_data_generation/requirements.txt
```

## 4. Run the example

From the repository folder, run:

```
python3 -m synthetic_data_generation.example.run_example
```

The script `synthetic_data_generation/example/run_example.py` does five numbered steps: it generates a dataset from `synthetic_data_generation/example/example_input.jsonl`, saves it, reads the CSV back, draws a plot, and then recreates the CSV from the summary and checks that the two are identical. The last line it prints is `identical: True`. It leaves three files in the folder `synthetic_data_generation/example/output/`, which is created if missing and is not tracked by git:

- `synthetic_data_generation/example/output/example_particles.csv`: one row per particle, with the columns `feature_label` (the class name), `object_id`, `contour_area_um2`, `equivalent_diameter_um` and `volume_um3`.
- `synthetic_data_generation/example/output/example_particles_summary.txt`: every setting used, and a block of reproduction parameters that lets the CSV be recreated.
- `synthetic_data_generation/example/output/example_particles_plot.png`: the plot.

Running the example again replaces these files. The example always uses the same seed (a whole number that fixes the random draws), so you get the same particles each time.

### The example input file

`synthetic_data_generation/example/example_input.jsonl` is a JSONL file: each line is one complete entry, written as a JSON object (curly braces holding `"name": value` pairs). A JSONL file cannot hold a comment, so here is what each key means:

- `class`: the name of the particle class, copied unchanged into the CSV column `feature_label`.
- `distribution`: the kind of random draw: `gamma`, `gaussian` or `gev`.
- `N`: how many particles to keep for this class.
- Gaussian: `mean` and `stdev` (spread). Gamma: `shape` and `scale`. GEV: `shape`, `loc` (location) and `scale`.

The example has three entries, in this order: `matrix` (gamma), `oxide` (gaussian) and `inclusion` (gev). The sieve limits (the smallest and largest diameter kept) and the seed are set in `synthetic_data_generation/example/run_example.py`.

## 5. Open the plot

Open `synthetic_data_generation/example/output/example_particles_plot.png` with any picture viewer or from your file manager. The left picture is a diameter histogram per class; the right one is a bar chart of how many particles each class kept.

## 6. The command-line form

The same kind of run can be started from the terminal with your own input file. This line uses the example's own input file, seed and sieve limits (10 and 120 micrometres), so the CSV it writes is byte for byte the same as the example's CSV. It writes into your home folder:

```
python3 -m synthetic_data_generation --input-file synthetic_data_generation/example/example_input.jsonl --seed 20261029 --threshold-lower 10 --threshold-upper 120 --output ~/synthetic_particles.csv
```

It writes `~/synthetic_particles.csv` and `~/synthetic_particles_summary.txt`. A second run refuses to replace existing files unless you add `--overwrite`. To recreate an earlier run exactly, use `--from-summary` with the summary file.

## Good to know

- The tool tries each particle again until it falls inside the sieve limits. `max_attempts` (default 100,000) caps the tries per particle, so the worst case is N × `max_attempts` draws before the tool stops with an error. Impossible limits therefore fail after a long wait, not instantly.
- A failed run writes no output files. One small risk remains: the tool saves the CSV and then the summary, and if the computer fails exactly between those two moments, or the second save fails after the CSV replaced an old one (with `--overwrite`), the new CSV can sit beside the old summary. The error message says so when the tool notices.
