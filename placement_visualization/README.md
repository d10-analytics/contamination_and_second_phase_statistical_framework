# Particle placement pictures

This part of the repository draws pictures of a particle placement run that the [placement simulator](../placement_simulation/README.md) saved. It draws still images of the slate as it fills, a time-lapse video and a packing map. Each function reads the two saved files of a run and writes new files beside them. It never changes the saved files. Commands are typed into a terminal and are written for Linux and macOS.

## 1. Install the packages and ffmpeg

Follow steps 1 to 3 of the [synthetic data guide](../synthetic_data_generation/README.md). Those steps install Python, a virtual environment and the packages in `synthetic_data_generation/requirements.txt`, which include matplotlib. The drawing functions need no other Python packages.

Only the video needs ffmpeg, a separate program that encodes video. It must have the `libopenh264` encoder. One way to install it is to type `conda install ffmpeg` in a conda environment. The video function looks for ffmpeg in the folder of the Python that runs it. If ffmpeg is not there, it looks on the PATH. If it finds no ffmpeg, only the video function stops, with a message that tells you to install it. The still images and the packing map do not need ffmpeg.

## 2. Run the example

From the repository root folder, run:

```
python3 -m placement_visualization.example.run_example
```

The script `placement_visualization/example/run_example.py` is written to be read top to bottom like a notebook, with commentary before each of its six numbered steps. It generates the generator example's dataset (400 particles) and a second dataset with every class twice as large (800 particles, the same mix of classes). It places them on the same square slate, one run in each placement mode, and saves each run with `write_placement_outputs` in its own folder:

- `random` placement of the 400 particles tries up to 20 random locations for each particle and leaves it unplaced when all are rejected. Its run goes in `placement_visualization/example/output_random/`.
- `centre` placement of the 800 particles puts each particle at the free spot closest to the slate centre, so one cluster grows outwards from the middle. The 800 particles would cover about 82 % of the slate, more than a cluster of circles can fill, so the cluster reaches the slate edge and the particles that no longer fit are left unplaced. Its run goes in `placement_visualization/example/output_center/`.

For each run it then draws a still image after every 100 particles (four for the random run, eight for the centre run), a dashboard video, the packing map and the packing map series from the saved files. It uses a small picture size for the pictures. The 20 second videos draw 80 dashboard frames for the random run and 160 for the centre run, so the example takes about 40 seconds.

Both folders are created if missing and are not tracked by git. `placement_visualization/example/output_random/` holds:

- `placement_visualization/example/output_random/example_placements.csv`: the placements of the run.
- `placement_visualization/example/output_random/example_summary.txt`: the settings and results of the run.
- `placement_visualization/example/output_random/example_step_100.png`: the slate after 100 particles were attempted.
- `placement_visualization/example/output_random/example_step_200.png`: the slate after 200 particles.
- `placement_visualization/example/output_random/example_step_300.png`: the slate after 300 particles.
- `placement_visualization/example/output_random/example_step_400.png`: the slate after all 400 particles.
- `placement_visualization/example/output_random/example_dashboard.mp4`: the time-lapse video with the dashboard layout.
- `placement_visualization/example/output_random/example_packing_map.png`: the packing map at the default sigma.
- `example_packing_map_sigma_5.0.png`, `example_packing_map_sigma_15.0.png`, `example_packing_map_sigma_30.0.png`, `example_packing_map_sigma_60.2.png`, `example_packing_map_sigma_120.0.png` and `example_packing_map_sigma_240.0.png`: the packing map at six blur widths, 60.2 um being the default (the median diameter) for this dataset.
- `placement_visualization/example/output_random/example_packing_map_sigma_series.png`: those six maps in one grid of two rows of three.

`placement_visualization/example/output_center/` holds the same names for the centre run, with two differences: still images run from `example_step_100.png` to `example_step_800.png`, and the default-sigma map is `example_packing_map_sigma_60.9.png`, the median diameter of the 800 particles. Running the example again replaces these files. The placement example and its output folder are not changed.

## 3. The functions

Import the four drawing functions from `placement_visualization` and `read_placement_outputs` from `placement_simulation`. Each one takes `base_path`, the path of a saved run without its ending. For the files `example_placements.csv` and `example_summary.txt`, the `base_path` is `example`, with the folder in front of it.

- `read_placement_outputs(base_path)` reads the two saved files back into the result that was saved. The drawing functions use it. You can call it to look at a saved run.
- `write_still_images(base_path, every, *, long_side_px=2000, overwrite=False)` writes one PNG for every shown step and returns their paths.
- `write_video(base_path, every, video_seconds, *, fps=30, layout="slate", overwrite=False)` writes one MP4 time-lapse and returns its path.
- `write_packing_map(base_path, *, sigma_um=None, long_side_px=2000, overwrite=False)` writes one PNG and returns its path.
- `write_packing_map_series(base_path, sigmas_um, *, long_side_px=2000, overwrite=False)` writes the packing map once for each sigma in `sigmas_um` and a grid of them all, and returns the map paths and the grid path.

What each argument means:

- `every`: a whole number of 1 or more. A step is one particle attempted: the next particle in the saved order, which ends either placed or, when every location tried for it was rejected, not placed. The location tries inside one attempt are not separate steps. The shown steps are `every`, `2 x every` and so on up to the particle count. The last step is always shown. The stills and the video use the same shown steps.
- `video_seconds`: how long the video lasts, in seconds. The video has `round(video_seconds x fps)` frames, with a half rounding up. Each shown step is held for an equal share of them, as near as whole frames allow, so two steps' holds differ by at most one frame. A step needs at least one frame. If the video is too short, the function stops before it draws anything and names the smallest `every` that works.
- `fps`: the video frame rate. The default is 30.
- `layout`: `slate` shows only the slate, with a legend. `dashboard` shows the slate on the left, with a thick outline, and four graphs beside it. Top row: a size histogram of all classes together, filled for the particles placed so far inside a black outline of all particles; and a size histogram of the particles placed so far, a very thick solid black line for all classes together and a thick solid line in each class's slate colour for that class, drawn on top, every line drawn from zero. Bottom row: a line graph whose x axis counts particles attempted, with the packing factor (the share of the slate covered) as a thick blue line on the left axis, labelled `Packing Factor (%)`, which ends at the final coverage rounded up to the next 10 %, and the share of attempted particles that were placed within the run's `max_tries` tries as a thick red dashed line on the right axis, labelled for example `Placed within 20 tries (% of attempted)` (a centre run, which makes one search per particle, is labelled `Placed (% of attempted)`), which runs from just below the lowest share in the run (rounded down to 5 %) to a little above 100 %; and a bar chart with, in each size bin, a black bar for all classes together followed by one narrow bar per class. For a random run the bars show the mean number of location tries per placed particle, or, when `max_tries` is 1, the share of attempted particles not placed. For a centre run, which always makes one search per particle, they show the mean distance of the placed particles from the slate centre in micrometres, so you can see whether a size ends up nearer the middle or the rim of the cluster. Each line-graph axis label is bold and in its line's colour. Both layouts are 1920 x 1080 pixels. The video is H.264 with the `yuv420p` pixel format. The function prints the length of the video to the terminal before it starts to draw.
- `dashboard`: this is the name of the second `layout`. It is not an argument of its own.
- `long_side_px`: the default is 2000. For `write_still_images`, it is the length, in pixels, of the long side of the slate in each PNG, and it must be at least 400. The legend of every class is added below the slate in a fixed 10 point font, with a margin around everything, so a PNG is taller than `long_side_px`, and wider than the slate when the legend needs more width than a narrow slate. A smaller number makes `write_still_images` raise a `ValueError` naming `long_side_px` and the minimum, and write nothing. For `write_packing_map`, it is the number of grid cells along the long side of the slate. The PNG also holds a colour bar and the sigma text, so the slate fills less than that many pixels of it. A small number draws faster.
- `sigma_um`: the width, in micrometres, of the blur of the packing map. Each pixel shows the covered share of the area around it. The default is the median diameter of all particles in the file, placed or not. The map is coloured with the `inferno` colour map. Two lines under it give the sigma and the global average packing factor in percent, both rounded to one decimal place. The packing factor is the true area of the placed particles divided by the slate area, the achieved coverage in the run's summary, so it is the same at every sigma.
- `sigmas_um`: a list of blur widths in micrometres for `write_packing_map_series`, each drawn exactly as `write_packing_map` would draw it. An entry of `None` is the default sigma. The grid shows the maps three to a row in the order given. An empty list, or two sigmas that round to the same one decimal place, is refused.
- `overwrite`: with the default `False`, a function refuses to replace a file that exists. It checks every destination before it draws. With `True` it replaces the files.

A refused argument or file raises a `ValueError` before anything is drawn. A missing ffmpeg raises a `RuntimeError`. A failed picture or video never leaves a cut-off or temporary file.

## 4. The output file names

Each name starts with the file name of `base_path`, which is `<base>` here.

- `<base>_step_<k>.png`: a still image. `k` is the step. It has leading zeros to the number of digits of the particle count, for example `example_step_100.png`.
- `<base>_slate.mp4` and `<base>_dashboard.mp4`: the video for the `slate` layout and the `dashboard` layout.
- `<base>_packing_map.png`: the packing map.
- `<base>_packing_map_sigma_<sigma>.png`: the packing map of a series at one sigma, to one decimal place, for example `example_packing_map_sigma_30.0.png`.
- `<base>_packing_map_sigma_series.png`: the grid of every map in a series.
