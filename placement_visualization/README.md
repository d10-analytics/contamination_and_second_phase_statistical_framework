# Particle placement pictures

This part of the repository draws pictures of a particle placement run that the [placement simulator](../placement_simulation/README.md) saved. It draws still images of the slate as it fills, a time-lapse video and a packing map. Each function reads the two saved files of a run and writes new files beside them. It never changes the saved files. Commands are typed into a terminal and are written for Linux and macOS.

## 1. Install the packages and ffmpeg

Follow steps 1 to 3 of the [synthetic data guide](../synthetic_data_generation/README.md) and the [placement guide](../placement_simulation/README.md). Those steps install Python, a virtual environment and the packages in `synthetic_data_generation/requirements.txt`, which include matplotlib.

Only the video needs ffmpeg, a separate program that encodes video. It must have the `libopenh264` encoder. One way to install it is to type `conda install ffmpeg` in a conda environment. The video function looks for ffmpeg in the folder of the Python that runs it. If ffmpeg is not there, it looks on the PATH. If it finds no ffmpeg, only the video function stops, with a message that tells you to install it. The still images and the packing map do not need ffmpeg.

## 2. Run the example

From the repository root folder, run:

```
python3 -m placement_visualization.example.run_example
```

The script `placement_visualization/example/run_example.py` is written to be read top to bottom like a notebook, with commentary before each of its six numbered steps. It generates the generator example's dataset (400 particles), places the particles at random on a square slate, and saves the run with `write_placement_outputs`. Then it draws four still images, a dashboard video and the packing map from the saved files. It uses a small picture size, so it runs in a few seconds.

It writes eight files into the folder `placement_visualization/example/output/`, which is created if missing and is not tracked by git:

- `placement_visualization/example/output/example_placements.csv`: the placements of the run.
- `placement_visualization/example/output/example_summary.txt`: the settings and results of the run.
- `placement_visualization/example/output/example_step_100.png`: the slate after 100 particles were taken.
- `placement_visualization/example/output/example_step_200.png`: the slate after 200 particles.
- `placement_visualization/example/output/example_step_300.png`: the slate after 300 particles.
- `placement_visualization/example/output/example_step_400.png`: the slate after all 400 particles.
- `placement_visualization/example/output/example_dashboard.mp4`: the time-lapse video with the dashboard layout.
- `placement_visualization/example/output/example_packing_map.png`: the packing map.

Running the example again replaces these files. The placement example and its output folder are not changed.

## 3. The functions

Import the three drawing functions from `placement_visualization` and `read_placement_outputs` from `placement_simulation`. Each one takes `base_path`, the path of a saved run without its ending. For the files `example_placements.csv` and `example_summary.txt`, the `base_path` is `example`, with the folder in front of it.

- `read_placement_outputs(base_path)` reads the two saved files back into the result that was saved. The drawing functions use it. You can call it to look at a saved run.
- `write_still_images(base_path, every, *, long_side_px=2000, overwrite=False)` writes one PNG for every shown step and returns their paths.
- `write_video(base_path, every, video_seconds, *, fps=30, layout="slate", overwrite=False)` writes one MP4 time-lapse and returns its path.
- `write_packing_map(base_path, *, sigma_um=None, long_side_px=2000, overwrite=False)` writes one PNG and returns its path.

What each argument means:

- `every`: a whole number of 1 or more. A step is one particle taken, placed or not, in the saved order. The shown steps are `every`, `2 x every` and so on up to the particle count. The last step is always shown. The stills and the video use the same shown steps.
- `video_seconds`: how long the video lasts, in seconds. The video has `round(video_seconds x fps)` frames. Each shown step is held for an equal share of them. A step needs at least one frame. If the video is too short, the function stops before it draws anything and names the smallest `every` that works.
- `fps`: the video frame rate. The default is 30.
- `layout`: `slate` shows only the slate, with a legend. `dashboard` shows the slate with a size histogram, a line graph over the steps and a bar chart by size bin. Both layouts are 1920 x 1080 pixels. The video is H.264 with the `yuv420p` pixel format. The function prints the length of the video to the terminal before it starts to draw.
- `dashboard`: this is the name of the second `layout`. It is not an argument of its own.
- `long_side_px`: the default is 2000. For `write_still_images`, it is the length, in pixels, of the long side of each PNG. The slate is drawn inside that picture with a margin and a legend. For `write_packing_map`, it is the number of grid cells along the long side of the slate. The PNG also holds a colour bar and the sigma text, so the slate fills less than that many pixels of it. A small number draws faster.
- `sigma_um`: the width, in micrometres, of the blur of the packing map. Each pixel shows the covered share of the area around it. The default is twice the largest particle diameter in the file.
- `overwrite`: with the default `False`, a function refuses to replace a file that exists. It checks every destination before it draws. With `True` it replaces the files.

A refused argument or file raises a `ValueError` before anything is drawn. A missing ffmpeg raises a `RuntimeError`. A failed picture or video never leaves a cut-off or temporary file.

## 4. The output file names

Each name starts with the file name of `base_path`, which is `<base>` here.

- `<base>_step_<k>.png`: a still image. `k` is the step. It has leading zeros to the number of digits of the particle count, for example `example_step_100.png`.
- `<base>_slate.mp4` and `<base>_dashboard.mp4`: the video for the `slate` layout and the `dashboard` layout.
- `<base>_packing_map.png`: the packing map.
