# This file writes the still images of a saved placement run. A step is one
# particle taken, placed or not, in the saved order. One picture is written for
# every chosen step and always one for the last step, each showing the
# particles placed among the steps taken so far. The shown-steps function is
# also what the video uses, so stills and video agree on which steps appear.

from pathlib import Path

from placement_visualization import picture_files, slate_drawing


def shown_steps(particle_count, every):
    """Return the shown steps: every, 2*every, ... up to the count, plus the count itself when it is not a multiple of every."""
    steps = list(range(every, particle_count + 1, every))
    if len(steps) == 0 or steps[-1] != particle_count:
        steps.append(particle_count)
    return steps


def still_paths(base_path, steps, particle_count):
    """Return the PNG path for each step: <base>_step_<k>.png, k padded with zeros to the digits of the particle count."""
    base_path = Path(base_path)
    digits = len(str(particle_count))
    paths = []
    for step in steps:
        paths.append(base_path.parent / f"{base_path.name}_step_{step:0{digits}d}.png")
    return paths


def build_still_figure(result, step, long_side_px):
    """Return the figure for one step: the slate with the particles placed among the first step taken, and the legend."""
    colours = slate_drawing.class_colours(result.rows)
    figure, axes = slate_drawing.make_slate_figure(result.slate, long_side_px, len(colours))
    slate_drawing.draw_slate(axes, result.slate, result.rows[:step], colours)
    slate_drawing.draw_legend(figure, colours)
    return figure


def write_still_images(base_path, every, *, long_side_px=2000, overwrite=False):
    """Write one PNG per shown step beside base_path and return their paths in step order.

    Arguments, the saved files, the particle count and every destination are
    checked before anything is drawn; each refusal is a ValueError. A failure
    while drawing leaves the files of the earlier steps in place.
    """
    every = picture_files.check_whole_number(every, "every")
    long_side_px = picture_files.check_whole_number(long_side_px, "long_side_px")
    result = picture_files.load_placement_result(base_path)
    particle_count = len(result.rows)
    steps = shown_steps(particle_count, every)
    paths = still_paths(base_path, steps, particle_count)
    picture_files.check_destinations(paths, overwrite)
    for step, path in zip(steps, paths):
        figure = build_still_figure(result, step, long_side_px)
        picture_files.write_figure_png(figure, path)
    return paths
