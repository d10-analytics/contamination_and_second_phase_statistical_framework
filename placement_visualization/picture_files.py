# This file holds what every drawing function shares: reading a saved run,
# checking a whole-number argument, checking the destinations before anything
# is drawn, and writing a finished figure. A figure is first written to a
# hidden temporary file and then renamed into place, so a failed drawing never
# leaves a cut-off picture or destroys an earlier good one.

import numbers
import os

from matplotlib import image

from placement_simulation import saving
from synthetic_data_generation import outputs


def check_whole_number(value, name):
    """Return value when it is a whole number of 1 or more (a boolean is refused); else raise ValueError."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
        raise ValueError(f"{name} must be a whole number of 1 or more, got {value!r}")
    return int(value)


def load_placement_result(base_path):
    """Read the saved run at base_path through the placement reader; raise ValueError for a refusal or a file with no particles."""
    result = saving.read_placement_outputs(base_path)
    if len(result.rows) == 0:
        raise ValueError(f"the placements file for {str(base_path)!r} has no particles, so there is nothing to draw")
    return result


def check_destinations(paths, overwrite):
    """Check every destination before anything is drawn; raise ValueError for a folder, a non-regular file or an existing file."""
    outputs.check_paths(paths, overwrite, saving.OVERWRITE_HINT)


def write_figure_png(figure, destination):
    """Write the figure as a PNG to a temporary file beside destination, then rename it into place; the temporary file is always removed."""
    temporary = outputs.temporary_path_for(destination)
    try:
        figure.savefig(temporary, format="png")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def write_image_png(pixels, destination):
    """Write an RGBA pixel array as a PNG to a temporary file beside destination, then rename it into place; the temporary file is always removed."""
    temporary = outputs.temporary_path_for(destination)
    try:
        image.imsave(temporary, pixels, format="png")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
