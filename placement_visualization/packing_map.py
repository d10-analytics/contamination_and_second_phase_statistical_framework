# This file writes the packing map of a saved placement run: a picture of how
# much of the slate the placed particles cover, smoothed so that a pixel shows
# the covered share of the neighbourhood around it. The slate is cut into
# square cells; a cell is covered when its centre lies inside a placed
# particle (true radius, no buffer) and is on the slate when its centre lies
# inside the slate edge, both decided by the same geometry functions the
# placement used. The covered cells are blurred with a Gaussian and divided by
# the same blur of the slate cells, so a cell near the slate edge is not read
# too low. The result is shown from 0 to 100 percent with a colour bar, and
# cells off the slate are left uncoloured.

from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from placement_simulation import geometry, placement
from placement_visualization import picture_files, slate_drawing

SUFFIX = "_packing_map.png"
SIGMA_FACTOR = 2.0
COLOUR_MAP = "viridis"
COLOUR_BAR_PAD = 0.03
TEXT_FILL = 0.7
TEXT_WIDTH_FRACTION = 0.96
MIN_FONT_POINTS = 1.0
FONT_STEP_POINTS = 0.5


def map_path(base_path):
    """Return the destination <base>_packing_map.png beside base_path."""
    return base_path.parent / (base_path.name + SUFFIX)


def default_sigma_um(rows):
    """Return twice the largest diameter of all particles in the file, placed or not."""
    return SIGMA_FACTOR * max(row.equivalent_diameter_um for row in rows)


def cell_grid(slate, long_side_px):
    """Return (cell size in um, centre x values, centre y values); long_side_px cells span the slate's long side, centred on the slate."""
    half_width, half_height = slate_drawing.slate_extent(slate)
    long_side = 2.0 * max(half_width, half_height)
    cell = long_side / long_side_px
    columns = max(1, round(2.0 * half_width / cell))
    rows = max(1, round(2.0 * half_height / cell))
    x_centres = (np.arange(columns) - (columns - 1) / 2.0) * cell
    y_centres = (np.arange(rows) - (rows - 1) / 2.0) * cell
    return cell, x_centres, y_centres


def covered_cells(rows, x_centres, y_centres):
    """Return a boolean grid (rows by columns, row 0 lowest y) of cells whose centre lies inside a placed particle.

    Each particle's clearance is evaluated only over the cells inside its
    bounding box, one cell wider on every side so rounding cannot drop an edge cell.
    """
    covered = np.zeros((len(y_centres), len(x_centres)), dtype=bool)
    for row in rows:
        if not row.placed:
            continue
        radius = row.equivalent_diameter_um / 2.0
        first_column = max(0, int(np.searchsorted(x_centres, row.x_um - radius, side="left")) - 1)
        last_column = min(len(x_centres), int(np.searchsorted(x_centres, row.x_um + radius, side="right")) + 1)
        first_row = max(0, int(np.searchsorted(y_centres, row.y_um - radius, side="left")) - 1)
        last_row = min(len(y_centres), int(np.searchsorted(y_centres, row.y_um + radius, side="right")) + 1)
        if first_column >= last_column or first_row >= last_row:
            continue
        grid_x, grid_y = np.meshgrid(x_centres[first_column:last_column], y_centres[first_row:last_row])
        gaps = geometry.clearance(grid_x, grid_y, row.x_um, row.y_um, radius, 0.0)
        covered[first_row:last_row, first_column:last_column] |= gaps <= 0
    return covered


def slate_cells(slate, x_centres, y_centres):
    """Return a boolean grid of cells whose centre lies on the slate (edge distance 0 or more)."""
    grid_x, grid_y = np.meshgrid(x_centres, y_centres)
    return geometry.edge_distance(slate, grid_x, grid_y) >= 0


def covered_percentage(result, sigma_um, long_side_px):
    """Return (the masked percentage grid, the cell size in um): blurred covered cells over blurred slate cells, times 100, masked off the slate."""
    cell, x_centres, y_centres = cell_grid(result.slate, long_side_px)
    covered = covered_cells(result.rows, x_centres, y_centres)
    on_slate = slate_cells(result.slate, x_centres, y_centres)
    sigma_cells = sigma_um / cell
    blurred_covered = gaussian_filter(covered.astype(float), sigma_cells)
    blurred_slate = gaussian_filter(on_slate.astype(float), sigma_cells)
    percentage = np.zeros(blurred_slate.shape)
    np.divide(100.0 * blurred_covered, blurred_slate, out=percentage, where=blurred_slate > 0)
    return np.ma.masked_array(percentage, mask=~on_slate), cell


def add_sigma_text(figure, sigma_um):
    """Add the sigma text in the strip under the slate, shrunk until it lies wholly inside the figure at any size."""
    width_px = figure.get_figwidth() * figure.dpi
    height_px = figure.get_figheight() * figure.dpi
    strip_px = min(slate_drawing.LEGEND_ROW_PX, slate_drawing.LEGEND_MAX_FRACTION * max(width_px, height_px))
    size = TEXT_FILL * strip_px * 72.0 / figure.dpi
    text = figure.text(0.5, strip_px / 2.0 / height_px, f"blur sigma {sigma_um:g} um", ha="center", va="center", fontsize=size)
    renderer = figure.canvas.get_renderer()
    while size > MIN_FONT_POINTS and text.get_window_extent(renderer).width > TEXT_WIDTH_FRACTION * width_px:
        size -= FONT_STEP_POINTS
        text.set_fontsize(size)
    return text


def build_packing_map_figure(result, sigma_um, long_side_px):
    """Return the figure: the slate's percentage map on a fixed 0-100 scale with a colour bar and the sigma used printed under the slate."""
    percentage, cell = covered_percentage(result, sigma_um, long_side_px)
    figure, axes = slate_drawing.make_slate_figure(result.slate, long_side_px, 1)
    half_width, half_height = slate_drawing.slate_extent(result.slate)
    columns = percentage.shape[1]
    rows = percentage.shape[0]
    extent = (-columns * cell / 2.0, columns * cell / 2.0, -rows * cell / 2.0, rows * cell / 2.0)
    image = axes.imshow(percentage, origin="lower", extent=extent, vmin=0.0, vmax=100.0, cmap=COLOUR_MAP, interpolation="nearest")
    axes.set_xlim(-half_width, half_width)
    axes.set_ylim(-half_height, half_height)
    slate_drawing.draw_outline(axes, result.slate)
    colour_bar = figure.colorbar(image, ax=axes, pad=COLOUR_BAR_PAD)
    colour_bar.set_label("covered (%)")
    add_sigma_text(figure, sigma_um)
    return figure


def write_packing_map(base_path, *, sigma_um=None, long_side_px=2000, overwrite=False):
    """Write <base>_packing_map.png and return its path.

    sigma_um is the blur width in micrometres and defaults to twice the largest
    diameter of all particles in the file. Arguments, the saved files and the
    destination are checked before anything is drawn; each refusal is a
    ValueError, and a failure while writing leaves no cut-off or temporary file.
    """
    if sigma_um is not None:
        sigma_um = placement.check_number("sigma_um", sigma_um, False)
    long_side_px = picture_files.check_whole_number(long_side_px, "long_side_px")
    result = picture_files.load_placement_result(base_path)
    destination = map_path(Path(base_path))
    picture_files.check_destinations([destination], overwrite)
    if sigma_um is None:
        sigma_um = default_sigma_um(result.rows)
    figure = build_packing_map_figure(result, sigma_um, long_side_px)
    picture_files.write_figure_png(figure, destination)
    return destination
