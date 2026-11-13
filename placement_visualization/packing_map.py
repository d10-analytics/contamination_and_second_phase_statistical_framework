# This file writes the packing map of a saved placement run: a picture of how
# much of the slate the placed particles cover, smoothed so that a pixel shows
# the covered share of the neighbourhood around it. The slate is cut into
# square cells; a cell is covered when its centre lies inside a placed
# particle (true radius, no buffer) and is on the slate when its centre lies
# inside the slate edge, both decided by the same geometry functions the
# placement used. The covered cells are blurred with a Gaussian and divided by
# the same blur of the slate cells, so a cell near the slate edge is not read
# too low. The result is shown from 0 to 100 percent with a colour bar, and
# cells off the slate are left uncoloured. Two lines under the slate give the
# sigma used and the global average packing factor: the true area of the
# placed particles over the slate area, the run's achieved coverage, which the
# blur does not change. write_packing_map_series draws the same map at several
# sigmas, one PNG each, and joins them into one grid picture, so the effect of
# the blur width can be compared side by side.

from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from placement_simulation import geometry, placement
from placement_visualization import picture_files, slate_drawing

SUFFIX = "_packing_map.png"
SERIES_SUFFIX = "_packing_map_sigma_series.png"
SERIES_COLUMNS = 3
COLOUR_MAP = "inferno"
COLUMN_FRACTION = 0.22
BAR_START_IN_COLUMN = 0.08
BAR_WIDTH_IN_COLUMN = 0.2
BAR_MAX_FONT_POINTS = 10.0
SPACING_IN_FONTS = 0.35
TEXT_FILL = 0.7
TEXT_WIDTH_FRACTION = 0.96
MIN_FONT_POINTS = 1.0
FONT_STEP_POINTS = 0.5


def map_path(base_path):
    """Return the destination <base>_packing_map.png beside base_path."""
    return base_path.parent / (base_path.name + SUFFIX)


def series_map_path(base_path, sigma_um):
    """Return the destination <base>_packing_map_sigma_<sigma to one decimal>.png beside base_path."""
    return base_path.parent / f"{base_path.name}_packing_map_sigma_{sigma_um:.1f}.png"


def series_grid_path(base_path):
    """Return the destination <base>_packing_map_sigma_series.png beside base_path."""
    return base_path.parent / (base_path.name + SERIES_SUFFIX)


def default_sigma_um(rows):
    """Return the median diameter of all particles in the file, placed or not."""
    return float(np.median([row.equivalent_diameter_um for row in rows]))


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


def caption_lines(sigma_um, packing_factor):
    """Return the two caption lines: the sigma and the global average packing factor in percent, each to one decimal place."""
    return [f"blur sigma {sigma_um:.1f} um", f"global average packing factor {100.0 * packing_factor:.1f} %"]


def add_caption(figure, lines):
    """Add the caption lines, one per row of the strip under the slate, top line first, at one font size shrunk until every line lies wholly inside the figure at any size."""
    width_px = figure.get_figwidth() * figure.dpi
    height_px = figure.get_figheight() * figure.dpi
    strip_px = slate_drawing.text_strip_px(max(width_px, height_px), len(lines))
    row_px = strip_px / len(lines)
    size = TEXT_FILL * row_px * 72.0 / figure.dpi
    texts = []
    for position, line in enumerate(lines):
        centre_px = strip_px - (position + 0.5) * row_px
        texts.append(figure.text(0.5, centre_px / height_px, line, ha="center", va="center", fontsize=size))
    renderer = figure.canvas.get_renderer()
    while size > MIN_FONT_POINTS and max(text.get_window_extent(renderer).width for text in texts) > TEXT_WIDTH_FRACTION * width_px:
        size -= FONT_STEP_POINTS
        for text in texts:
            text.set_fontsize(size)
    return texts


def inside_figure(box, figure):
    """Say whether a bounding box lies wholly inside the figure."""
    return box.x0 >= 0 and box.y0 >= 0 and box.x1 <= figure.bbox.x1 and box.y1 <= figure.bbox.y1


def add_colour_bar(figure, axes, image):
    """Add the 0-100 colour bar in a column right of the slate, shrinking its fonts until bar, ticks and label lie wholly inside the figure."""
    width_px = figure.get_figwidth() * figure.dpi
    margin = slate_drawing.MARGIN_FRACTION * max(width_px, figure.get_figheight() * figure.dpi) / width_px
    column = COLUMN_FRACTION
    position = axes.get_position()
    axes.set_position([margin, position.y0, 1.0 - 2.0 * margin - column, position.height])
    axes.apply_aspect()
    box = axes.get_position()
    left = 1.0 - margin - column + BAR_START_IN_COLUMN * column
    cax = figure.add_axes([left, box.y0, BAR_WIDTH_IN_COLUMN * column, box.height])
    colour_bar = figure.colorbar(image, cax=cax)
    renderer = figure.canvas.get_renderer()
    size = BAR_MAX_FONT_POINTS
    while True:
        cax.tick_params(labelsize=size, length=SPACING_IN_FONTS * size, pad=SPACING_IN_FONTS * size)
        colour_bar.set_label("covered (%)", fontsize=size, labelpad=SPACING_IN_FONTS * size)
        if size <= MIN_FONT_POINTS or inside_figure(cax.get_tightbbox(renderer), figure):
            return colour_bar
        size -= FONT_STEP_POINTS


def build_packing_map_figure(result, sigma_um, long_side_px):
    """Return the figure: the slate's percentage map on a fixed 0-100 scale with a colour bar, and the sigma used and the global average packing factor printed under the slate."""
    percentage, cell = covered_percentage(result, sigma_um, long_side_px)
    lines = caption_lines(sigma_um, result.achieved_coverage)
    figure, axes = slate_drawing.make_slate_figure(result.slate, long_side_px, len(lines))
    half_width, half_height = slate_drawing.slate_extent(result.slate)
    columns = percentage.shape[1]
    rows = percentage.shape[0]
    extent = (-columns * cell / 2.0, columns * cell / 2.0, -rows * cell / 2.0, rows * cell / 2.0)
    image = axes.imshow(percentage, origin="lower", extent=extent, vmin=0.0, vmax=100.0, cmap=COLOUR_MAP, interpolation="nearest")
    axes.set_xlim(-half_width, half_width)
    axes.set_ylim(-half_height, half_height)
    slate_drawing.draw_outline(axes, result.slate)
    add_colour_bar(figure, axes, image)
    add_caption(figure, lines)
    return figure


def write_packing_map(base_path, *, sigma_um=None, long_side_px=2000, overwrite=False):
    """Write <base>_packing_map.png and return its path.

    sigma_um is the blur width in micrometres and defaults to the median
    diameter of all particles in the file, placed or not. Arguments, the saved files and the
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


def figure_pixels(figure):
    """Return the drawn figure as an RGBA array of whole pixels."""
    figure.canvas.draw()
    return np.asarray(figure.canvas.buffer_rgba()).copy()


def grid_of(pictures, columns):
    """Return one RGBA array holding the pictures in rows of columns, left to right and top to bottom, every cell as large as the largest picture and empty cells white."""
    cell_height = max(picture.shape[0] for picture in pictures)
    cell_width = max(picture.shape[1] for picture in pictures)
    rows = -(-len(pictures) // columns)
    grid = np.full((rows * cell_height, columns * cell_width, 4), 255, dtype=np.uint8)
    for position, picture in enumerate(pictures):
        top = (position // columns) * cell_height
        left = (position % columns) * cell_width
        grid[top : top + picture.shape[0], left : left + picture.shape[1]] = picture
    return grid


def write_packing_map_series(base_path, sigmas_um, *, long_side_px=2000, overwrite=False):
    """Write one packing map per sigma and a grid of them all; return (the map paths in sigma order, the grid path).

    sigmas_um lists the blur widths in micrometres; an entry of None is the
    default, the median diameter of all particles. Each map is
    <base>_packing_map_sigma_<sigma>.png, the sigma to one decimal place, drawn
    exactly as write_packing_map draws it, and the grid
    <base>_packing_map_sigma_series.png holds them in rows of three in the order
    given. An empty list, or two sigmas that name the same file, is refused.
    Arguments, the saved files and every destination are checked before
    anything is drawn; each refusal is a ValueError.
    """
    sigmas_um = list(sigmas_um)
    if len(sigmas_um) == 0:
        raise ValueError("sigmas_um must list at least one sigma")
    checked = [None if sigma is None else placement.check_number("sigma_um", sigma, False) for sigma in sigmas_um]
    long_side_px = picture_files.check_whole_number(long_side_px, "long_side_px")
    base_path = Path(base_path)
    result = picture_files.load_placement_result(base_path)
    default = default_sigma_um(result.rows)
    sigmas = [default if sigma is None else sigma for sigma in checked]
    paths = [series_map_path(base_path, sigma) for sigma in sigmas]
    if len(set(paths)) != len(paths):
        raise ValueError(f"sigmas_um {sigmas_um!r} name the same file twice after rounding to one decimal place")
    grid_path = series_grid_path(base_path)
    picture_files.check_destinations(paths + [grid_path], overwrite)
    pictures = []
    for sigma, path in zip(sigmas, paths):
        figure = build_packing_map_figure(result, sigma, long_side_px)
        picture_files.write_figure_png(figure, path)
        pictures.append(figure_pixels(figure))
    picture_files.write_image_png(grid_of(pictures, SERIES_COLUMNS), grid_path)
    return paths, grid_path
