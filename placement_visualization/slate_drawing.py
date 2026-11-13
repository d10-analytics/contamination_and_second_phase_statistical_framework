# This file draws the slate for every picture: the slate outline and each
# placed particle as a true-size circle (its real diameter in micrometres, not
# a marker size), coloured by class, with a legend naming every class. Figures
# are built with matplotlib's Figure class directly, never through pyplot, so
# nothing is left open between pictures. The axes show exactly the slate, with
# equal scaling, so a pixel position can be found through the axes' transData.

import math

from matplotlib import colormaps
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import EllipseCollection
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Patch, Rectangle

from placement_simulation import geometry

FIGURE_DPI = 100
MARGIN_FRACTION = 0.02
LEGEND_ROW_PX = 24
LEGEND_MAX_FRACTION = 0.3
LEGEND_MAX_FONT_POINTS = 10.0
LEGEND_MIN_FONT_POINTS = 1.0
LEGEND_FONT_MARGIN_POINTS = 0.1
BACKGROUND = "white"
OUTLINE_COLOUR = "black"


def slate_extent(slate):
    """Return the half width and half height of the slate in micrometres (a circle gives its radius twice)."""
    if slate["shape"] == "circle":
        radius = slate["diameter_um"] / 2.0
        return radius, radius
    return geometry.half_extents(slate)


def class_colours(rows):
    """Return a dictionary from every class in the rows to a tab10 colour, in sorted-label order, repeating after ten."""
    palette = colormaps["tab10"].colors
    labels = sorted({row.feature_label for row in rows})
    colours = {}
    for position, label in enumerate(labels):
        colours[label] = palette[position % len(palette)]
    return colours


def legend_handles(labels):
    """Return one patch handle per label, in order, for measuring a legend."""
    return [Patch(label=label) for label in labels]


def fit_columns(figure, handles, size):
    """Return the most columns that fit across the figure at this font size (one when none does)."""
    width_px = figure.get_figwidth() * figure.dpi
    renderer = figure.canvas.get_renderer()
    low, high = 1, len(handles)
    while low < high:
        columns = (low + high + 1) // 2
        legend = figure.legend(handles=handles, loc="lower center", ncol=columns, frameon=False, fontsize=size)
        box = legend.get_window_extent(renderer)
        legend.remove()
        if box.x0 >= 0 and box.x1 <= width_px:
            low = columns
        else:
            high = columns - 1
    return low


def add_legend_columns(figure, handles, size):
    """Add a legend with the most columns that fit across the figure at this font size (one column when none does) and return (legend, its pixel box)."""
    columns = fit_columns(figure, handles, size)
    legend = figure.legend(handles=handles, loc="lower center", ncol=columns, frameon=False, fontsize=size)
    return legend, legend.get_window_extent(figure.canvas.get_renderer())


def add_fitted_legend(figure, handles, max_height_px):
    """Add a legend along the bottom of the figure and return it.

    Used by the video and dashboard frames, whose size is fixed: columns are as
    many as fit across the figure, giving the fewest rows, and the font shrinks
    until the legend also fits the figure width and max_height_px, down to a
    minimum size. At that minimum the legend is returned as it is and may be
    wider or taller than allowed.
    """
    width_px = figure.get_figwidth() * figure.dpi
    size = LEGEND_MAX_FONT_POINTS
    while True:
        legend, box = add_legend_columns(figure, handles, size)
        ratio = 1.0
        if box.x0 < 0 or box.x1 > width_px:
            ratio = min(ratio, width_px / box.width)
        if box.y1 > max_height_px:
            ratio = min(ratio, max_height_px / box.y1)
        if ratio >= 1.0 or size <= LEGEND_MIN_FONT_POINTS:
            return legend
        legend.remove()
        size = max(LEGEND_MIN_FONT_POINTS, math.floor(size * ratio * 10.0) / 10.0 - LEGEND_FONT_MARGIN_POINTS)


def fit_picture(half_width, half_height, long_side_px, margin, strip):
    """Return (width px, height px, pixels per micrometre) of the slate fitted in the long side with the margin and strip."""
    by_width = (long_side_px - 2 * margin) / (2 * half_width)
    by_height = (long_side_px - 2 * margin - strip) / (2 * half_height)
    scale = min(by_width, by_height)
    width = round(2 * half_width * scale + 2 * margin)
    height = round(2 * half_height * scale + 2 * margin + strip)
    if by_width <= by_height:
        width = long_side_px
    else:
        height = long_side_px
    return max(width, 1), max(height, 1), scale


def new_figure(width, height):
    """Return a figure whose canvas is exactly width x height pixels."""
    # The half pixel keeps the size from rounding down when converted to inches and back.
    figure = Figure(figsize=((width + 0.5) / FIGURE_DPI, (height + 0.5) / FIGURE_DPI), dpi=FIGURE_DPI, facecolor=BACKGROUND)
    FigureCanvasAgg(figure)
    return figure


def slate_axes(figure, slate, left_px, bottom_px, scale, width_px, height_px):
    """Add the one axes showing exactly the slate at scale pixels per micrometre, its lower left corner at the pixel position given, positions being fractions of the width_px and height_px given."""
    half_width, half_height = slate_extent(slate)
    axes = figure.add_axes(
        [left_px / width_px, bottom_px / height_px, 2 * half_width * scale / width_px, 2 * half_height * scale / height_px]
    )
    axes.set_xlim(-half_width, half_width)
    axes.set_ylim(-half_height, half_height)
    axes.set_aspect("equal")
    axes.set_axis_off()
    return axes


def make_slate_figure(slate, long_side_px):
    """Return (a Figure whose PNG long side is long_side_px, its one axes showing exactly the slate), with a margin and one text row under the slate."""
    half_width, half_height = slate_extent(slate)
    margin = MARGIN_FRACTION * long_side_px
    strip = min(LEGEND_ROW_PX, LEGEND_MAX_FRACTION * long_side_px)
    width, height, scale = fit_picture(half_width, half_height, long_side_px, margin, strip)
    figure = new_figure(width, height)
    return figure, slate_axes(figure, slate, margin, margin + strip, scale, width, height)


def make_slate_figure_with_legend(slate, long_side_px, colours):
    """Return (a Figure, its one axes showing exactly the slate), the slate's long side being exactly long_side_px pixels.

    The legend of the classes is drawn below the slate in LEGEND_MAX_FONT_POINTS
    type. Its columns are chosen once, from the width of the slate and its
    margins (one column when that is too narrow), and it is measured once; the
    picture is made as tall as the slate, the legend and the margins need, and
    as wide as the slate or the legend needs, so the legend never touches the
    slate or leaves the picture.
    """
    half_width, half_height = slate_extent(slate)
    scale = long_side_px / (2 * max(half_width, half_height))
    slate_width, slate_height = 2 * half_width * scale, 2 * half_height * scale
    margin = MARGIN_FRACTION * long_side_px
    handles = [Patch(facecolor=colour, label=label) for label, colour in colours.items()]
    scratch = new_figure(math.ceil(slate_width + 2 * margin), 1)
    columns = fit_columns(scratch, legend_handles(colours), LEGEND_MAX_FONT_POINTS)
    legend = scratch.legend(handles=handles, ncol=columns, frameon=False, fontsize=LEGEND_MAX_FONT_POINTS)
    box = legend.get_window_extent(scratch.canvas.get_renderer())
    width = math.ceil(max(slate_width, box.width) + 2 * margin)
    height = math.ceil(slate_height + box.height + 3 * margin)
    figure = new_figure(width, height)
    figure.legend(
        handles=handles,
        loc="lower center",
        ncol=columns,
        frameon=False,
        fontsize=LEGEND_MAX_FONT_POINTS,
        bbox_to_anchor=(width / 2.0 / figure.get_figwidth() / figure.dpi, margin / figure.get_figheight() / figure.dpi),
        bbox_transform=figure.transFigure,
        borderaxespad=0.0,
    )
    full_width, full_height = figure.get_figwidth() * figure.dpi, figure.get_figheight() * figure.dpi
    return figure, slate_axes(figure, slate, (width - slate_width) / 2.0, 2 * margin + box.height, scale, full_width, full_height)


def draw_outline(axes, slate):
    """Draw the slate outline on the axes."""
    half_width, half_height = slate_extent(slate)
    if slate["shape"] == "circle":
        outline = Circle((0.0, 0.0), half_width, fill=False, edgecolor=OUTLINE_COLOUR, linewidth=1.0)
    else:
        outline = Rectangle((-half_width, -half_height), 2 * half_width, 2 * half_height, fill=False, edgecolor=OUTLINE_COLOUR, linewidth=1.0)
    axes.add_patch(outline)


def draw_slate(axes, slate, rows, colours):
    """Draw the slate outline and, as true-size circles, every placed row in rows coloured by its class."""
    draw_outline(axes, slate)
    placed = [row for row in rows if row.placed]
    if len(placed) == 0:
        return
    diameters = [row.equivalent_diameter_um for row in placed]
    collection = EllipseCollection(
        diameters,
        diameters,
        0.0,
        units="xy",
        offsets=[(row.x_um, row.y_um) for row in placed],
        offset_transform=axes.transData,
        facecolors=[colours[row.feature_label] for row in placed],
        edgecolors="none",
    )
    axes.add_collection(collection)


def draw_legend(figure, colours, max_height_px):
    """Add a legend listing every class, in sorted order, along the bottom of the figure, fitted by add_fitted_legend to the width and to max_height_px."""
    handles = [Patch(facecolor=colour, label=label) for label, colour in colours.items()]
    return add_fitted_legend(figure, handles, max_height_px)
