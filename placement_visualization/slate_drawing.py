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


def add_legend_columns(figure, handles, size):
    """Add a legend with the most columns that fit across the figure at this font size (one column when none does) and return (legend, its pixel box)."""
    width_px = figure.get_figwidth() * figure.dpi
    renderer = figure.canvas.get_renderer()
    low, high = 1, len(handles)
    while True:
        columns = (low + high + 1) // 2
        legend = figure.legend(handles=handles, loc="lower center", ncol=columns, frameon=False, fontsize=size)
        box = legend.get_window_extent(renderer)
        fits = box.x0 >= 0 and box.x1 <= width_px
        if low >= high:
            return legend, box
        legend.remove()
        if fits:
            low = columns
        else:
            high = columns - 1


def add_fitted_legend(figure, handles, max_height_px=None):
    """Add a legend along the bottom of the figure and return it, laid out so it lies wholly inside the figure width.

    Columns are as many as fit across the figure, giving the fewest rows; the
    font shrinks until the legend also fits the given height (when there is
    one), down to a minimum size.
    """
    width_px = figure.get_figwidth() * figure.dpi
    size = LEGEND_MAX_FONT_POINTS
    while True:
        legend, box = add_legend_columns(figure, handles, size)
        ratio = 1.0
        if box.x0 < 0 or box.x1 > width_px:
            ratio = min(ratio, width_px / box.width)
        if max_height_px is not None and box.y1 > max_height_px:
            ratio = min(ratio, max_height_px / box.y1)
        if ratio >= 1.0 or size <= LEGEND_MIN_FONT_POINTS:
            return legend
        legend.remove()
        size = max(LEGEND_MIN_FONT_POINTS, math.floor(size * ratio * 10.0) / 10.0 - LEGEND_FONT_MARGIN_POINTS)


def legend_height_px(width_px, labels, long_side_px):
    """Return the height in pixels the legend of these labels measures, from the picture bottom, in a picture width_px wide."""
    cap = LEGEND_MAX_FRACTION * long_side_px
    scratch = Figure(figsize=(width_px / FIGURE_DPI, max(cap, 1.0) / FIGURE_DPI), dpi=FIGURE_DPI)
    FigureCanvasAgg(scratch)
    legend = add_fitted_legend(scratch, legend_handles(labels), cap)
    return legend.get_window_extent(scratch.canvas.get_renderer()).y1


def picture_layout(slate, long_side_px, labels=None):
    """Return (width px, height px, strip px, margin px, pixels per micrometre); the longer side is long_side_px.

    The strip under the slate holds the legend of the labels as measured at the
    picture's own width, or a single text row when labels is None.
    """
    half_width, half_height = slate_extent(slate)
    margin = MARGIN_FRACTION * long_side_px
    if labels is None:
        strip = min(LEGEND_ROW_PX, LEGEND_MAX_FRACTION * long_side_px)
    else:
        strip = legend_height_px(long_side_px, labels, long_side_px)
    while True:
        width, height, scale = fit_picture(half_width, half_height, long_side_px, margin, strip)
        if labels is None:
            break
        needed = legend_height_px(width, labels, long_side_px)
        if needed <= strip:
            break
        strip = needed
    return width, height, strip, margin, scale


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


def make_slate_figure(slate, long_side_px, labels=None):
    """Return (a Figure whose PNG long side is long_side_px, its one axes showing exactly the slate), with room under it for the legend of the labels."""
    width, height, legend, margin, scale = picture_layout(slate, long_side_px, labels)
    half_width, half_height = slate_extent(slate)
    # The half pixel keeps the size from rounding down when converted to inches and back.
    figure = Figure(figsize=((width + 0.5) / FIGURE_DPI, (height + 0.5) / FIGURE_DPI), dpi=FIGURE_DPI, facecolor=BACKGROUND)
    FigureCanvasAgg(figure)
    axes = figure.add_axes(
        [margin / width, (margin + legend) / height, 2 * half_width * scale / width, 2 * half_height * scale / height]
    )
    axes.set_xlim(-half_width, half_width)
    axes.set_ylim(-half_height, half_height)
    axes.set_aspect("equal")
    axes.set_axis_off()
    return figure, axes


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


def draw_legend(figure, colours):
    """Add a legend listing every class, in sorted order, along the bottom of the figure, wholly inside its width."""
    handles = [Patch(facecolor=colour, label=label) for label, colour in colours.items()]
    long_side_px = max(figure.get_figwidth(), figure.get_figheight()) * figure.dpi
    max_height_px = LEGEND_MAX_FRACTION * long_side_px
    return add_fitted_legend(figure, handles, max_height_px)
