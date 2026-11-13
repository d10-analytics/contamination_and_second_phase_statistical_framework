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
LEGEND_COLUMN_PX = 120
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


def legend_columns(width_px, class_count):
    """Return how many legend columns fit across the picture."""
    return max(1, min(class_count, width_px // LEGEND_COLUMN_PX))


def legend_height_px(width_px, class_count, long_side_px):
    """Return the height in pixels reserved under the slate for the legend, capped at part of the picture."""
    rows = math.ceil(class_count / legend_columns(width_px, class_count))
    return min(rows * LEGEND_ROW_PX, LEGEND_MAX_FRACTION * long_side_px)


def picture_layout(slate, long_side_px, class_count):
    """Return (width px, height px, legend px, margin px, pixels per micrometre); the longer side is long_side_px."""
    half_width, half_height = slate_extent(slate)
    margin = MARGIN_FRACTION * long_side_px
    legend = legend_height_px(long_side_px, class_count, long_side_px)
    by_width = (long_side_px - 2 * margin) / (2 * half_width)
    by_height = (long_side_px - 2 * margin - legend) / (2 * half_height)
    scale = min(by_width, by_height)
    width = round(2 * half_width * scale + 2 * margin)
    height = round(2 * half_height * scale + 2 * margin + legend)
    if by_width <= by_height:
        width = long_side_px
    else:
        height = long_side_px
    return max(width, 1), max(height, 1), legend, margin, scale


def make_slate_figure(slate, long_side_px, class_count):
    """Return (a Figure whose PNG long side is long_side_px, its one axes showing exactly the slate)."""
    width, height, legend, margin, scale = picture_layout(slate, long_side_px, class_count)
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
    """Add a legend listing every class, in sorted order, along the bottom of the figure."""
    handles = [Patch(facecolor=colour, label=label) for label, colour in colours.items()]
    width_px = figure.get_figwidth() * figure.dpi
    figure.legend(handles=handles, loc="lower center", ncol=legend_columns(int(width_px), len(handles)), frameon=False)
