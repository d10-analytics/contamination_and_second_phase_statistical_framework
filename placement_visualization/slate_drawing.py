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
MIN_SLATE_PX = 10
# Pixel boxes measured by matplotlib carry rounding noise, so a legend that lies
# inside the picture can measure a tiny fraction of a pixel outside it; every
# inside check allows this much. ROUNDING_PX absorbs noise in comparisons of two
# lengths that should be equal.
PIXEL_TOLERANCE_PX = 0.5
ROUNDING_PX = 1e-6
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
        fits = fits_width(box, width_px)
        if low >= high:
            return legend, box
        legend.remove()
        if fits:
            low = columns
        else:
            high = columns - 1


def add_fitted_legend(figure, handles, max_height_px=None):
    """Add a legend along the bottom of the figure and return it.

    Columns are as many as fit across the figure, giving the fewest rows; the
    font shrinks until the legend also fits the figure width and the given
    height (when there is one), down to a minimum size. At that minimum the
    legend is returned as it is and may be wider or taller than allowed, so a
    caller that needs it inside the figure checks its extent with legend_inside.
    """
    width_px = figure.get_figwidth() * figure.dpi
    size = LEGEND_MAX_FONT_POINTS
    while True:
        legend, box = add_legend_columns(figure, handles, size)
        ratio = 1.0
        if not fits_width(box, width_px):
            ratio = min(ratio, width_px / box.width)
        if max_height_px is not None and box.y1 > max_height_px + ROUNDING_PX:
            ratio = min(ratio, max_height_px / box.y1)
        if ratio >= 1.0 or size <= LEGEND_MIN_FONT_POINTS:
            return legend
        legend.remove()
        size = max(LEGEND_MIN_FONT_POINTS, math.floor(size * ratio * 10.0) / 10.0 - LEGEND_FONT_MARGIN_POINTS)


def fits_width(box, width_px):
    """Say whether a legend's pixel box lies within the width, up to PIXEL_TOLERANCE_PX."""
    return box.x0 >= -PIXEL_TOLERANCE_PX and box.x1 <= width_px + PIXEL_TOLERANCE_PX


def legend_inside(box, width_px):
    """Say whether a legend's pixel box lies within the picture width, starting at the picture bottom, up to PIXEL_TOLERANCE_PX."""
    return fits_width(box, width_px) and box.y0 >= -PIXEL_TOLERANCE_PX


def legend_box(width_px, labels, long_side_px):
    """Return the pixel box the legend of these labels measures, in a picture width_px wide whose long side is long_side_px."""
    cap = LEGEND_MAX_FRACTION * long_side_px
    scratch = Figure(figsize=(width_px / FIGURE_DPI, max(cap, 1.0) / FIGURE_DPI), dpi=FIGURE_DPI)
    FigureCanvasAgg(scratch)
    legend = add_fitted_legend(scratch, legend_handles(labels), cap)
    return legend.get_window_extent(scratch.canvas.get_renderer())


def try_layout(slate, long_side_px, labels=None):
    """Return the picture layout, or None when the legend of the labels cannot lie wholly inside the picture under a usable slate."""
    half_width, half_height = slate_extent(slate)
    margin = MARGIN_FRACTION * long_side_px
    if labels is None:
        strip = min(LEGEND_ROW_PX, LEGEND_MAX_FRACTION * long_side_px)
        width, height, scale = fit_picture(half_width, half_height, long_side_px, margin, strip)
        return width, height, strip, margin, scale
    box = legend_box(long_side_px, labels, long_side_px)
    box_width = long_side_px
    strip = box.y1
    while True:
        if not legend_inside(box, box_width):
            return None
        width, height, scale = fit_picture(half_width, half_height, long_side_px, margin, strip)
        if 2 * max(half_width, half_height) * scale < MIN_SLATE_PX - ROUNDING_PX:
            return None
        box = legend_box(width, labels, long_side_px)
        box_width = width
        if legend_inside(box, width) and box.y1 <= strip + ROUNDING_PX:
            return width, height, strip, margin, scale
        strip = box.y1


def workable_size(slate, labels, refused_px):
    """Return a long side larger than refused_px for which the labels' legend fits, close to the smallest: found by doubling, then halving the gap to 10 percent."""
    low, high = refused_px, refused_px * 2
    while try_layout(slate, high, labels) is None:
        low, high = high, high * 2
    while high - low > max(1, high // 10):
        middle = (low + high) // 2
        if try_layout(slate, middle, labels) is None:
            low = middle
        else:
            high = middle
    return high


def picture_layout(slate, long_side_px, labels=None):
    """Return (width px, height px, strip px, margin px, pixels per micrometre); the longer side is long_side_px.

    The strip under the slate holds the legend of the labels as measured at the
    picture's own width, or a single text row when labels is None. A ValueError
    naming long_side_px is raised, with a size that works, when the legend
    cannot lie wholly inside the picture at the minimum font with a slate at
    least MIN_SLATE_PX pixels along its longer side left above it.
    """
    layout = try_layout(slate, long_side_px, labels)
    if layout is None:
        working = workable_size(slate, labels, long_side_px)
        raise ValueError(
            f"long_side_px={long_side_px} is too small to draw a legend of {len(labels)} classes inside the picture "
            f"with a slate at least {MIN_SLATE_PX} pixels long; long_side_px={working} works"
        )
    return layout


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


def draw_legend(figure, colours, max_height_px=None):
    """Add a legend listing every class, in sorted order, along the bottom of the figure, fitted by add_fitted_legend to the width and to max_height_px (a fraction of the long side when not given)."""
    handles = [Patch(facecolor=colour, label=label) for label, colour in colours.items()]
    if max_height_px is None:
        long_side_px = max(figure.get_figwidth(), figure.get_figheight()) * figure.dpi
        max_height_px = LEGEND_MAX_FRACTION * long_side_px
    return add_fitted_legend(figure, handles, max_height_px)
