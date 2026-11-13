# This file draws the dashboard of the time-lapse video: the slate on the left
# and four graphs beside it, a global size histogram, a size histogram by
# class, a line graph over steps and a bar chart by size bin. Every number on
# every panel is worked out in advance for all shown steps by panel_data, from
# the saved rows alone, so the numbers can be checked without drawing and the
# fixed axis limits are known before the first frame. build_dashboard_figure
# then draws one step from those numbers only. Axis limits never change between
# steps. The histograms and the bar chart share one set of size bins from
# size_bin_edges. The global histogram fills the particles placed so far, all
# classes together, inside the outline of all particles. The class histogram
# draws the placed particles of all classes together as a very thick black
# line and each class's placed particles as a thick line in its slate colour
# on top of it, all from zero, so no class hides another. The bar chart puts a black bar for
# all classes together and one narrow bar per class side by side. What the
# bars measure depends on the run: the mean location tries per placed
# particle for a random run with several tries, the share not placed for a
# random run with one try, and the mean distance of the placed particles from
# the slate centre for a centre run, whose particles always take one search. The line
# graph has two colour-coded y axes: the packing factor (slate coverage) on
# the left, and the share placed within the run's tries on the right, scaled from just below its lowest value to 100 %. A step is one
# particle attempted: the next particle in the saved order, which ends either
# placed or, after all its location tries were rejected, not placed. The
# location tries inside one attempt are not steps.

import math
from dataclasses import dataclass

import numpy
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from placement_simulation import geometry, placement
from placement_visualization import slate_drawing

BIN_COUNT = 20
PERCENT_MAX = 100.0
FIGURE_DPI = 100
COVERAGE_STEP = 10.0
SHARE_STEP = 5.0
SHARE_HEADROOM_FRACTION = 0.05
COVERAGE_COLOUR = "#0047ab"
COVERAGE_STYLE = "-"
SHARE_COLOUR = "#c00000"
SHARE_STYLE = "--"
LINE_WIDTH = 3.0
SLATE_OUTLINE_WIDTH = 3.0
ALL_OUTLINE_COLOUR = "black"
ALL_OUTLINE_WIDTH = 1.5
PLACED_FILL_COLOUR = "dimgray"
POPULATION_LINE_COLOUR = "black"
POPULATION_LINE_WIDTH = 7.0
CLASS_LINE_WIDTH = 4.0
CLASS_HISTOGRAM_HEADROOM = 1.05
AGGREGATE_BAR_COLOUR = "black"
STEP_LABEL = "Particles attempted (placed or not)"
LEGEND_ROOM = 0.1
TRIES = "tries"
UNPLACED = "unplaced"
DISTANCE = "distance"
BAR_LABELS = {
    TRIES: ("Mean location tries per placed particle", "Mean location tries by size (black: all classes)"),
    UNPLACED: ("Not placed (% of attempted)", "Share not placed by size (black: all classes)"),
    DISTANCE: ("Mean distance from slate centre (um)", "Mean distance from centre by size (black: all classes)"),
}


@dataclass(frozen=True)
class PanelData:
    """Every panel number: the bins, the classes in sorted order, the running line values for all steps, the counts of all particles and, per shown step, the placed counts and the bar values, each as one row per class (NaN: no bar), with the all-class bar values and every fixed axis limit."""

    edges: object
    classes: tuple
    steps: tuple
    coverage_percent: tuple
    share_percent: tuple
    outline_counts: object
    filled_counts: dict
    bar_values: dict
    aggregate_bar_values: dict
    bar_measure: str
    histogram_max: float
    class_histogram_max: float
    bar_max: float
    coverage_max: float
    share_min: float
    share_top: float
    share_label: str


def size_bin_edges(diameters):
    """Return the size bin edges shared by the histogram and the bar chart: numpy.histogram_bin_edges of all diameters with 20 bins."""
    return numpy.histogram_bin_edges(numpy.asarray(diameters, dtype=float), bins=BIN_COUNT)


def running_lines(result):
    """Return (slate coverage %, share placed %) for steps 1 to N: placed true area over the slate area, and placed so far over attempted so far."""
    slate_area = geometry.slate_area(result.slate)
    area = 0.0
    placed_count = 0
    coverage = []
    share = []
    for attempted, row in enumerate(result.rows, start=1):
        if row.placed:
            area += placement.particle_area(row.equivalent_diameter_um)
            placed_count += 1
        coverage.append(100.0 * area / slate_area)
        share.append(100.0 * placed_count / attempted)
    return tuple(coverage), tuple(share)


def bin_counts(values, edges, weights=None):
    """Return the histogram counts (or weight sums) of values in the shared bins."""
    counts, _ = numpy.histogram(values, bins=edges, weights=weights)
    return counts


def bar_measure(result):
    """Return what the bars of a run measure: the distance from the centre for a centre run, the share not placed for a random run with one try, and the tries otherwise."""
    if result.mode == "centre":
        return DISTANCE
    if result.max_tries == 1:
        return UNPLACED
    return TRIES


def placed_measure(row, measure):
    """Return the value one placed row adds to its bin's mean: its saved tries, or its distance from the slate centre."""
    if measure == DISTANCE:
        return math.hypot(row.x_um, row.y_um)
    return row.tries


def bar_values_for_step(rows, edges, measure):
    """Return one value per bin for the first rows attempted: the % of attempted particles not placed, or the mean over the placed particles of their tries or their distance from the slate centre; NaN when the bin has nothing to average."""
    attempted = diameter_counts(rows, edges)
    placed_rows = [row for row in rows if row.placed]
    placed = diameter_counts(placed_rows, edges)
    values = numpy.full(len(attempted), numpy.nan)
    if measure == UNPLACED:
        has_attempted = attempted > 0
        values[has_attempted] = 100.0 * (attempted[has_attempted] - placed[has_attempted]) / attempted[has_attempted]
        return values
    sums = bin_counts([row.equivalent_diameter_um for row in placed_rows], edges, [placed_measure(row, measure) for row in placed_rows])
    has_placed = placed > 0
    values[has_placed] = sums[has_placed] / placed[has_placed]
    return values


def coverage_axis_max(coverage):
    """Return the top of the coverage axis: the final coverage rounded up to the next 10 %, at least 10 and at most 100."""
    final = coverage[-1] if coverage else 0.0
    return min(PERCENT_MAX, max(COVERAGE_STEP, COVERAGE_STEP * math.ceil(final / COVERAGE_STEP)))


def share_axis_limits(share):
    """Return (bottom, top) of the share-placed axis: the bottom is the lowest share rounded down to a multiple of 5 %, a further 5 % lower when the share reaches it, never below 0; the top leaves 5 % of the span above 100 % so a line at 100 % stays clear of the frame."""
    lowest = min(share) if share else PERCENT_MAX
    bottom = SHARE_STEP * math.floor(lowest / SHARE_STEP)
    if bottom >= lowest:
        bottom -= SHARE_STEP
    bottom = max(0.0, bottom)
    return bottom, PERCENT_MAX + SHARE_HEADROOM_FRACTION * (PERCENT_MAX - bottom)


def share_axis_label(mode, max_tries):
    """Return the share axis label: in random mode it names the run's tries per particle, for example "Placed within 20 tries (% of attempted)"; centre placement makes one search per particle, so it says "Placed (% of attempted)"."""
    if mode == "centre":
        return "Placed (% of attempted)"
    tries = "try" if max_tries == 1 else "tries"
    return f"Placed within {max_tries} {tries} (% of attempted)"


def rows_of_class(rows, label):
    """Return the rows whose class is label, in their saved order."""
    return [row for row in rows if row.feature_label == label]


def diameter_counts(rows, edges):
    """Return the counts of the rows' diameters in the shared bins."""
    return bin_counts([row.equivalent_diameter_um for row in rows], edges)


def placed_diameter_counts(rows, edges):
    """Return the counts of the placed rows' diameters in the shared bins."""
    return diameter_counts([row for row in rows if row.placed], edges)


def panel_data(result, steps):
    """Return the PanelData of a run for the shown steps, with every fixed axis maximum computed over all of them."""
    rows = result.rows
    classes = tuple(sorted({row.feature_label for row in rows}))
    edges = size_bin_edges([row.equivalent_diameter_um for row in rows])
    measure = bar_measure(result)
    coverage, share = running_lines(result)
    outline = numpy.array([diameter_counts(rows_of_class(rows, label), edges) for label in classes])
    filled = {}
    bars = {}
    aggregate = {}
    largest = 1.0
    for step in steps:
        attempted_rows = rows[:step]
        by_class = [rows_of_class(attempted_rows, label) for label in classes]
        filled[step] = numpy.array([placed_diameter_counts(class_rows, edges) for class_rows in by_class])
        bars[step] = numpy.array([bar_values_for_step(class_rows, edges, measure) for class_rows in by_class])
        aggregate[step] = bar_values_for_step(attempted_rows, edges, measure)
        for values in (bars[step], aggregate[step]):
            if not numpy.all(numpy.isnan(values)):
                largest = max(largest, float(numpy.nanmax(values)))
    bar_max = PERCENT_MAX if measure == UNPLACED else largest
    share_min, share_top = share_axis_limits(share)
    return PanelData(
        edges=edges,
        classes=classes,
        steps=tuple(steps),
        coverage_percent=coverage,
        share_percent=share,
        outline_counts=outline,
        filled_counts=filled,
        bar_values=bars,
        aggregate_bar_values=aggregate,
        bar_measure=measure,
        histogram_max=float(outline.sum(axis=0).max()),
        class_histogram_max=max(1.0, max((float(counts.sum(axis=0).max()) for counts in filled.values()), default=0.0)),
        bar_max=bar_max,
        coverage_max=coverage_axis_max(coverage),
        share_min=share_min,
        share_top=share_top,
        share_label=share_axis_label(result.mode, result.max_tries),
    )


def draw_global_histogram(axes, data, step):
    """Draw the size distribution of all classes together: the particles placed among the first step attempted, filled, inside the outline of all particles."""
    axes.stairs(data.filled_counts[step].sum(axis=0), data.edges, fill=True, color=PLACED_FILL_COLOUR)
    axes.stairs(data.outline_counts.sum(axis=0), data.edges, color=ALL_OUTLINE_COLOUR, linewidth=ALL_OUTLINE_WIDTH)
    axes.set_xlim(data.edges[0], data.edges[-1])
    axes.set_ylim(0, data.histogram_max)
    axes.set_xlabel("Diameter (um)")
    axes.set_ylabel("Particles")
    axes.set_title("Size histogram, all classes (filled: placed so far, outline: all)")


def draw_class_histogram(axes, data, step, colours):
    """Draw the size distribution of the particles placed among the first step attempted: all classes together as a thick black line, then each class as a thinner line in its slate colour on top, every line from zero on the shared bins, with 5 % room above the highest count so the thick lines are not cut by the frame."""
    axes.stairs(data.filled_counts[step].sum(axis=0), data.edges, color=POPULATION_LINE_COLOUR, linewidth=POPULATION_LINE_WIDTH)
    for label, placed_counts in zip(data.classes, data.filled_counts[step]):
        axes.stairs(placed_counts, data.edges, color=colours[label], linewidth=CLASS_LINE_WIDTH)
    axes.set_xlim(data.edges[0], data.edges[-1])
    axes.set_ylim(0, CLASS_HISTOGRAM_HEADROOM * data.class_histogram_max)
    axes.set_xlabel("Diameter (um)")
    axes.set_ylabel("Particles placed")
    axes.set_title("Size histogram by class, placed so far (black: all classes)")


def draw_lines(axes, data, step):
    """Draw the coverage line on the left axis (0 to coverage_max) and the share-placed line on a right axis (share_min to share_top), over steps 1 to step; return the right axes."""
    steps = list(range(1, step + 1))
    axes.plot(steps, data.coverage_percent[:step], color=COVERAGE_COLOUR, linestyle=COVERAGE_STYLE, linewidth=LINE_WIDTH)
    axes.set_xlim(0, len(data.coverage_percent))
    axes.set_ylim(0, data.coverage_max)
    axes.set_xlabel(STEP_LABEL, fontweight="bold")
    axes.set_ylabel("Packing Factor (%)", color=COVERAGE_COLOUR, fontweight="bold")
    axes.tick_params(axis="y", colors=COVERAGE_COLOUR)
    axes.set_title("Progress", fontweight="bold")
    share_axes = axes.twinx()
    share_axes.plot(steps, data.share_percent[:step], color=SHARE_COLOUR, linestyle=SHARE_STYLE, linewidth=LINE_WIDTH)
    share_axes.set_ylim(data.share_min, data.share_top)
    share_axes.set_ylabel(data.share_label, color=SHARE_COLOUR, fontweight="bold")
    share_axes.tick_params(axis="y", colors=SHARE_COLOUR)
    for tick_label in axes.get_xticklabels() + axes.get_yticklabels() + share_axes.get_yticklabels():
        tick_label.set_fontweight("bold")
    return share_axes


def draw_bars(axes, data, step, colours):
    """Draw the bar chart by size bin: in every bin a black bar for all classes together, then one narrow bar per class in sorted class order, side by side, leaving out every bar that has no value, on an axis whose maximum was fixed in advance."""
    series = [(AGGREGATE_BAR_COLOUR, data.aggregate_bar_values[step])]
    series += [(colours[label], values) for label, values in zip(data.classes, data.bar_values[step])]
    widths = numpy.diff(data.edges) / len(series)
    lefts = data.edges[:-1]
    for position, (colour, values) in enumerate(series):
        shown = ~numpy.isnan(values)
        axes.bar(lefts[shown] + position * widths[shown], values[shown], width=widths[shown], align="edge", color=colour)
    axes.set_xlim(data.edges[0], data.edges[-1])
    axes.set_ylim(0, data.bar_max)
    axes.set_xlabel("Diameter (um)")
    ylabel, title = BAR_LABELS[data.bar_measure]
    axes.set_ylabel(ylabel)
    axes.set_title(title)


def build_dashboard_figure(result, data, step, width_px=1920, height_px=1080):
    """Return the figure for one shown step, drawn from data only: the slate on the left, over both rows, and beside it the global and class histograms on the top row and the line graph and bar chart on the bottom row.

    figure.axes holds the slate, global histogram, class histogram, line graph
    and bar chart in that order, then the line graph's right (share placed) axis.
    """
    colours = slate_drawing.class_colours(result.rows)
    figure = Figure(figsize=(width_px / FIGURE_DPI, height_px / FIGURE_DPI), dpi=FIGURE_DPI, facecolor=slate_drawing.BACKGROUND)
    FigureCanvasAgg(figure)
    grid = figure.add_gridspec(2, 3, bottom=LEGEND_ROOM + 0.05, top=0.95, left=0.03, right=0.96, hspace=0.35, wspace=0.3, width_ratios=[1.3, 1.0, 1.0])
    slate_axes = figure.add_subplot(grid[:, 0])
    global_axes = figure.add_subplot(grid[0, 1])
    class_axes = figure.add_subplot(grid[0, 2])
    line_axes = figure.add_subplot(grid[1, 1])
    bar_axes = figure.add_subplot(grid[1, 2])
    half_width, half_height = slate_drawing.slate_extent(result.slate)
    slate_axes.set_xlim(-half_width, half_width)
    slate_axes.set_ylim(-half_height, half_height)
    slate_axes.set_aspect("equal")
    slate_axes.set_axis_off()
    slate_drawing.draw_slate(slate_axes, result.slate, result.rows[:step], colours, SLATE_OUTLINE_WIDTH)
    draw_global_histogram(global_axes, data, step)
    draw_class_histogram(class_axes, data, step, colours)
    draw_lines(line_axes, data, step)
    draw_bars(bar_axes, data, step, colours)
    slate_drawing.draw_legend(figure, colours, LEGEND_ROOM * height_px)
    return figure
