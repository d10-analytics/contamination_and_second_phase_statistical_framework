# This file draws the dashboard of the time-lapse video: a 2 x 2 grid of the
# slate, a size histogram, a line graph over steps and a bar chart by size
# bin. Every number on every panel is worked out in advance for all shown
# steps by panel_data, from the saved rows alone, so the numbers can be checked
# without drawing and the fixed axis maxima are known before the first frame.
# build_dashboard_figure then draws one step from those numbers only. Axis
# limits never change between steps. The histogram and the bar chart share one
# set of size bins from size_bin_edges.

from dataclasses import dataclass

import numpy
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from placement_simulation import geometry, placement
from placement_visualization import slate_drawing

BIN_COUNT = 20
PERCENT_MAX = 100.0
FIGURE_DPI = 100
OUTLINE_COLOUR = "black"
FILL_COLOUR = "tab:blue"
BAR_COLOUR = "tab:orange"
COVERAGE_COLOUR = "tab:green"
SHARE_COLOUR = "tab:red"
LEGEND_ROOM = 0.1


@dataclass(frozen=True)
class PanelData:
    """Every panel number: the bins, the running line values for all steps and, per shown step, the filled counts and the bar values (NaN: no bar)."""

    edges: object
    steps: tuple
    coverage_percent: tuple
    share_percent: tuple
    outline_counts: object
    filled_counts: dict
    bar_values: dict
    bar_is_unplaced_share: bool
    histogram_max: float
    bar_max: float


def size_bin_edges(diameters):
    """Return the size bin edges shared by the histogram and the bar chart: numpy.histogram_bin_edges of all diameters with 20 bins."""
    return numpy.histogram_bin_edges(numpy.asarray(diameters, dtype=float), bins=BIN_COUNT)


def running_lines(result):
    """Return (slate coverage %, share placed %) for steps 1 to N: placed true area over the slate area, and placed so far over taken so far."""
    slate_area = geometry.slate_area(result.slate)
    area = 0.0
    placed_count = 0
    coverage = []
    share = []
    for taken, row in enumerate(result.rows, start=1):
        if row.placed:
            area += placement.particle_area(row.equivalent_diameter_um)
            placed_count += 1
        coverage.append(100.0 * area / slate_area)
        share.append(100.0 * placed_count / taken)
    return tuple(coverage), tuple(share)


def bin_counts(values, edges, weights=None):
    """Return the histogram counts (or weight sums) of values in the shared bins."""
    counts, _ = numpy.histogram(values, bins=edges, weights=weights)
    return counts


def bar_values_for_step(rows, edges, unplaced_share):
    """Return one value per bin for the first rows taken: the mean saved tries of the placed particles, or the % of taken particles not placed; NaN when the bin has nothing to average."""
    taken = bin_counts([row.equivalent_diameter_um for row in rows], edges)
    placed_rows = [row for row in rows if row.placed]
    placed = bin_counts([row.equivalent_diameter_um for row in placed_rows], edges)
    values = numpy.full(len(taken), numpy.nan)
    if unplaced_share:
        has_taken = taken > 0
        values[has_taken] = 100.0 * (taken[has_taken] - placed[has_taken]) / taken[has_taken]
        return values
    tries = bin_counts([row.equivalent_diameter_um for row in placed_rows], edges, [row.tries for row in placed_rows])
    has_placed = placed > 0
    values[has_placed] = tries[has_placed] / placed[has_placed]
    return values


def panel_data(result, steps):
    """Return the PanelData of a run for the shown steps, with both fixed axis maxima computed over all of them."""
    rows = result.rows
    edges = size_bin_edges([row.equivalent_diameter_um for row in rows])
    unplaced_share = result.max_tries == 1 or result.mode == "centre"
    coverage, share = running_lines(result)
    outline = bin_counts([row.equivalent_diameter_um for row in rows], edges)
    filled = {}
    bars = {}
    largest = 1.0
    for step in steps:
        taken_rows = rows[:step]
        placed_diameters = [row.equivalent_diameter_um for row in taken_rows if row.placed]
        filled[step] = bin_counts(placed_diameters, edges)
        bars[step] = bar_values_for_step(taken_rows, edges, unplaced_share)
        if not numpy.all(numpy.isnan(bars[step])):
            largest = max(largest, float(numpy.nanmax(bars[step])))
    bar_max = PERCENT_MAX if unplaced_share else largest
    return PanelData(
        edges=edges,
        steps=tuple(steps),
        coverage_percent=coverage,
        share_percent=share,
        outline_counts=outline,
        filled_counts=filled,
        bar_values=bars,
        bar_is_unplaced_share=unplaced_share,
        histogram_max=float(outline.max()),
        bar_max=bar_max,
    )


def draw_histogram(axes, data, step):
    """Draw the histogram: the outline of all particles and, filled, those placed among the first step taken."""
    widths = numpy.diff(data.edges)
    lefts = data.edges[:-1]
    axes.bar(lefts, data.outline_counts, width=widths, align="edge", fill=False, edgecolor=OUTLINE_COLOUR)
    axes.bar(lefts, data.filled_counts[step], width=widths, align="edge", color=FILL_COLOUR)
    axes.set_xlim(data.edges[0], data.edges[-1])
    axes.set_ylim(0, data.histogram_max)
    axes.set_xlabel("Diameter (um)")
    axes.set_ylabel("Particles")
    axes.set_title("Size histogram")


def draw_lines(axes, data, step):
    """Draw the coverage and share-placed lines over steps 1 to step, on axes fixed to 0 to N and 0 to 100."""
    steps = list(range(1, step + 1))
    axes.plot(steps, data.coverage_percent[:step], color=COVERAGE_COLOUR, label="Slate coverage (%)")
    axes.plot(steps, data.share_percent[:step], color=SHARE_COLOUR, label="Placed of taken (%)")
    axes.set_xlim(0, len(data.coverage_percent))
    axes.set_ylim(0, PERCENT_MAX)
    axes.set_xlabel("Particles taken")
    axes.set_ylabel("%")
    axes.legend(loc="lower left", frameon=False)
    axes.set_title("Progress")


def draw_bars(axes, data, step):
    """Draw the bar chart by size bin, leaving out every bin that has no value, on an axis whose maximum was fixed in advance."""
    values = data.bar_values[step]
    shown = ~numpy.isnan(values)
    widths = numpy.diff(data.edges)
    axes.bar(data.edges[:-1][shown], values[shown], width=widths[shown], align="edge", color=BAR_COLOUR)
    axes.set_xlim(data.edges[0], data.edges[-1])
    axes.set_ylim(0, data.bar_max)
    axes.set_xlabel("Diameter (um)")
    if data.bar_is_unplaced_share:
        axes.set_ylabel("Not placed (% of taken)")
        axes.set_title("Share not placed by size")
    else:
        axes.set_ylabel("Mean tries")
        axes.set_title("Mean tries by size")


def build_dashboard_figure(result, data, step, width_px=1920, height_px=1080):
    """Return the figure for one shown step: the slate, histogram, line graph and bar chart in a 2 x 2 grid, drawn from data only."""
    colours = slate_drawing.class_colours(result.rows)
    figure = Figure(figsize=(width_px / FIGURE_DPI, height_px / FIGURE_DPI), dpi=FIGURE_DPI, facecolor=slate_drawing.BACKGROUND)
    FigureCanvasAgg(figure)
    grid = figure.add_gridspec(2, 2, bottom=LEGEND_ROOM + 0.05, top=0.95, hspace=0.35, wspace=0.25)
    slate_axes = figure.add_subplot(grid[0, 0])
    half_width, half_height = slate_drawing.slate_extent(result.slate)
    slate_axes.set_xlim(-half_width, half_width)
    slate_axes.set_ylim(-half_height, half_height)
    slate_axes.set_aspect("equal")
    slate_axes.set_axis_off()
    slate_drawing.draw_slate(slate_axes, result.slate, result.rows[:step], colours)
    draw_histogram(figure.add_subplot(grid[0, 1]), data, step)
    draw_lines(figure.add_subplot(grid[1, 0]), data, step)
    draw_bars(figure.add_subplot(grid[1, 1]), data, step)
    slate_drawing.draw_legend(figure, colours, LEGEND_ROOM * height_px)
    return figure
