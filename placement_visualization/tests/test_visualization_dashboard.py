# This file checks the dashboard of the video. The panel numbers are computed
# without drawing from hand-built rows: random placement with several tries,
# random with one try and centre placement (whose bars are the distance from
# the slate centre), placed and unplaced particles, an
# empty size bin and an early particle with many tries. It checks the running
# coverage (ending at the achieved coverage), the share placed out of those
# attempted, the histogram counts, the bar values and every fixed axis limit,
# and that a replacement size-bin function reaches every size panel.
# Three-class runs check the size panels: the global histogram fills the
# placed particles of all classes inside the outline of all particles, the
# class histogram draws all classes' placed particles as a solid black line and
# each class's as a solid line in its slate colour, all from zero, and the bar chart puts a black all-class bar and
# one bar per class side by side. Figures for the first and last shown steps
# are then drawn and checked for identical axis limits, outline and bar
# heights and line data equal to the numbers, the two colour-coded, bold y axes
# of the line graph with no legend on it and the share axis running from just
# below its lowest value to 100 %, and for the slate-drawing function being the
# one called with a thick outline.

import dataclasses
import math

import numpy
import pytest

from placement_simulation import placement
from placement_simulation.placement import PlacementResult, PlacementRow
from placement_visualization import dashboard, slate_drawing

SQUARE = {"shape": "square", "side_um": 100.0}
INPUT = {"kind": "generator", "measurements_sha256": "0" * 64}


class MarkerError(Exception):
    """Raised by a replacement so a test can tell the replacement was reached."""


def make_row(order, diameter, placed, tries, label="A"):
    """Return a placement row; an unplaced particle has no position."""
    return PlacementRow(
        placement_order=order,
        source_index=order,
        object_id=f"p{order}",
        feature_label=label,
        equivalent_diameter_um=diameter,
        placed=placed,
        x_um=10.0 if placed else None,
        y_um=10.0 if placed else None,
        tries=tries,
        edge_rejections=0,
        overlap_rejections=0,
    )


def make_result(rows, max_tries=5, mode="random"):
    """Return a finished run of the rows on the square slate."""
    return PlacementResult(
        slate=SQUARE,
        mode=mode,
        max_tries=max_tries,
        buffer_um=1.0,
        pixel_um=1.0,
        order="as_given",
        seed=1,
        input=INPUT,
        requested_coverage=0.0,
        achieved_coverage=placement.achieved_coverage(rows, SQUARE),
        rows=tuple(rows),
    )


def tries_rows():
    """Return six particles of diameters 10 and 30 (first and last bins, a gap between), some unplaced, with distinct tries."""
    return [
        make_row(1, 10.0, True, 9),
        make_row(2, 10.0, True, 1),
        make_row(3, 30.0, False, 5),
        make_row(4, 30.0, True, 2),
        make_row(5, 10.0, False, 5),
        make_row(6, 30.0, True, 3),
    ]


def test_bin_edges_are_numpy_histogram_bin_edges_of_all_diameters():
    """The shared bins are twenty numpy bins over every diameter, placed or not."""
    diameters = [10.0, 10.0, 30.0, 30.0, 10.0, 30.0]
    assert numpy.array_equal(dashboard.size_bin_edges(diameters), numpy.histogram_bin_edges(diameters, bins=20))
    assert len(dashboard.size_bin_edges(diameters)) == 21


def test_line_values_use_true_area_and_share_of_taken():
    """Coverage is the placed true area over the slate; share is placed over taken so far."""
    rows = tries_rows()
    data = dashboard.panel_data(make_result(rows), [3, 6])
    area = math.pi * 100.0 / 4.0
    assert data.coverage_percent[0] == pytest.approx(100.0 * area / 10000.0)
    assert data.coverage_percent[1] == pytest.approx(200.0 * area / 10000.0)
    assert data.coverage_percent[2] == data.coverage_percent[1]
    assert list(data.share_percent[:3]) == pytest.approx([100.0, 100.0, 200.0 / 3.0])
    assert data.share_percent[5] == pytest.approx(100.0 * 4 / 6)
    assert data.share_percent[5] < data.share_percent[1]


def test_final_coverage_equals_achieved_coverage():
    """The coverage at step N is 100 x the achieved coverage."""
    result = make_result(tries_rows())
    data = dashboard.panel_data(result, [6])
    assert data.coverage_percent[-1] == pytest.approx(100.0 * result.achieved_coverage, rel=1e-12)


def test_histogram_counts_outline_all_and_filled_placed_among_first_steps():
    """The outline counts every particle of the class in the file; the filled counts those placed among the first k."""
    data = dashboard.panel_data(make_result(tries_rows()), [2, 6])
    assert data.outline_counts[0][0] == 3
    assert data.outline_counts[0][-1] == 3
    assert data.outline_counts.sum() == 6
    assert data.classes == ("A",)
    assert list(data.filled_counts[2][0][[0, -1]]) == [2, 0]
    assert list(data.filled_counts[6][0][[0, -1]]) == [2, 2]
    assert data.histogram_max == 3.0


def test_mean_tries_per_bin_and_empty_bins_have_no_bar():
    """Mean tries averages the saved tries of placed particles; a bin with none placed, or empty, is NaN."""
    data = dashboard.panel_data(make_result(tries_rows()), [3, 6])
    assert data.bar_measure == dashboard.TRIES
    first = data.bar_values[3][0]
    assert first[0] == pytest.approx(5.0)
    assert numpy.isnan(first[-1])
    assert numpy.all(numpy.isnan(first[1:-1]))
    last = data.bar_values[6][0]
    assert last[0] == pytest.approx(5.0)
    assert last[-1] == pytest.approx(2.5)
    assert numpy.all(numpy.isnan(last[1:-1]))


def test_mean_tries_maximum_covers_an_early_large_mean_that_falls():
    """The axis maximum is over all shown steps, so it exceeds the final-state maximum."""
    rows = [make_row(1, 10.0, True, 9)] + [make_row(number, 10.0, True, 1) for number in range(2, 6)] + [make_row(6, 30.0, True, 2)]
    data = dashboard.panel_data(make_result(rows), [1, 5, 6])
    assert data.bar_values[1][0][0] == pytest.approx(9.0)
    assert data.bar_values[6][0][0] == pytest.approx(13.0 / 5.0)
    assert data.bar_max == pytest.approx(9.0)
    assert data.bar_max > float(numpy.nanmax(data.bar_values[6]))


def test_mean_tries_maximum_is_one_when_no_bin_has_a_value():
    """With nothing placed in any shown step the maximum is 1."""
    rows = [make_row(1, 10.0, False, 5), make_row(2, 30.0, False, 5)]
    data = dashboard.panel_data(make_result(rows), [1, 2])
    assert numpy.all(numpy.isnan(data.bar_values[2]))
    assert data.bar_max == 1.0


def test_one_try_run_shows_percent_of_taken_not_placed():
    """With max_tries of 1 the bars are the % of taken particles in the bin not placed; a bin with none taken has no bar."""
    data = dashboard.panel_data(make_result(tries_rows(), max_tries=1), [3, 6])
    assert data.bar_measure == dashboard.UNPLACED
    assert data.bar_values[3][0][0] == pytest.approx(0.0)
    assert data.bar_values[3][0][-1] == pytest.approx(100.0)
    assert data.bar_values[6][0][0] == pytest.approx(100.0 / 3.0)
    assert data.bar_values[6][0][-1] == pytest.approx(100.0 / 3.0)
    assert numpy.all(numpy.isnan(data.bar_values[6][0][1:-1]))
    assert data.bar_max == 100.0


def centre_rows():
    """Return four placed particles at known distances from the slate centre, two in the first bin (3-4-5 and 6-8-10 triangles) and two in the last, one in each class, and one unplaced particle."""
    def placed(order, diameter, x, y, label):
        """Return a placed row at (x, y)."""
        row = make_row(order, diameter, True, 1, label=label)
        return dataclasses.replace(row, x_um=x, y_um=y)

    return [
        placed(1, 10.0, 3.0, 4.0, "A"),
        placed(2, 10.0, -6.0, 8.0, "B"),
        placed(3, 30.0, 0.0, -20.0, "A"),
        placed(4, 30.0, 12.0, 0.0, "B"),
        make_row(5, 30.0, False, 1, label="B"),
    ]


def test_centre_run_shows_mean_distance_from_the_slate_centre():
    """Centre placement, which saves one try per particle, shows each bin's mean distance of its placed particles from the slate centre, per class and for all classes, ignoring the unplaced one."""
    data = dashboard.panel_data(make_result(centre_rows(), max_tries=1, mode="centre"), [5])
    assert data.bar_measure == dashboard.DISTANCE
    assert [values[0] for values in data.bar_values[5]] == pytest.approx([5.0, 10.0])
    assert [values[-1] for values in data.bar_values[5]] == pytest.approx([20.0, 12.0])
    assert data.aggregate_bar_values[5][[0, -1]] == pytest.approx([7.5, 16.0])
    assert data.bar_max == pytest.approx(20.0)
    figure = dashboard.build_dashboard_figure(make_result(centre_rows(), max_tries=1, mode="centre"), data, 5, 640, 360)
    assert figure.axes[4].get_ylabel() == "Mean distance from slate centre (um)"
    assert figure.axes[4].get_ylim() == (0, pytest.approx(20.0))


def test_replacement_bin_edges_are_used_by_both_panels(monkeypatch):
    """A replacement size_bin_edges with distinct edges sets both histograms' and the bar chart's bins."""
    edges = numpy.array([0.0, 20.0, 40.0])

    def replacement_edges(diameters):
        """Return the replacement edges whatever the diameters."""
        return edges

    monkeypatch.setattr(dashboard, "size_bin_edges", replacement_edges)
    result = make_result(tries_rows())
    data = dashboard.panel_data(result, [6])
    assert numpy.array_equal(data.edges, edges)
    assert list(data.outline_counts[0]) == [3, 3]
    assert list(data.bar_values[6][0]) == pytest.approx([5.0, 2.5])
    figure = dashboard.build_dashboard_figure(result, data, 6, 640, 360)
    for histogram in (figure.axes[1], figure.axes[2]):
        assert len(histogram.patches) > 0
        for step_patch in histogram.patches:
            assert list(step_patch.get_data().edges) == [0.0, 20.0, 40.0]
    aggregate, class_bars = figure.axes[4].containers
    assert [patch.get_x() for patch in aggregate] == [0.0, 20.0]
    assert [patch.get_x() for patch in class_bars] == [10.0, 30.0]
    assert [patch.get_width() for patch in class_bars] == [10.0, 10.0]


def three_class_rows():
    """Return nine particles of three classes, every class with one placed particle in the first and last bins and one unplaced, with distinct tries."""
    rows = []
    order = 0
    for label, tries in (("A", 1), ("B", 3), ("C", 5)):
        for diameter, placed, extra in ((10.0, True, 0), (30.0, True, 1), (30.0, False, 0)):
            order += 1
            rows.append(make_row(order, diameter, placed, tries + extra, label=label))
    return rows


def test_three_classes_get_their_own_counts_and_bar_values_and_an_all_class_bar():
    """Each class has its own row of all-particle counts, placed counts and mean tries, in sorted class order; the all-class bar is the mean over every class; the global maximum is the summed count and the class maximum the largest placed count of all classes together."""
    data = dashboard.panel_data(make_result(three_class_rows()), [9])
    assert data.classes == ("A", "B", "C")
    for row in data.outline_counts:
        assert list(row[[0, -1]]) == [1, 2]
    assert data.histogram_max == 6.0
    assert data.class_histogram_max == 3.0
    assert data.filled_counts[9].shape == (3, 20)
    for row in data.filled_counts[9]:
        assert list(row[[0, -1]]) == [1, 1]
    assert [values[0] for values in data.bar_values[9]] == pytest.approx([1.0, 3.0, 5.0])
    assert [values[-1] for values in data.bar_values[9]] == pytest.approx([2.0, 4.0, 6.0])
    assert data.aggregate_bar_values[9][[0, -1]] == pytest.approx([3.0, 4.0])
    assert numpy.all(numpy.isnan(data.aggregate_bar_values[9][1:-1]))
    assert data.bar_max == pytest.approx(6.0)


def test_the_all_class_bar_is_the_share_not_placed_in_a_one_try_run():
    """With one try the black bar is the % of all attempted particles in the bin not placed."""
    data = dashboard.panel_data(make_result(three_class_rows(), max_tries=1), [9])
    assert data.aggregate_bar_values[9][[0, -1]] == pytest.approx([0.0, 50.0])


def step_outlines(axes):
    """Return the step patches of a histogram with a visible line, keyed by (rgb colour, line style, filled)."""
    found = {}
    for step_patch in axes.patches:
        colour = step_patch.get_facecolor() if step_patch.get_fill() else step_patch.get_edgecolor()
        key = (tuple(round(value, 6) for value in colour[:3]), step_patch.get_linestyle(), step_patch.get_fill())
        found[key] = step_patch
    return found


def test_the_global_histogram_fills_all_placed_inside_the_outline_of_all():
    """The global panel has one filled step of the placed counts summed over classes and one black outline of all particles summed over classes."""
    result = make_result(three_class_rows())
    data = dashboard.panel_data(result, [3, 9])
    for step in (3, 9):
        figure = dashboard.build_dashboard_figure(result, data, step, 640, 360)
        axes = figure.axes[1]
        filled = [patch for patch in axes.patches if patch.get_fill()]
        outline = [patch for patch in axes.patches if not patch.get_fill()]
        assert len(filled) == 1 and len(outline) == 1
        assert list(filled[0].get_data().values) == list(data.filled_counts[step].sum(axis=0))
        assert list(outline[0].get_data().values) == list(data.outline_counts.sum(axis=0))
        assert outline[0].get_edgecolor()[:3] == (0.0, 0.0, 0.0)
        assert axes.get_ylim() == (0, data.histogram_max)
    assert list(data.filled_counts[3].sum(axis=0)[[0, -1]]) == [1, 1]
    assert list(data.filled_counts[9].sum(axis=0)[[0, -1]]) == [3, 3]


def test_the_class_histogram_draws_all_classes_in_black_and_each_class_in_its_colour_from_zero():
    """One solid black line is the placed counts of all classes together, drawn first and thicker; each class has one solid line of its placed counts in its slate colour on top; all start from zero."""
    result = make_result(three_class_rows())
    data = dashboard.panel_data(result, [3, 9])
    colours = slate_drawing.class_colours(result.rows)
    for step in (3, 9):
        figure = dashboard.build_dashboard_figure(result, data, step, 640, 360)
        lines = step_outlines(figure.axes[2])
        assert len(lines) == 4
        population = figure.axes[2].patches[0]
        assert lines[((0.0, 0.0, 0.0), "solid", False)] is population
        assert list(population.get_data().values) == list(data.filled_counts[step].sum(axis=0))
        assert population.get_data().baseline == 0
        assert population.get_linewidth() > max(patch.get_linewidth() for patch in figure.axes[2].patches[1:])
        for position, label in enumerate(data.classes):
            colour = tuple(round(value, 6) for value in colours[label])
            placed = lines[(colour, "solid", False)].get_data()
            assert list(placed.values) == list(data.filled_counts[step][position])
            assert placed.baseline == 0
        assert figure.axes[2].get_ylim() == (0, 1.05 * data.class_histogram_max)
    assert list(data.filled_counts[3][0][[0, -1]]) == [1, 1]
    assert list(data.filled_counts[3][2][[0, -1]]) == [0, 0]


def test_the_bar_chart_puts_a_black_all_class_bar_then_one_bar_per_class_side_by_side():
    """In a bin, the black all-class bar comes first and the three class bars follow, each a quarter of the bin wide, without overlap, with their colours and values."""
    result = make_result(three_class_rows())
    data = dashboard.panel_data(result, [9])
    figure = dashboard.build_dashboard_figure(result, data, 9, 640, 360)
    colours = slate_drawing.class_colours(result.rows)
    containers = figure.axes[4].containers
    assert len(containers) == 4
    bin_width = data.edges[1] - data.edges[0]
    expected = [((0.0, 0.0, 0.0), data.aggregate_bar_values[9][0])]
    expected += [(colours[label], data.bar_values[9][position][0]) for position, label in enumerate(data.classes)]
    for position, (container, (colour, value)) in enumerate(zip(containers, expected)):
        first_bar = container.patches[0]
        assert first_bar.get_x() == pytest.approx(data.edges[0] + position * bin_width / 4.0)
        assert first_bar.get_width() == pytest.approx(bin_width / 4.0)
        assert first_bar.get_height() == pytest.approx(value)
        assert first_bar.get_facecolor()[:3] == pytest.approx(colour)
    assert "black: all classes" in figure.axes[4].get_title()


def test_coverage_axis_tops_at_the_final_coverage_rounded_up_to_ten():
    """The coverage axis ends at the next 10 % above the final coverage, never below 10 or above 100."""
    assert dashboard.coverage_axis_max((5.0, 36.2)) == 40.0
    assert dashboard.coverage_axis_max((40.0,)) == 40.0
    assert dashboard.coverage_axis_max((0.0, 0.0)) == 10.0
    assert dashboard.coverage_axis_max((99.5,)) == 100.0


def test_share_axis_runs_from_just_below_the_lowest_share_to_100():
    """The bottom is the lowest share rounded down to 5 %, 5 % lower when the share sits on it, never below 0; the top leaves 5 % of the span above 100."""
    assert dashboard.share_axis_limits((100.0, 92.0, 93.5)) == (90.0, 100.5)
    assert dashboard.share_axis_limits((100.0, 90.0)) == (85.0, 100.75)
    assert dashboard.share_axis_limits((100.0, 100.0)) == (95.0, 100.25)
    assert dashboard.share_axis_limits((100.0, 2.0)) == (0.0, 105.0)


def test_share_axis_label_names_the_runs_tries_per_particle():
    """In random mode the share label carries max_tries from the run, with try in the singular for one; centre mode, one search per particle, names no tries."""
    assert dashboard.panel_data(make_result(tries_rows(), max_tries=20), [6]).share_label == "Placed within 20 tries (% of attempted)"
    assert dashboard.panel_data(make_result(tries_rows(), max_tries=1), [6]).share_label == "Placed within 1 try (% of attempted)"
    assert dashboard.panel_data(make_result(tries_rows(), max_tries=1, mode="centre"), [6]).share_label == "Placed (% of attempted)"


def test_figures_for_first_and_last_steps_share_limits_and_match_the_data():
    """Axis limits never change; outlines, bar heights and lines equal the precomputed numbers; the line graph has two labelled y axes, the share axis from just below its lowest value to 100, and no legend."""
    result = make_result(tries_rows())
    data = dashboard.panel_data(result, [2, 6])
    first = dashboard.build_dashboard_figure(result, data, 2, 640, 360)
    last = dashboard.build_dashboard_figure(result, data, 6, 640, 360)
    assert len(last.axes) == 6
    for number in range(1, 6):
        assert first.axes[number].get_xlim() == last.axes[number].get_xlim()
        assert first.axes[number].get_ylim() == last.axes[number].get_ylim()
    line_axes, bar_axes, share_axes = last.axes[3], last.axes[4], last.axes[5]
    assert last.axes[1].get_ylim() == (0, data.histogram_max)
    assert line_axes.get_xlim() == (0, 6)
    assert line_axes.get_ylim() == (0, data.coverage_max)
    assert min(data.share_percent) == pytest.approx(60.0)
    assert share_axes.get_ylim() == (55.0, 102.25)
    assert bar_axes.get_ylim() == (0, data.bar_max)
    coverage_label, share_label = line_axes.yaxis.label, share_axes.yaxis.label
    assert coverage_label.get_text() == "Packing Factor (%)"
    assert share_label.get_text() == "Placed within 5 tries (% of attempted)"
    assert line_axes.get_xlabel() == "Particles attempted (placed or not)"
    assert coverage_label.get_color() == dashboard.COVERAGE_COLOUR
    assert share_label.get_color() == dashboard.SHARE_COLOUR
    for label in (coverage_label, share_label, line_axes.xaxis.label, line_axes.title):
        assert label.get_fontweight() == "bold"
    for tick_label in line_axes.get_yticklabels() + share_axes.get_yticklabels():
        assert tick_label.get_fontweight() == "bold"
    assert line_axes.get_legend() is None
    for figure, step in ((first, 2), (last, 6)):
        class_lines = figure.axes[2].patches
        assert [list(line.get_data().values) for line in class_lines] == [list(data.filled_counts[step][0])] * 2
        (coverage,) = figure.axes[3].get_lines()
        (share,) = figure.axes[5].get_lines()
        assert (coverage.get_color(), coverage.get_linestyle(), coverage.get_linewidth()) == (dashboard.COVERAGE_COLOUR, "-", dashboard.LINE_WIDTH)
        assert (share.get_color(), share.get_linestyle(), share.get_linewidth()) == (dashboard.SHARE_COLOUR, "--", dashboard.LINE_WIDTH)
        assert list(coverage.get_xdata()) == list(range(1, step + 1))
        assert list(coverage.get_ydata()) == list(data.coverage_percent[:step])
        assert list(share.get_ydata()) == list(data.share_percent[:step])
        values = data.bar_values[step][0]
        heights = [patch.get_height() for patch in figure.axes[4].containers[1]]
        assert heights == list(values[~numpy.isnan(values)])


def test_dashboard_figure_draws_the_slate_through_slate_drawing(monkeypatch):
    """The slate panel calls the slate-drawing function of its module, with the thick dashboard outline."""
    def raise_marker(axes, slate, rows, colours, outline_width=1.0):
        """Raise the marker carrying the outline width."""
        raise MarkerError(outline_width)

    monkeypatch.setattr(slate_drawing, "draw_slate", raise_marker)
    result = make_result(tries_rows())
    data = dashboard.panel_data(result, [6])
    with pytest.raises(MarkerError) as raised:
        dashboard.build_dashboard_figure(result, data, 6, 640, 360)
    assert raised.value.args == (dashboard.SLATE_OUTLINE_WIDTH,)
    assert dashboard.SLATE_OUTLINE_WIDTH > 1.0


def test_a_hundred_class_dashboard_keeps_its_legend_in_the_room_under_the_panels():
    """With 100 long-named classes the legend of a 1920 x 1080 dashboard stays within the bottom tenth of the figure, below every panel and inside its width."""
    rows = [make_row(number, 10.0 + number % 7, True, 1, label=f"Particle class {number:03d}") for number in range(1, 101)]
    result = make_result(rows)
    data = dashboard.panel_data(result, [100])
    figure = dashboard.build_dashboard_figure(result, data, 100, 1920, 1080)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    legend_box = figure.legends[0].get_window_extent(renderer)
    assert legend_box.y1 <= dashboard.LEGEND_ROOM * 1080
    assert legend_box.x0 >= 0 and legend_box.x1 <= 1920
    for axes in figure.axes:
        assert legend_box.y1 <= axes.get_window_extent(renderer).y0
