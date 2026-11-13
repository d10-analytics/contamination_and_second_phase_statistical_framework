# This file checks the dashboard of the video. The panel numbers are computed
# without drawing from hand-built rows: random placement with several tries,
# random with one try and centre placement, placed and unplaced particles, an
# empty size bin and an early particle with many tries. It checks the running
# coverage (ending at the achieved coverage), the share placed out of those
# taken, the histogram counts, the bar values and both fixed maxima, and that
# a replacement size-bin function reaches both panels. Figures for the first
# and last shown steps are then drawn and checked for identical axis limits,
# bar heights and line data equal to the numbers, and for the slate-drawing
# function being the one called.

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
    """The outline counts every particle in the file; the filled counts those placed among the first k."""
    data = dashboard.panel_data(make_result(tries_rows()), [2, 6])
    assert data.outline_counts[0] == 3
    assert data.outline_counts[-1] == 3
    assert data.outline_counts.sum() == 6
    assert list(data.filled_counts[2][[0, -1]]) == [2, 0]
    assert list(data.filled_counts[6][[0, -1]]) == [2, 2]
    assert data.histogram_max == 3.0


def test_mean_tries_per_bin_and_empty_bins_have_no_bar():
    """Mean tries averages the saved tries of placed particles; a bin with none placed, or empty, is NaN."""
    data = dashboard.panel_data(make_result(tries_rows()), [3, 6])
    assert not data.bar_is_unplaced_share
    first = data.bar_values[3]
    assert first[0] == pytest.approx(5.0)
    assert numpy.isnan(first[-1])
    assert numpy.all(numpy.isnan(first[1:-1]))
    last = data.bar_values[6]
    assert last[0] == pytest.approx(5.0)
    assert last[-1] == pytest.approx(2.5)
    assert numpy.all(numpy.isnan(last[1:-1]))


def test_mean_tries_maximum_covers_an_early_large_mean_that_falls():
    """The axis maximum is over all shown steps, so it exceeds the final-state maximum."""
    rows = [make_row(1, 10.0, True, 9)] + [make_row(number, 10.0, True, 1) for number in range(2, 6)] + [make_row(6, 30.0, True, 2)]
    data = dashboard.panel_data(make_result(rows), [1, 5, 6])
    assert data.bar_values[1][0] == pytest.approx(9.0)
    assert data.bar_values[6][0] == pytest.approx(13.0 / 5.0)
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
    assert data.bar_is_unplaced_share
    assert data.bar_values[3][0] == pytest.approx(0.0)
    assert data.bar_values[3][-1] == pytest.approx(100.0)
    assert data.bar_values[6][0] == pytest.approx(100.0 / 3.0)
    assert data.bar_values[6][-1] == pytest.approx(100.0 / 3.0)
    assert numpy.all(numpy.isnan(data.bar_values[6][1:-1]))
    assert data.bar_max == 100.0


def test_centre_run_shows_percent_not_placed():
    """Centre placement, which saves one try per particle, also shows the not-placed share."""
    data = dashboard.panel_data(make_result(tries_rows(), max_tries=1, mode="centre"), [6])
    assert data.bar_is_unplaced_share
    assert data.bar_values[6][0] == pytest.approx(100.0 / 3.0)


def test_replacement_bin_edges_are_used_by_both_panels(monkeypatch):
    """A replacement size_bin_edges with distinct edges sets the histogram and the bar chart bins."""
    edges = numpy.array([0.0, 20.0, 40.0])

    def replacement_edges(diameters):
        """Return the replacement edges whatever the diameters."""
        return edges

    monkeypatch.setattr(dashboard, "size_bin_edges", replacement_edges)
    result = make_result(tries_rows())
    data = dashboard.panel_data(result, [6])
    assert numpy.array_equal(data.edges, edges)
    assert list(data.outline_counts) == [3, 3]
    assert list(data.bar_values[6]) == pytest.approx([5.0, 2.5])
    figure = dashboard.build_dashboard_figure(result, data, 6, 640, 360)
    histogram, _, bars = figure.axes[1], figure.axes[2], figure.axes[3]
    assert [patch.get_x() for patch in histogram.containers[0]] == [0.0, 20.0]
    assert [patch.get_x() for patch in bars.containers[0]] == [0.0, 20.0]
    assert [patch.get_width() for patch in bars.containers[0]] == [20.0, 20.0]


def test_figures_for_first_and_last_steps_share_limits_and_match_the_data():
    """Axis limits never change; bar heights and lines equal the precomputed numbers."""
    result = make_result(tries_rows())
    data = dashboard.panel_data(result, [2, 6])
    first = dashboard.build_dashboard_figure(result, data, 2, 640, 360)
    last = dashboard.build_dashboard_figure(result, data, 6, 640, 360)
    for number in range(1, 4):
        assert first.axes[number].get_xlim() == last.axes[number].get_xlim()
        assert first.axes[number].get_ylim() == last.axes[number].get_ylim()
    assert last.axes[1].get_ylim() == (0, data.histogram_max)
    assert last.axes[2].get_xlim() == (0, 6)
    assert last.axes[2].get_ylim() == (0, 100.0)
    assert last.axes[3].get_ylim() == (0, data.bar_max)
    for figure, step in ((first, 2), (last, 6)):
        outline, filled = figure.axes[1].containers
        assert [patch.get_height() for patch in outline] == list(data.outline_counts)
        assert [patch.get_height() for patch in filled] == list(data.filled_counts[step])
        coverage, share = figure.axes[2].get_lines()
        assert list(coverage.get_xdata()) == list(range(1, step + 1))
        assert list(coverage.get_ydata()) == list(data.coverage_percent[:step])
        assert list(share.get_ydata()) == list(data.share_percent[:step])
        values = data.bar_values[step]
        heights = [patch.get_height() for patch in figure.axes[3].containers[0]]
        assert heights == list(values[~numpy.isnan(values)])


def test_dashboard_figure_draws_the_slate_through_slate_drawing(monkeypatch):
    """The slate panel calls the slate-drawing function of its module."""
    def raise_marker(axes, slate, rows, colours):
        """Raise the marker."""
        raise MarkerError

    monkeypatch.setattr(slate_drawing, "draw_slate", raise_marker)
    result = make_result(tries_rows())
    data = dashboard.panel_data(result, [6])
    with pytest.raises(MarkerError):
        dashboard.build_dashboard_figure(result, data, 6, 640, 360)
