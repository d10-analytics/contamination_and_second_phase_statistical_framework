# This file checks centre placement. An oracle built here holds a map over every
# grid point in the slate's bounding box, lowers it over the whole array after
# each placement and picks the nearest fit by (i*i + j*j, -j, i); the real
# search, which keeps a growing window, must agree with it on square, rectangle
# and circle slates, with and without a buffer. The window sizes the search
# used are read back from the searcher the run creates. Smaller tests check the
# tie-break, the rows, the refusal of several tries, an unplaced particle, grid
# coordinates, a 1 m slate, and that the map reads the shared clearance.

import math
from pathlib import Path

import numpy as np
import pytest

from placement_simulation import centre_search, geometry, place_particles

HEADER = "feature_label,object_id,contour_area_um2,equivalent_diameter_um,volume_um3\n"
SQUARE_60 = {"shape": "square", "side_um": 60.0}
RECTANGLE_80_40 = {"shape": "rectangle", "width_um": 80.0, "height_um": 40.0}
CIRCLE_60 = {"shape": "circle", "diameter_um": 60.0}
RECORDED_SEARCHES = []


class Marker(Exception):
    """Raised by a replacement function so a test can see that the replaced function was reached."""


class RecordingSearch(centre_search.ClearanceSearch):
    """The real searcher that also remembers each instance, so a test can read its window sizes."""

    def __init__(self, slate, buffer_um, pixel_um, max_radius):
        """Build the searcher as usual and record it."""
        super().__init__(slate, buffer_um, pixel_um, max_radius)
        RECORDED_SEARCHES.append(self)


def write_csv(path, diameters):
    """Write a measurements CSV with the given diameters (object ids P1, P2, ...) and return its path."""
    lines = [HEADER]
    for number, diameter in enumerate(diameters, start=1):
        lines.append(f"Class A,P{number},1.5,{diameter!r},2.5\n")
    Path(path).write_text("".join(lines), encoding="utf-8")
    return path


def mixed_radii(count, seed):
    """Return count radii between 1 and 4 micrometres, small before large, the largest exactly 4."""
    radii = sorted(float(value) for value in np.random.default_rng(seed).uniform(1.0, 4.0, count))
    radii[-1] = 4.0
    return radii


def run_centre(tmp_path, radii, slate, buffer_um, pixel_um):
    """Place particles of these radii in the given order on the slate in centre mode."""
    path = write_csv(tmp_path / "particles.csv", [2.0 * radius for radius in radii])
    return place_particles(path, slate, mode="centre", buffer_um=buffer_um, pixel_um=pixel_um, seed=1)


def oracle_places(slate, radii, buffer_um, pixel_um):
    """Return (x, y) or None per radius from a whole-bounding-box map with no window, cap or update radius."""
    if slate["shape"] == "circle":
        half_x = slate["diameter_um"] / 2.0
        half_y = half_x
    else:
        half_x, half_y = geometry.half_extents(slate)
    cells_x = int(math.floor(half_x / pixel_um))
    cells_y = int(math.floor(half_y / pixel_um))
    index_i = np.arange(-cells_x, cells_x + 1)
    index_j = np.arange(-cells_y, cells_y + 1)
    xs = index_i * pixel_um
    ys = index_j * pixel_um
    values = geometry.edge_distance(slate, xs[:, None], ys[None, :])
    squares = index_i[:, None] ** 2 + index_j[None, :] ** 2
    outcomes = []
    for radius in radii:
        fits = values >= radius
        if not fits.any():
            outcomes.append(None)
            continue
        nearest = squares[fits].min()
        found_x, found_y = np.nonzero(fits & (squares == nearest))
        keys = []
        for position_x, position_y in zip(found_x, found_y):
            keys.append((-int(index_j[position_y]), int(index_i[position_x])))
        best_j, best_i = min(keys)
        x = best_i * pixel_um
        y = -best_j * pixel_um
        outcomes.append((x, y))
        values = np.minimum(values, geometry.clearance(xs[:, None], ys[None, :], x, y, radius, buffer_um))
    return outcomes


def check_against_oracle(tmp_path, slate, buffer_um):
    """Run about 200 mixed particles and return the searcher after comparing every placement to the oracle."""
    radii = mixed_radii(200, 7)
    RECORDED_SEARCHES.clear()
    result = run_centre(tmp_path, radii, slate, buffer_um, 0.5)
    expected = oracle_places(slate, radii, buffer_um, 0.5)
    unplaced = 0
    placed_circles = []
    for row, outcome, radius in zip(result.rows, expected, radii):
        if outcome is None:
            assert not row.placed
            assert row.x_um is None and row.y_um is None
            unplaced += 1
            continue
        assert row.placed
        assert (row.x_um, row.y_um) == outcome
        assert geometry.edge_distance(slate, row.x_um, row.y_um) >= radius
        for other_x, other_y, other_radius in placed_circles:
            gap = geometry.clearance(row.x_um, row.y_um, other_x, other_y, other_radius, buffer_um)
            assert gap >= radius
        placed_circles.append((row.x_um, row.y_um, radius))
    assert unplaced >= 1
    assert len(RECORDED_SEARCHES) == 1
    return RECORDED_SEARCHES[0]


def check_windows(windows, slate, buffer_um):
    """The window starts at buffer + 4 + 0.5 and each later entry doubles or reaches the slate's half-extent."""
    if slate["shape"] == "circle":
        half = (slate["diameter_um"] / 2.0, slate["diameter_um"] / 2.0)
    else:
        half = geometry.half_extents(slate)
    start = buffer_um + 4.0 + 0.5
    assert windows[0] == (min(start, half[0]), min(start, half[1]))
    assert len(windows) >= 3
    for before, after in zip(windows, windows[1:]):
        for axis in range(2):
            assert after[axis] >= 2 * before[axis] or after[axis] == half[axis]
            assert after[axis] >= before[axis]


def record_searches(monkeypatch):
    """Make the placement create RecordingSearch instead of the plain searcher."""
    monkeypatch.setattr(centre_search, "ClearanceSearch", RecordingSearch)


def test_square_slate_matches_oracle_and_window_doubles(tmp_path, monkeypatch):
    """On the 60 um square without a buffer the window half-extents are exactly 4.5, 9, 18, 30."""
    record_searches(monkeypatch)
    searcher = check_against_oracle(tmp_path, SQUARE_60, 0.0)
    assert searcher.window_half_widths == [(4.5, 4.5), (9.0, 9.0), (18.0, 18.0), (30.0, 30.0)]


def test_square_slate_with_buffer_matches_oracle(tmp_path, monkeypatch):
    """A 0.5 um buffer starts the window at 5.0 um and still equals the oracle."""
    record_searches(monkeypatch)
    searcher = check_against_oracle(tmp_path, SQUARE_60, 0.5)
    check_windows(searcher.window_half_widths, SQUARE_60, 0.5)
    assert searcher.window_half_widths[0] == (5.0, 5.0)


def test_rectangle_slate_matches_oracle(tmp_path, monkeypatch):
    """The 80 x 40 um rectangle clips each axis separately; both buffers equal the oracle."""
    record_searches(monkeypatch)
    for buffer_um in (0.0, 0.5):
        searcher = check_against_oracle(tmp_path, RECTANGLE_80_40, buffer_um)
        check_windows(searcher.window_half_widths, RECTANGLE_80_40, buffer_um)


def test_circle_slate_matches_oracle(tmp_path, monkeypatch):
    """The 60 um circle equals the oracle with both buffers."""
    record_searches(monkeypatch)
    for buffer_um in (0.0, 0.5):
        searcher = check_against_oracle(tmp_path, CIRCLE_60, buffer_um)
        check_windows(searcher.window_half_widths, CIRCLE_60, buffer_um)


def test_tie_break_places_equal_particles_by_coordinates(tmp_path):
    """Radius 5, pixel 1: (0, 0), then (0, 10) by largest y, then (-10, 0) by smallest x."""
    result = run_centre(tmp_path, [5.0, 5.0, 5.0], {"shape": "square", "side_um": 100.0}, 0.0, 1.0)
    assert [(row.x_um, row.y_um) for row in result.rows] == [(0.0, 0.0), (0.0, 10.0), (-10.0, 0.0)]


def test_centre_rows_have_one_try_and_no_rejections(tmp_path):
    """Placed and unplaced rows alike report tries 1 and zero rejections of both kinds."""
    result = run_centre(tmp_path, [1.0, 2.0, 30.0, 3.0], {"shape": "square", "side_um": 20.0}, 0.0, 1.0)
    assert [row.placed for row in result.rows] == [True, True, False, True]
    for row in result.rows:
        assert row.tries == 1
        assert row.edge_rejections == 0
        assert row.overlap_rejections == 0
    assert result.max_tries == 1
    assert result.mode == "centre"


def test_unplaced_particle_is_skipped_and_the_run_continues(tmp_path):
    """A particle too large for the slate is not placed; the next small one lands at the centre."""
    result = run_centre(tmp_path, [15.0, 2.0], {"shape": "square", "side_um": 20.0}, 0.0, 1.0)
    first, second = result.rows
    assert not first.placed
    assert first.x_um is None and first.y_um is None
    assert second.placed
    assert (second.x_um, second.y_um) == (0.0, 0.0)


def test_coordinates_are_whole_multiples_of_the_pixel(tmp_path):
    """With pixel 2.5 every placed coordinate divided by 2.5 is a whole number."""
    result = run_centre(tmp_path, mixed_radii(40, 3), {"shape": "square", "side_um": 80.0}, 0.5, 2.5)
    placed = [row for row in result.rows if row.placed]
    assert len(placed) > 5
    for row in placed:
        for value in (row.x_um, row.y_um):
            assert value / 2.5 == round(value / 2.5)


def test_several_tries_is_refused_before_any_output(tmp_path, capsys):
    """Centre mode with max_tries 2 raises ValueError and prints nothing on stderr."""
    path = write_csv(tmp_path / "particles.csv", [4.0])
    with pytest.raises(ValueError, match="max_tries"):
        place_particles(path, SQUARE_60, mode="centre", max_tries=2)
    assert capsys.readouterr().err == ""


def test_centre_mode_never_draws_locations_and_ignores_the_seed(tmp_path, monkeypatch):
    """Random points are never drawn, and two different seeds give the same placements."""
    path = write_csv(tmp_path / "particles.csv", [4.0, 6.0, 8.0])

    def refuse(slate, rng):
        """Fail when a random location is requested."""
        raise Marker("random_point was called")

    monkeypatch.setattr(geometry, "random_point", refuse)
    first = place_particles(path, SQUARE_60, mode="centre", seed=1)
    second = place_particles(path, SQUARE_60, mode="centre", seed=2)
    assert first.rows == second.rows


def test_one_metre_slate_places_as_a_small_slate_does(tmp_path):
    """Three small particles on a 1 m square at 1 um complete, the first at (0, 0), as on a 100 um slate."""
    path = write_csv(tmp_path / "particles.csv", [10.0, 10.0, 6.0])
    big = place_particles(path, {"shape": "square", "side_um": 1.0e6}, mode="centre", pixel_um=1.0, seed=1)
    small = place_particles(path, {"shape": "square", "side_um": 100.0}, mode="centre", pixel_um=1.0, seed=1)
    assert (big.rows[0].x_um, big.rows[0].y_um) == (0.0, 0.0)
    assert [(row.x_um, row.y_um) for row in big.rows] == [(row.x_um, row.y_um) for row in small.rows]
    assert all(row.placed for row in big.rows)


def test_the_map_reads_the_shared_clearance(tmp_path, monkeypatch):
    """With clearance 0.5 um smaller, the second equal particle no longer lands at (0, 10)."""
    original = geometry.clearance
    slate = {"shape": "square", "side_um": 100.0}
    normal = run_centre(tmp_path, [5.0, 5.0], slate, 0.0, 1.0)
    assert (normal.rows[1].x_um, normal.rows[1].y_um) == (0.0, 10.0)

    def smaller(x, y, placed_x, placed_y, placed_radius, buffer_um):
        """The shared clearance, 0.5 um smaller."""
        return original(x, y, placed_x, placed_y, placed_radius, buffer_um) - 0.5

    monkeypatch.setattr(geometry, "clearance", smaller)
    shifted = run_centre(tmp_path, [5.0, 5.0], slate, 0.0, 1.0)
    assert (shifted.rows[1].x_um, shifted.rows[1].y_um) != (0.0, 10.0)
