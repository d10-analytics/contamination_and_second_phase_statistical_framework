# This file checks the packing map of a saved placement run. Runs are built by
# hand (one or a few particles on a small slate), saved with the placement
# writer and mapped at a low resolution. It reads pixels of the written PNG at
# positions found through the transData of a figure built the same way: a
# particle as large as a circular slate reads the top colour everywhere on the
# slate, edge band included; a pixel outside every particle but within sigma of
# one is above 0 percent; a circle's corners are uncoloured and a rectangle's
# are coloured; the default sigma equals twice the largest diameter of all
# particles, placed or not; the sigma text and the colour bar limits are shown.
# It replaces the shared clearance and edge distance functions to prove the map
# calls them, once per placed particle on a part of the grid, and checks the
# writing rules (a failed save keeps an earlier good map and leaves no
# temporary file) and that every refusal is a ValueError with the folder unchanged.

import numpy
import pytest
from matplotlib import colormaps
from matplotlib.figure import Figure
from matplotlib.image import imread

from placement_simulation import geometry, saving
from placement_simulation.placement import PlacementResult, PlacementRow
from placement_visualization import packing_map, write_packing_map

SQUARE = {"shape": "square", "side_um": 100.0}
RECTANGLE = {"shape": "rectangle", "width_um": 200.0, "height_um": 80.0}
CIRCLE = {"shape": "circle", "diameter_um": 100.0}
INPUT = {"kind": "generator", "measurements_sha256": "0" * 64}
PIXELS = 100
WHITE = numpy.array([255, 255, 255])


class MarkerError(Exception):
    """Raised by a replacement so a test can tell the replacement was reached."""


def make_row(order, diameter, x, y):
    """Return a placement row; x of None means the particle was not placed."""
    return PlacementRow(
        placement_order=order,
        source_index=order,
        object_id=f"p{order}",
        feature_label="A",
        equivalent_diameter_um=diameter,
        placed=x is not None,
        x_um=x,
        y_um=y,
        tries=1,
        edge_rejections=0,
        overlap_rejections=0,
    )


def save_rows(folder, rows, slate=SQUARE):
    """Save a hand-built run in the folder and return its base path."""
    result = PlacementResult(
        slate=slate,
        mode="random",
        max_tries=5,
        buffer_um=1.0,
        pixel_um=1.0,
        order="shuffled",
        seed=1,
        input=INPUT,
        requested_coverage=0.0,
        achieved_coverage=0.0,
        rows=tuple(rows),
    )
    base = folder / "run"
    saving.write_placement_outputs(result, base)
    return base


def folder_state(folder):
    """Return every entry of the folder, hidden ones included, with its bytes."""
    state = {}
    for path in sorted(folder.iterdir()):
        state[path.name] = path.read_bytes()
    return state


def png_pixels(path):
    """Return the PNG as an array of 8-bit RGBA values."""
    return numpy.round(imread(path) * 255).astype(numpy.uint8)


def map_and_figure(base, sigma_um=None):
    """Write the map and return (its pixels, a drawn figure built the same way from the saved files)."""
    path = write_packing_map(base, sigma_um=sigma_um, long_side_px=PIXELS, overwrite=True)
    result = saving.read_placement_outputs(base)
    sigma = sigma_um if sigma_um is not None else packing_map.default_sigma_um(result.rows)
    figure = packing_map.build_packing_map_figure(result, sigma, PIXELS)
    figure.canvas.draw()
    pixels = png_pixels(path)
    assert numpy.array_equal(pixels, numpy.asarray(figure.canvas.buffer_rgba()))
    return pixels, figure


def pixel_at(figure, pixels, x, y):
    """Return the RGB value (0-255) at the data position (x, y) of the figure's slate axes."""
    column, row_from_bottom = figure.axes[0].transData.transform((x, y))
    row = pixels.shape[0] - 1 - int(row_from_bottom)
    return pixels[row, int(column), :3].astype(int)


def scale_colour(percentage):
    """Return the colour bar colour at the percentage as 0-255 integers."""
    return numpy.round(numpy.array(colormaps[packing_map.COLOUR_MAP](percentage / 100.0)[:3]) * 255).astype(int)


def same_colour(found, expected):
    """Say whether two RGB triples agree within 3 steps per channel."""
    return bool(numpy.all(numpy.abs(found - expected) <= 3))


def test_a_particle_as_large_as_the_circle_reads_the_top_colour_everywhere_on_the_slate(tmp_path):
    """Every pixel inside the circle, edge band included, is the top colour; an undivided blur would read about half at the edge."""
    base = save_rows(tmp_path, [make_row(1, 100.0, 0.0, 0.0)], CIRCLE)
    pixels, figure = map_and_figure(base)
    top = scale_colour(100.0)
    box = figure.axes[0].bbox
    inverse = figure.axes[0].transData.inverted()
    checked = 0
    for column in range(int(box.x0) + 1, int(box.x1)):
        for row_from_bottom in range(int(box.y0) + 1, int(box.y1)):
            x, y = inverse.transform((column + 0.5, row_from_bottom + 0.5))
            if numpy.hypot(x, y) > 45.5:
                continue
            found = pixels[pixels.shape[0] - 1 - row_from_bottom, column, :3].astype(int)
            assert same_colour(found, top), (x, y)
            checked += 1
    assert checked > 1500
    assert same_colour(pixel_at(figure, pixels, 46.0, 0.0), top)
    assert same_colour(pixel_at(figure, pixels, 0.0, -46.0), top)


def test_a_pixel_outside_every_particle_but_within_sigma_is_above_zero_and_a_far_one_is_zero(tmp_path):
    """The blur spreads coverage: next to the particle the colour is not the 0 percent colour, and with a small sigma far away it is."""
    base = save_rows(tmp_path, [make_row(1, 20.0, 0.0, 0.0)], SQUARE)
    pixels, figure = map_and_figure(base)
    near = pixel_at(figure, pixels, 15.0, 0.0)
    assert not same_colour(near, scale_colour(0.0))
    assert not same_colour(near, WHITE)
    small = tmp_path / "small"
    small.mkdir()
    base = save_rows(small, [make_row(1, 20.0, 0.0, 0.0)], SQUARE)
    pixels, figure = map_and_figure(base, sigma_um=2.0)
    assert not same_colour(pixel_at(figure, pixels, 12.0, 0.0), scale_colour(0.0))
    assert same_colour(pixel_at(figure, pixels, 0.0, 0.0), scale_colour(100.0))
    assert same_colour(pixel_at(figure, pixels, 40.0, 0.0), scale_colour(0.0))


def test_circle_corners_are_uncoloured_and_rectangle_corners_are_coloured(tmp_path):
    """Pixels at the bounding corner of a circle slate are the background; those at a rectangle's corners are on the colour scale."""
    circle = tmp_path / "circle"
    circle.mkdir()
    pixels, figure = map_and_figure(save_rows(circle, [make_row(1, 20.0, 0.0, 0.0)], CIRCLE))
    for sign_x in (-1, 1):
        for sign_y in (-1, 1):
            assert same_colour(pixel_at(figure, pixels, sign_x * 48.0, sign_y * 48.0), WHITE)
    rectangle = tmp_path / "rectangle"
    rectangle.mkdir()
    pixels, figure = map_and_figure(save_rows(rectangle, [make_row(1, 20.0, 0.0, 0.0)], RECTANGLE))
    for sign_x in (-1, 1):
        for sign_y in (-1, 1):
            found = pixel_at(figure, pixels, sign_x * 98.0, sign_y * 38.0)
            assert not same_colour(found, WHITE), (sign_x, sign_y)


def test_the_default_sigma_is_twice_the_largest_diameter_of_all_particles_placed_or_not(tmp_path):
    """The default PNG equals the one with sigma 80 (the unplaced 40 um particle), and differs from the one with sigma 20 (twice the placed 10 um one)."""
    rows = [make_row(1, 10.0, 0.0, 0.0), make_row(2, 40.0, None, None)]
    base = save_rows(tmp_path, rows)
    path = write_packing_map(base, long_side_px=PIXELS)
    default = path.read_bytes()
    write_packing_map(base, sigma_um=80.0, long_side_px=PIXELS, overwrite=True)
    assert path.read_bytes() == default
    write_packing_map(base, sigma_um=20.0, long_side_px=PIXELS, overwrite=True)
    assert path.read_bytes() != default


def test_the_figure_shows_the_sigma_used_and_a_colour_bar_from_0_to_100(tmp_path):
    """The figure text ends with sigma as %g followed by um, for the default and an explicit number, and the colour bar runs 0 to 100."""
    base = save_rows(tmp_path, [make_row(1, 10.0, 0.0, 0.0), make_row(2, 12.5, 20.0, 0.0)])
    result = saving.read_placement_outputs(base)
    figure = packing_map.build_packing_map_figure(result, packing_map.default_sigma_um(result.rows), PIXELS)
    assert [text.get_text() for text in figure.texts] == ["blur sigma 25 um"]
    other = packing_map.build_packing_map_figure(result, 12.5, PIXELS)
    assert [text.get_text() for text in other.texts] == ["blur sigma 12.5 um"]
    colour_bar_axes = figure.axes[1]
    assert colour_bar_axes.get_ylim() == (0.0, 100.0)
    assert len(figure.axes) == 2


def test_each_particle_clearance_is_evaluated_on_a_slice_of_the_grid_and_edge_distance_is_called(tmp_path, monkeypatch):
    """The shared clearance is called once per placed particle (none for the unplaced one) on fewer cells than the grid; the edge distance is used for the slate."""
    rows = [make_row(1, 10.0, -30.0, 0.0), make_row(2, 10.0, 30.0, 20.0), make_row(3, 30.0, None, None)]
    base = save_rows(tmp_path, rows)
    original_clearance = geometry.clearance
    original_edge = geometry.edge_distance
    sizes = []
    edge_calls = []

    def recording_clearance(x, y, placed_x, placed_y, placed_radius, buffer_um):
        """Record the evaluated cell count and the buffer, then delegate."""
        sizes.append((numpy.size(x), buffer_um))
        return original_clearance(x, y, placed_x, placed_y, placed_radius, buffer_um)

    def recording_edge(slate, x, y):
        """Record the call, then delegate."""
        edge_calls.append(numpy.size(x))
        return original_edge(slate, x, y)

    monkeypatch.setattr(geometry, "clearance", recording_clearance)
    monkeypatch.setattr(geometry, "edge_distance", recording_edge)
    write_packing_map(base, long_side_px=PIXELS)
    assert len(sizes) == 2
    assert all(count < PIXELS * PIXELS and buffer == 0.0 for count, buffer in sizes)
    assert len(edge_calls) == 1


def test_marker_replacements_of_the_shared_geometry_reach_the_writer(tmp_path, monkeypatch):
    """A clearance replacement and an edge distance replacement that each raise are reached through write_packing_map."""
    base = save_rows(tmp_path, [make_row(1, 20.0, 0.0, 0.0)])

    def marker_clearance(x, y, placed_x, placed_y, placed_radius, buffer_um):
        """Raise the marker instead of measuring."""
        raise MarkerError("clearance")

    def marker_edge(slate, x, y):
        """Raise the marker instead of measuring."""
        raise MarkerError("edge")

    with monkeypatch.context() as patch:
        patch.setattr(geometry, "clearance", marker_clearance)
        with pytest.raises(MarkerError, match="clearance"):
            write_packing_map(base, long_side_px=PIXELS)
    with monkeypatch.context() as patch:
        patch.setattr(geometry, "edge_distance", marker_edge)
        with pytest.raises(MarkerError, match="edge"):
            write_packing_map(base, long_side_px=PIXELS)
    assert sorted(entry.name for entry in tmp_path.iterdir()) == ["run_placements.csv", "run_summary.txt"]


def test_a_failed_save_keeps_the_earlier_good_map_and_leaves_no_temporary_file(tmp_path, monkeypatch):
    """A save that writes partial bytes and raises OSError makes the writer raise, leaving an earlier map byte-identical and no hidden file."""
    base = save_rows(tmp_path, [make_row(1, 20.0, 0.0, 0.0)])
    path = write_packing_map(base, long_side_px=PIXELS)
    before = folder_state(tmp_path)
    calls = []

    def partial_savefig(self, destination, format=None):
        """Leave partial bytes at the destination and fail."""
        calls.append(destination)
        with open(destination, "wb") as handle:
            handle.write(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(Figure, "savefig", partial_savefig)
    with pytest.raises(OSError):
        write_packing_map(base, long_side_px=PIXELS, overwrite=True)
    assert len(calls) == 1
    assert calls[0].name.startswith(".") and calls[0].name.endswith(".tmp")
    assert folder_state(tmp_path) == before
    assert path.read_bytes().startswith(b"\x89PNG")


def test_the_map_is_written_beside_the_base_path_with_the_requested_long_side(tmp_path):
    """The returned path is <base>_packing_map.png, the only new file, and the picture's longer side is long_side_px."""
    base = save_rows(tmp_path, [make_row(1, 20.0, 0.0, 0.0)], RECTANGLE)
    path = write_packing_map(base, long_side_px=PIXELS)
    assert path == tmp_path / "run_packing_map.png"
    assert sorted(entry.name for entry in tmp_path.iterdir()) == ["run_packing_map.png", "run_placements.csv", "run_summary.txt"]
    assert max(imread(path).shape[:2]) == PIXELS


def test_each_refused_argument_is_a_value_error_with_the_folder_unchanged(tmp_path):
    """A sigma that is not None or a finite number above 0, and a resolution that is not a whole number of 1 or more, are refused."""
    base = save_rows(tmp_path, [make_row(1, 20.0, 0.0, 0.0)])
    before = folder_state(tmp_path)
    for bad in [0, -1.0, float("nan"), float("inf"), True, "5"]:
        with pytest.raises(ValueError, match="sigma_um"):
            write_packing_map(base, sigma_um=bad, long_side_px=PIXELS)
        assert folder_state(tmp_path) == before
    for bad in [0, -1, 1.5, True]:
        with pytest.raises(ValueError, match="long_side_px"):
            write_packing_map(base, long_side_px=bad)
        assert folder_state(tmp_path) == before


def test_a_file_with_no_particles_is_refused(tmp_path):
    """A header-only placements file has nothing to map and no default sigma."""
    base = save_rows(tmp_path, [])
    before = folder_state(tmp_path)
    with pytest.raises(ValueError, match="no particles"):
        write_packing_map(base, long_side_px=PIXELS)
    assert folder_state(tmp_path) == before


def test_reader_refusals_reach_the_caller_with_the_folder_unchanged(tmp_path):
    """A damaged placements header and a missing run are ValueErrors and write nothing."""
    base = save_rows(tmp_path, [make_row(1, 20.0, 0.0, 0.0)])
    placements = tmp_path / "run_placements.csv"
    text = placements.read_text(encoding="utf-8")
    placements.write_text(text.replace("placement_order", "order"), encoding="utf-8")
    before = folder_state(tmp_path)
    with pytest.raises(ValueError, match="header"):
        write_packing_map(base, long_side_px=PIXELS)
    assert folder_state(tmp_path) == before
    with pytest.raises(ValueError, match="does not exist"):
        write_packing_map(tmp_path / "other", long_side_px=PIXELS)
    assert folder_state(tmp_path) == before


def test_an_existing_map_is_refused_without_overwrite_and_replaced_with_it(tmp_path):
    """The existing file stays byte-identical when refused; overwrite=True replaces it with a real PNG."""
    base = save_rows(tmp_path, [make_row(1, 20.0, 0.0, 0.0)])
    existing = tmp_path / "run_packing_map.png"
    existing.write_bytes(b"old picture")
    before = folder_state(tmp_path)
    with pytest.raises(ValueError, match="overwrite=True"):
        write_packing_map(base, long_side_px=PIXELS)
    assert folder_state(tmp_path) == before
    write_packing_map(base, long_side_px=PIXELS, overwrite=True)
    assert existing.read_bytes().startswith(b"\x89PNG")
    assert sorted(entry.name for entry in tmp_path.iterdir()) == ["run_packing_map.png", "run_placements.csv", "run_summary.txt"]


def test_the_sigma_text_lies_inside_the_picture_and_is_drawn_at_every_resolution(tmp_path):
    """At 100, 200, 400 and 800 cells on a wide rectangle the text's extent is inside the figure, its string ends with the sigma, and dark text pixels are in the written PNG."""
    base = save_rows(tmp_path, [make_row(1, 20.0, 0.0, 0.0)], RECTANGLE)
    result = saving.read_placement_outputs(base)
    for size in [100, 200, 400, 800]:
        path = write_packing_map(base, sigma_um=40.0, long_side_px=size, overwrite=True)
        figure = packing_map.build_packing_map_figure(result, 40.0, size)
        figure.canvas.draw()
        [text] = figure.texts
        assert text.get_text().endswith("40 um")
        extent = text.get_window_extent(figure.canvas.get_renderer())
        assert extent.x0 >= 0 and extent.y0 >= 0, size
        assert extent.x1 <= figure.bbox.x1 and extent.y1 <= figure.bbox.y1, size
        pixels = png_pixels(path)
        strip = pixels[int(pixels.shape[0] - extent.y1) : int(pixels.shape[0] - extent.y0) + 1, :, :3].astype(int)
        assert (strip.sum(axis=2) < 600).any(), size
        assert numpy.array_equal(pixels, numpy.asarray(figure.canvas.buffer_rgba())), size
