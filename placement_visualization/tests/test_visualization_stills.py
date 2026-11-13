# This file checks the still images of a saved placement run. Runs are
# built by hand (a few particles on a small slate), saved with the placement
# writer and drawn at a low resolution. It checks the file names and the
# folder contents for several step sizes; reads pixels of the written PNG at
# positions found through the transData of the figure the writer saves
# (colour inside a particle's true radius, background outside it, a particle
# taken later absent, an unplaced particle never drawn, an eleven-class file
# repeating the first colour); checks that the legend lies below the slate and
# inside the picture, that the slate keeps its pixel scale whatever the legend,
# and that a size below the minimum is refused; and
# checks the writing rules: an existing file is refused without overwrite and
# nothing earlier is written, a failed save leaves no cut-off or temporary
# file, and every refused argument or file is a ValueError with the folder
# unchanged.

import numpy
import pytest
from matplotlib import colormaps
from matplotlib.figure import Figure
from matplotlib.image import imread

from placement_simulation import saving
from placement_simulation.placement import PlacementResult, PlacementRow
from placement_visualization import picture_files, slate_drawing, stills, write_still_images

SQUARE = {"shape": "square", "side_um": 100.0}
RECTANGLE = {"shape": "rectangle", "width_um": 200.0, "height_um": 80.0}
CIRCLE = {"shape": "circle", "diameter_um": 100.0}
INPUT = {"kind": "generator", "measurements_sha256": "0" * 64}
PIXELS = 400
TAB10 = colormaps["tab10"].colors


class MarkerError(Exception):
    """Raised by a replacement so a test can tell the replacement was reached."""


def make_row(order, label, diameter, x, y):
    """Return a placement row; x of None means the particle was not placed."""
    placed = x is not None
    return PlacementRow(
        placement_order=order,
        source_index=order,
        object_id=f"p{order}",
        feature_label=label,
        equivalent_diameter_um=diameter,
        placed=placed,
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


def four_rows():
    """Return four particles: classes B, A, an unplaced C, and a B taken last."""
    return [
        make_row(1, "B", 20.0, -25.0, 25.0),
        make_row(2, "A", 20.0, 25.0, 25.0),
        make_row(3, "C", 30.0, None, None),
        make_row(4, "B", 20.0, 25.0, -25.0),
    ]


def ten_rows():
    """Return ten placed particles of one class on a line."""
    return [make_row(number, "A", 4.0, -45.0 + 9.0 * number, 0.0) for number in range(1, 11)]


def folder_state(folder):
    """Return every entry of the folder, hidden ones included, with its bytes."""
    state = {}
    for path in sorted(folder.iterdir()):
        state[path.name] = path.read_bytes()
    return state


def png_pixels(path):
    """Return the PNG as an array of 8-bit RGBA values."""
    return numpy.round(imread(path) * 255).astype(numpy.uint8)


def pixel_at(figure, pixels, x, y):
    """Return the RGB value (0-255) at the data position (x, y) of the figure's one axes."""
    axes = figure.axes[0]
    column, row_from_bottom = axes.transData.transform((x, y))
    row = pixels.shape[0] - 1 - int(row_from_bottom)
    return pixels[row, int(column), :3].astype(int)


def colour_of(position):
    """Return the tab10 colour at the position as 0-255 integers."""
    return numpy.round(numpy.array(TAB10[position % 10]) * 255).astype(int)


def same_colour(found, expected):
    """Say whether two RGB triples agree within 3 steps per channel."""
    return bool(numpy.all(numpy.abs(found - expected) <= 3))


def test_shown_steps_for_each_kind_of_step_size():
    """Shown steps are every, 2*every, ... plus the last step when it is not a multiple; one image at the count when every exceeds it."""
    assert stills.shown_steps(10, 1) == list(range(1, 11))
    assert stills.shown_steps(10, 2) == [2, 4, 6, 8, 10]
    assert stills.shown_steps(10, 3) == [3, 6, 9, 10]
    assert stills.shown_steps(10, 10) == [10]
    assert stills.shown_steps(10, 25) == [10]
    assert stills.shown_steps(1, 1) == [1]


def test_file_names_and_folder_contents(tmp_path):
    """The names are padded to the digits of the count, and the folder holds exactly those files plus the two inputs."""
    cases = [
        (1, ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]),
        (2, ["02", "04", "06", "08", "10"]),
        (3, ["03", "06", "09", "10"]),
        (25, ["10"]),
    ]
    for every, steps in cases:
        folder = tmp_path / f"every_{every}"
        folder.mkdir()
        base = save_rows(folder, ten_rows())
        paths = write_still_images(base, every, long_side_px=PIXELS)
        names = [f"run_step_{step}.png" for step in steps]
        assert [path.name for path in paths] == names
        assert all(path.parent == folder for path in paths)
        assert sorted(entry.name for entry in folder.iterdir()) == sorted(names + ["run_placements.csv", "run_summary.txt"])


def test_a_count_of_one_hundred_pads_to_three_digits(tmp_path):
    """With 100 particles the step number has three digits, so a step of 5 is named _step_005."""
    rows = [make_row(number, "A", 1.0, -49.0 + 0.9 * number, 0.0) for number in range(1, 101)]
    base = save_rows(tmp_path, rows)
    paths = write_still_images(base, 50, long_side_px=PIXELS)
    assert [path.name for path in paths] == ["run_step_050.png", "run_step_100.png"]


def test_pixels_show_true_size_circles_for_each_step(tmp_path):
    """At 0.9 of a radius the class colour shows and at 1.1 the background; a later particle is absent; an unplaced one is never drawn."""
    base = save_rows(tmp_path, four_rows())
    paths = write_still_images(base, 2, long_side_px=PIXELS)
    rows = stills.shown_steps(4, 2)
    assert rows == [2, 4]
    colours = {"A": colour_of(0), "B": colour_of(1), "C": colour_of(2)}
    result = saving.read_placement_outputs(base)
    for step, path in zip([2, 4], paths):
        figure = stills.build_still_figure(result, step, PIXELS)
        figure.canvas.draw()
        pixels = png_pixels(path)
        assert numpy.array_equal(pixels, numpy.asarray(figure.canvas.buffer_rgba())), step
        assert pixels.shape[:2] == (figure.canvas.get_width_height()[1], figure.canvas.get_width_height()[0])
        for row in result.rows[:step]:
            if not row.placed:
                continue
            radius = row.equivalent_diameter_um / 2.0
            inside = pixel_at(figure, pixels, row.x_um + 0.9 * radius, row.y_um)
            outside = pixel_at(figure, pixels, row.x_um + 1.1 * radius, row.y_um)
            assert same_colour(inside, colours[row.feature_label]), (step, row.object_id)
            assert same_colour(outside, numpy.array([255, 255, 255])), (step, row.object_id)
        box = figure.axes[0].bbox
        region = pixels[pixels.shape[0] - int(box.y1) : pixels.shape[0] - int(box.y0), int(box.x0) : int(box.x1), :3].astype(int)
        unplaced_colour = numpy.all(numpy.abs(region - colours["C"]) <= 3, axis=2)
        assert not unplaced_colour.any(), step
    early = stills.build_still_figure(result, 2, PIXELS)
    early.canvas.draw()
    early_pixels = png_pixels(paths[0])
    late_row = result.rows[3]
    assert same_colour(pixel_at(early, early_pixels, late_row.x_um, late_row.y_um), numpy.array([255, 255, 255]))
    late = stills.build_still_figure(result, 4, PIXELS)
    late.canvas.draw()
    assert same_colour(pixel_at(late, png_pixels(paths[1]), late_row.x_um, late_row.y_um), colours["B"])


def test_an_eleventh_class_takes_the_first_colour(tmp_path):
    """Eleven classes in sorted order give the eleventh the first tab10 colour, and the tenth the last."""
    rows = []
    for number in range(11):
        rows.append(make_row(number + 1, f"k{number:02d}", 6.0, -45.0 + 9.0 * number, 0.0))
    base = save_rows(tmp_path, rows)
    paths = write_still_images(base, 11, long_side_px=PIXELS)
    result = saving.read_placement_outputs(base)
    figure = stills.build_still_figure(result, 11, PIXELS)
    figure.canvas.draw()
    pixels = png_pixels(paths[0])
    assert numpy.array_equal(pixels, numpy.asarray(figure.canvas.buffer_rgba()))
    for number in range(11):
        row = result.rows[number]
        found = pixel_at(figure, pixels, row.x_um, row.y_um)
        assert same_colour(found, colour_of(number)), number
    assert same_colour(pixel_at(figure, pixels, result.rows[10].x_um, 0.0), colour_of(0))
    assert not same_colour(colour_of(10), colour_of(9))


def test_the_legend_lists_every_class_in_sorted_order_at_every_step(tmp_path):
    """The legend names every class in the file in sorted order, including one not yet drawn or never placed."""
    base = save_rows(tmp_path, four_rows())
    result = saving.read_placement_outputs(base)
    for step in [1, 2, 4]:
        figure = stills.build_still_figure(result, step, PIXELS)
        labels = [text.get_text() for text in figure.legends[0].get_texts()]
        assert labels == ["A", "B", "C"], step


def test_the_slate_long_side_is_the_requested_size_for_each_slate_shape_and_the_legend_adds_below(tmp_path):
    """In the written PNG the slate's long side is long_side_px for a square, a wide rectangle, a tall rectangle and a circle, and the PNG is taller than it by the legend."""
    slates = [SQUARE, RECTANGLE, {"shape": "rectangle", "width_um": 60.0, "height_um": 150.0}, CIRCLE]
    for number, slate in enumerate(slates):
        folder = tmp_path / str(number)
        folder.mkdir()
        base = save_rows(folder, [make_row(1, "A", 10.0, 0.0, 0.0)], slate)
        result = saving.read_placement_outputs(base)
        for size in [PIXELS, 733]:
            [path] = write_still_images(base, 1, long_side_px=size, overwrite=True)
            figure = stills.build_still_figure(result, 1, size)
            figure.canvas.draw()
            axes_box = figure.axes[0].get_window_extent(figure.canvas.get_renderer())
            assert max(axes_box.width, axes_box.height) == pytest.approx(size, abs=1e-6), (slate, size)
            height, width = imread(path).shape[:2]
            assert (width, height) == figure.canvas.get_width_height(), (slate, size)
            assert height > axes_box.height, (slate, size)


def test_circles_and_the_outline_follow_the_slate_extent(tmp_path):
    """The axes limits are the slate's extent with equal aspect, so a rectangle's pixels are square."""
    base = save_rows(tmp_path, [make_row(1, "A", 10.0, 0.0, 0.0)], RECTANGLE)
    result = saving.read_placement_outputs(base)
    figure = stills.build_still_figure(result, 1, PIXELS)
    figure.canvas.draw()
    axes = figure.axes[0]
    assert axes.get_xlim() == (-100.0, 100.0)
    assert axes.get_ylim() == (-40.0, 40.0)
    assert axes.get_aspect() == 1.0
    assert isinstance(figure, Figure)


def test_an_existing_last_step_file_without_overwrite_writes_nothing_earlier(tmp_path):
    """A pre-existing file for the last shown step is refused before any earlier step's file is written."""
    base = save_rows(tmp_path, ten_rows())
    existing = tmp_path / "run_step_10.png"
    existing.write_bytes(b"old picture")
    before = folder_state(tmp_path)
    with pytest.raises(ValueError, match="overwrite=True"):
        write_still_images(base, 3, long_side_px=PIXELS)
    assert folder_state(tmp_path) == before
    assert existing.read_bytes() == b"old picture"


def test_overwrite_replaces_existing_files(tmp_path):
    """With overwrite the existing file is replaced by a real PNG."""
    base = save_rows(tmp_path, ten_rows())
    existing = tmp_path / "run_step_10.png"
    existing.write_bytes(b"old picture")
    paths = write_still_images(base, 3, long_side_px=PIXELS, overwrite=True)
    assert existing in paths
    assert existing.read_bytes().startswith(b"\x89PNG")
    assert sorted(entry.name for entry in tmp_path.iterdir()) == sorted(
        ["run_placements.csv", "run_summary.txt", "run_step_03.png", "run_step_06.png", "run_step_09.png", "run_step_10.png"]
    )


def test_a_folder_at_a_destination_is_refused(tmp_path):
    """A folder named like an output is refused before anything is written."""
    base = save_rows(tmp_path, ten_rows())
    (tmp_path / "run_step_10.png").mkdir()
    with pytest.raises(ValueError):
        write_still_images(base, 5, long_side_px=PIXELS, overwrite=True)
    assert sorted(entry.name for entry in tmp_path.iterdir()) == ["run_placements.csv", "run_step_10.png", "run_summary.txt"]


def partial_save_after(good_saves, monkeypatch):
    """Replace Figure.savefig so the call after good_saves real ones writes partial bytes and raises OSError."""
    original = Figure.savefig
    calls = []

    def savefig(self, destination, format=None):
        """Save for real until the chosen call, then leave partial bytes and fail."""
        calls.append(destination)
        if len(calls) > good_saves:
            with open(destination, "wb") as handle:
                handle.write(b"partial")
            raise OSError("disk full")
        original(self, destination, format=format)

    monkeypatch.setattr(Figure, "savefig", savefig)
    return calls


def test_a_failed_save_leaves_the_earlier_good_file_and_no_temporary_file(tmp_path, monkeypatch):
    """A save that writes partial bytes and raises leaves an existing file byte-identical and no hidden file behind."""
    base = save_rows(tmp_path, ten_rows())
    existing = tmp_path / "run_step_05.png"
    existing.write_bytes(b"good old picture")
    before = folder_state(tmp_path)
    calls = partial_save_after(0, monkeypatch)
    with pytest.raises(OSError):
        write_still_images(base, 5, long_side_px=PIXELS, overwrite=True)
    assert len(calls) == 1
    assert calls[0].name.startswith(".") and calls[0].name.endswith(".tmp")
    assert folder_state(tmp_path) == before


def test_a_failure_part_way_leaves_the_earlier_steps_complete(tmp_path, monkeypatch):
    """When the second picture fails, the first stays as a complete PNG and no hidden temporary file remains."""
    base = save_rows(tmp_path, ten_rows())
    partial_save_after(1, monkeypatch)
    with pytest.raises(OSError):
        write_still_images(base, 5, long_side_px=PIXELS)
    assert sorted(entry.name for entry in tmp_path.iterdir()) == ["run_placements.csv", "run_step_05.png", "run_summary.txt"]
    assert (tmp_path / "run_step_05.png").read_bytes().startswith(b"\x89PNG")


def test_each_refused_argument_is_a_value_error_with_nothing_written(tmp_path):
    """Zero, a negative, a fraction and a boolean are refused for both arguments, before any output."""
    base = save_rows(tmp_path, ten_rows())
    before = folder_state(tmp_path)
    for bad in [0, -1, 1.5, True]:
        with pytest.raises(ValueError, match="every"):
            write_still_images(base, bad, long_side_px=PIXELS)
        with pytest.raises(ValueError, match="long_side_px"):
            write_still_images(base, 3, long_side_px=bad)
        assert folder_state(tmp_path) == before
    assert picture_files.check_whole_number(numpy.int64(3), "every") == 3


def test_a_file_with_no_particles_is_refused(tmp_path):
    """A header-only placements file has nothing to draw."""
    base = save_rows(tmp_path, [])
    before = folder_state(tmp_path)
    with pytest.raises(ValueError, match="no particles"):
        write_still_images(base, 1, long_side_px=PIXELS)
    assert folder_state(tmp_path) == before


def test_reader_refusals_reach_the_caller_unchanged(tmp_path):
    """A missing summary and a damaged placements header are ValueErrors and write nothing."""
    base = save_rows(tmp_path, ten_rows())
    placements = tmp_path / "run_placements.csv"
    text = placements.read_text(encoding="utf-8")
    placements.write_text(text.replace("placement_order", "order"), encoding="utf-8")
    before = folder_state(tmp_path)
    with pytest.raises(ValueError, match="header"):
        write_still_images(base, 3, long_side_px=PIXELS)
    assert folder_state(tmp_path) == before
    with pytest.raises(ValueError, match="does not exist"):
        write_still_images(tmp_path / "other", 3, long_side_px=PIXELS)


def test_the_drawing_reads_the_saved_files_only_through_the_placement_reader(tmp_path, monkeypatch):
    """A replacement of saving.read_placement_outputs is the one the writer calls."""
    base = save_rows(tmp_path, ten_rows())

    def marker_reader(base_path):
        """Raise the marker instead of reading."""
        raise MarkerError(str(base_path))

    monkeypatch.setattr(saving, "read_placement_outputs", marker_reader)
    with pytest.raises(MarkerError):
        write_still_images(base, 3, long_side_px=PIXELS)


def last_step_only(particle_count, every):
    """Return only the last step."""
    return [particle_count]


def test_replacements_of_shown_steps_and_slate_drawing_reach_the_writer(tmp_path, monkeypatch):
    """The writer calls the shown-steps and slate-drawing functions on their modules."""
    base = save_rows(tmp_path, ten_rows())
    monkeypatch.setattr(stills, "shown_steps", last_step_only)
    paths = write_still_images(base, 3, long_side_px=PIXELS)
    assert [path.name for path in paths] == ["run_step_10.png"]

    def marker_draw(axes, slate, rows, colours):
        """Raise the marker instead of drawing."""
        raise MarkerError("draw")

    monkeypatch.setattr(slate_drawing, "draw_slate", marker_draw)
    with pytest.raises(MarkerError):
        write_still_images(base, 3, long_side_px=PIXELS, overwrite=True)


def class_rows(count, prefix="Class"):
    """Return one placed particle for each of count classes, spread along the middle of the slate."""
    return [make_row(number, f"{prefix} {number:02d}", 8.0, -40.0 + 8.0 * number, -40.0 + 8.0 * number) for number in range(1, count + 1)]


def result_of(rows, slate):
    """Return a placement result holding the rows on the slate, without saving it."""
    return PlacementResult(
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


TALL = {"shape": "rectangle", "width_um": 100.0, "height_um": 1000.0}
WIDE = {"shape": "rectangle", "width_um": 1000.0, "height_um": 100.0}
LONG, SHORT = "Particle class", "Class"
LEGEND_CASES = [
    (WIDE, 11, 2000, LONG),
    (TALL, 11, 2000, LONG),
    (SQUARE, 11, 400, LONG),
    (TALL, 3, 400, LONG),
    (SQUARE, 5, 100, LONG),
    (WIDE, 3, 137, SHORT),
    (WIDE, 25, 400, LONG),
    (TALL, 25, 400, LONG),
    (SQUARE, 1, 400, SHORT),
]


def drawn_still(slate, class_count, long_side_px, prefix):
    """Return the drawn still figure of a run with that many classes named with the prefix."""
    result = result_of(class_rows(class_count, prefix), slate)
    figure = stills.build_still_figure(result, class_count, long_side_px)
    figure.canvas.draw()
    return figure


def test_the_legend_lies_under_the_slate_and_wholly_inside_the_picture_at_the_set_font():
    """The legend's drawn extent is below the slate axes and inside the figure, in 10 point type, so it never covers the slate or runs off the picture."""
    for slate, class_count, long_side_px, prefix in LEGEND_CASES:
        case = (slate, class_count, long_side_px, prefix)
        figure = drawn_still(slate, class_count, long_side_px, prefix)
        renderer = figure.canvas.get_renderer()
        legend_box = figure.legends[0].get_window_extent(renderer)
        axes_box = figure.axes[0].get_window_extent(renderer)
        assert legend_box.y1 <= axes_box.y0, case
        assert legend_box.x0 >= 0, case
        assert legend_box.y0 >= 0, case
        assert legend_box.x1 <= figure.bbox.x1, case
        assert legend_box.y1 <= figure.bbox.y1, case
        assert {text.get_fontsize() for text in figure.legends[0].get_texts()} == {slate_drawing.LEGEND_MAX_FONT_POINTS}, case


def test_the_slate_pixels_are_the_same_with_and_without_the_legend():
    """Removing the legend changes no pixel inside the slate axes, so the legend overlaps nothing the slate shows."""
    for slate, class_count, long_side_px, prefix in LEGEND_CASES:
        case = (slate, class_count, long_side_px, prefix)
        figure = drawn_still(slate, class_count, long_side_px, prefix)
        box = figure.axes[0].get_window_extent(figure.canvas.get_renderer())
        height = figure.canvas.get_width_height()[1]
        top, bottom = int(height - box.y1), int(numpy.ceil(height - box.y0))
        left, right = int(box.x0), int(numpy.ceil(box.x1))
        with_legend = numpy.asarray(figure.canvas.buffer_rgba())[top:bottom, left:right].copy()
        figure.legends[0].remove()
        figure.canvas.draw()
        without_legend = numpy.asarray(figure.canvas.buffer_rgba())[top:bottom, left:right]
        assert numpy.array_equal(with_legend, without_legend), case


def test_the_slate_scale_does_not_depend_on_the_classes():
    """The slate axes' long side is long_side_px pixels with 1, 5 and 25 classes, and one micrometre spans the same number of pixels."""
    for slate, size in [(SQUARE, 400), (WIDE, 733), (TALL, 500)]:
        half = max(slate_drawing.slate_extent(slate))
        for class_count in [1, 5, 25]:
            figure = drawn_still(slate, class_count, size, LONG)
            axes_box = figure.axes[0].get_window_extent(figure.canvas.get_renderer())
            assert max(axes_box.width, axes_box.height) == pytest.approx(size, abs=1e-6), (slate, size, class_count)
            origin = figure.axes[0].transData.transform((0.0, 0.0))
            unit = figure.axes[0].transData.transform((1.0, 1.0)) - origin
            assert unit == pytest.approx([size / (2 * half)] * 2, abs=1e-9), (slate, size, class_count)


def test_a_particle_spans_its_true_diameter_in_pixels_below_a_large_legend():
    """A 50 micrometre particle on a 100 micrometre square slate at 400 pixels is 200 pixels across, though 25 classes put a tall legend below it."""
    rows = class_rows(25)
    rows.append(make_row(26, "Class 00", 50.0, 0.0, 0.0))
    figure = stills.build_still_figure(result_of(rows, SQUARE), 26, PIXELS)
    figure.canvas.draw()
    pixels = numpy.asarray(figure.canvas.buffer_rgba())[:, :, :3].astype(int)
    centre_row = pixels.shape[0] - 1 - int(figure.axes[0].transData.transform((0.0, 0.0))[1])
    painted = numpy.all(numpy.abs(pixels[centre_row] - colour_of(0)) <= 3, axis=1)
    assert abs(int(painted.sum()) - 200) <= 2


def saved_class_run(folder, slate, class_count):
    """Save a run of that many long-labelled classes on the slate and return its base path."""
    return save_rows(folder, class_rows(class_count, LONG), slate)


def test_a_size_below_the_minimum_is_refused_before_anything_is_written(tmp_path):
    """One pixel below the minimum raises a ValueError naming long_side_px and the minimum, and the folder is as it was."""
    base = saved_class_run(tmp_path, SQUARE, 11)
    before = folder_state(tmp_path)
    with pytest.raises(ValueError, match="long_side_px") as refused:
        write_still_images(base, 10, long_side_px=stills.MIN_LONG_SIDE_PX - 1)
    assert str(stills.MIN_LONG_SIDE_PX) in str(refused.value)
    assert folder_state(tmp_path) == before


def test_exactly_the_minimum_size_is_drawn(tmp_path):
    """The minimum itself is accepted: a PNG is written for every shown step."""
    base = saved_class_run(tmp_path, SQUARE, 11)
    paths = write_still_images(base, 10, long_side_px=stills.MIN_LONG_SIDE_PX)
    assert [path.name for path in paths] == ["run_step_10.png", "run_step_11.png"]
    assert all(path.read_bytes().startswith(b"\x89PNG") for path in paths)
