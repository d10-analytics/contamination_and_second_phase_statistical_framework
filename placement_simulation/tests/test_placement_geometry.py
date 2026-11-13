# This file checks the slate geometry and the shared fit test with explicit
# numbers: slate validation, areas, edge distance, touching contacts, the edge
# reason winning over overlap, array and float agreement, that the candidate
# test uses the shared clearance, and the spread of random points. It also
# checks the folder rules: the allowed imports, the test file names and the
# absence of an __init__.py file in the tests folder.

import ast
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from placement_simulation import geometry

SIMULATOR_FOLDER = Path(__file__).resolve().parent.parent
SQUARE = {"shape": "square", "side_um": 2000.0}
RECTANGLE = {"shape": "rectangle", "width_um": 400.0, "height_um": 100.0}
CIRCLE = {"shape": "circle", "diameter_um": 2000.0}
NO_X = np.array([])
NO_Y = np.array([])
NO_R = np.array([])


def refusal_message(slate):
    """Return the ValueError text check_slate gives for a slate, failing if it accepts it."""
    with pytest.raises(ValueError) as caught:
        geometry.check_slate(slate)
    return str(caught.value)


def test_check_slate_accepts_each_shape_and_returns_floats():
    """Each shape is accepted and comes back as a new dictionary with float dimensions."""
    square = {"shape": "square", "side_um": 10}
    checked = geometry.check_slate(square)
    assert checked == {"shape": "square", "side_um": 10.0}
    assert isinstance(checked["side_um"], float)
    assert checked is not square
    rectangle = geometry.check_slate({"shape": "rectangle", "width_um": 4, "height_um": 2.5})
    assert rectangle == {"shape": "rectangle", "width_um": 4.0, "height_um": 2.5}
    assert isinstance(rectangle["width_um"], float)
    circle = geometry.check_slate({"shape": "circle", "diameter_um": np.float64(8)})
    assert circle == {"shape": "circle", "diameter_um": 8.0}
    assert geometry.check_slate(circle) == circle


def test_check_slate_refuses_a_non_dictionary_and_unknown_shape():
    """A list, a missing shape and an unknown shape are refused with a message naming the shape."""
    assert "dictionary" in refusal_message([("shape", "square")])
    assert "shape" in refusal_message({"side_um": 5})
    assert "hexagon" in refusal_message({"shape": "hexagon", "side_um": 5})


def test_check_slate_refuses_missing_and_extra_keys():
    """A missing key and an extra key are refused and the message names the key."""
    assert "side_um" in refusal_message({"shape": "square"})
    assert "height_um" in refusal_message({"shape": "rectangle", "width_um": 4.0})
    assert "diameter_um" in refusal_message({"shape": "circle", "side_um": 4.0})
    assert "colour" in refusal_message({"shape": "square", "side_um": 4.0, "colour": 1})
    assert "side_um" in refusal_message({"shape": "circle", "diameter_um": 4.0, "side_um": 4.0})


def test_check_slate_refuses_bad_dimensions():
    """A boolean, a text, a missing number, a non-finite value, zero and a negative are refused."""
    bad_values = [True, "5", None, float("nan"), float("inf"), 0, 0.0, -3.0]
    for value in bad_values:
        message = refusal_message({"shape": "rectangle", "width_um": 4.0, "height_um": value})
        assert "rectangle" in message and "height_um" in message, repr(value)
    assert "diameter_um" in refusal_message({"shape": "circle", "diameter_um": np.bool_(True)})


def test_slate_area_for_each_shape():
    """The area is side squared, width times height and pi times radius squared."""
    assert geometry.slate_area({"shape": "square", "side_um": 5.0}) == 25.0
    assert geometry.slate_area({"shape": "rectangle", "width_um": 4.0, "height_um": 2.5}) == 10.0
    assert geometry.slate_area({"shape": "circle", "diameter_um": 4.0}) == pytest.approx(4.0 * math.pi)


def test_edge_distance_is_positive_inside_and_negative_outside():
    """Edge distance uses the smaller half-extent minus |x| and |y|, or radius minus centre distance."""
    assert geometry.edge_distance({"shape": "square", "side_um": 10.0}, 1.0, -2.0) == 3.0
    assert geometry.edge_distance({"shape": "square", "side_um": 10.0}, 6.0, 0.0) == -1.0
    assert geometry.edge_distance({"shape": "rectangle", "width_um": 20.0, "height_um": 10.0}, 3.0, 4.0) == 1.0
    assert geometry.edge_distance({"shape": "rectangle", "width_um": 20.0, "height_um": 10.0}, 9.0, 0.0) == 1.0
    assert geometry.edge_distance({"shape": "circle", "diameter_um": 20.0}, 3.0, 4.0) == 5.0
    assert geometry.edge_distance({"shape": "circle", "diameter_um": 20.0}, 12.0, 5.0) == -3.0


def test_array_and_float_calls_agree():
    """Calling edge_distance and clearance on arrays equals calling them point by point."""
    xs = np.array([-3.0, 0.5, 7.0])
    ys = np.array([2.0, -4.5, 1.0])
    for slate in [SQUARE, RECTANGLE, CIRCLE]:
        together = geometry.edge_distance(slate, xs, ys)
        for index in range(3):
            single = geometry.edge_distance(slate, float(xs[index]), float(ys[index]))
            assert together[index] == single
    radii = np.array([3.0, 4.0, 5.5])
    together = geometry.clearance(1.0, 2.0, xs, ys, radii, 0.5)
    for index in range(3):
        single = geometry.clearance(1.0, 2.0, float(xs[index]), float(ys[index]), float(radii[index]), 0.5)
        assert together[index] == single


def test_clearance_is_distance_minus_radius_minus_buffer():
    """A 3-4-5 triangle gives distance 5, so the gap is 5 - 1 - 0.5."""
    assert geometry.clearance(3.0, 4.0, 0.0, 0.0, 1.0, 0.5) == 3.5


def classify_pair(distance, buffer_um):
    """Classify a radius 4 candidate at the given distance from a radius 3 particle at the origin."""
    placed_x = np.array([0.0])
    placed_y = np.array([0.0])
    placed_r = np.array([3.0])
    return geometry.classify_candidate(SQUARE, distance, 0.0, 4.0, placed_x, placed_y, placed_r, buffer_um)


def test_touching_particles_fit_and_closer_ones_overlap():
    """Radii 3 and 4 with buffer 1: centre distance 8 fits, 7.5 overlaps; with no buffer 7 fits."""
    assert classify_pair(8.0, 1.0) is None
    assert classify_pair(7.5, 1.0) == "overlap"
    assert classify_pair(9.0, 1.0) is None
    assert classify_pair(7.0, 0.0) is None
    assert classify_pair(6.5, 0.0) == "overlap"


def test_edge_touching_fits_and_the_buffer_does_not_apply_at_the_edge():
    """A radius 4 particle with edge distance exactly 4 fits, also with a buffer; 3.5 crosses."""
    slate = {"shape": "square", "side_um": 20.0}
    for buffer_um in [0.0, 1.0]:
        assert geometry.classify_candidate(slate, 6.0, 0.0, 4.0, NO_X, NO_Y, NO_R, buffer_um) is None
        assert geometry.classify_candidate(slate, 6.5, 0.0, 4.0, NO_X, NO_Y, NO_R, buffer_um) == "edge"
    circle = {"shape": "circle", "diameter_um": 20.0}
    assert geometry.classify_candidate(circle, 6.0, 0.0, 4.0, NO_X, NO_Y, NO_R, 1.0) is None
    assert geometry.classify_candidate(circle, 6.5, 0.0, 4.0, NO_X, NO_Y, NO_R, 1.0) == "edge"


def test_a_candidate_that_crosses_the_edge_and_overlaps_reports_edge():
    """When both problems apply, the reason is edge."""
    slate = {"shape": "square", "side_um": 20.0}
    placed_x = np.array([8.0])
    placed_y = np.array([0.0])
    placed_r = np.array([3.0])
    reason = geometry.classify_candidate(slate, 9.0, 0.0, 4.0, placed_x, placed_y, placed_r, 0.0)
    assert reason == "edge"
    reason = geometry.classify_candidate(slate, 0.0, 0.0, 4.0, placed_x, placed_y, placed_r, 0.0)
    assert reason is None
    reason = geometry.classify_candidate(slate, 3.0, 0.0, 4.0, placed_x, placed_y, placed_r, 0.0)
    assert reason == "overlap"


def test_overlap_with_any_one_of_several_placed_particles_is_found():
    """Only the second of three placed particles is too close, and that is enough."""
    placed_x = np.array([-40.0, 6.0, 40.0])
    placed_y = np.array([0.0, 0.0, 0.0])
    placed_r = np.array([3.0, 3.0, 3.0])
    reason = geometry.classify_candidate(SQUARE, 0.0, 0.0, 4.0, placed_x, placed_y, placed_r, 0.0)
    assert reason == "overlap"


def test_classify_candidate_uses_the_shared_clearance(monkeypatch):
    """With clearance 0.5 smaller, the touching pair (distance 8) is classified as overlap."""
    assert classify_pair(8.0, 1.0) is None
    original = geometry.clearance

    def smaller_clearance(x, y, placed_x, placed_y, placed_radius, buffer_um):
        """Return the shared clearance reduced by 0.5 micrometres."""
        return original(x, y, placed_x, placed_y, placed_radius, buffer_um) - 0.5

    monkeypatch.setattr(geometry, "clearance", smaller_clearance)
    assert classify_pair(8.0, 1.0) == "overlap"


def draw_points(slate, count, seed):
    """Draw count random points on the slate and return their x and y values as two arrays."""
    rng = np.random.default_rng(seed)
    xs = np.empty(count)
    ys = np.empty(count)
    for index in range(count):
        xs[index], ys[index] = geometry.random_point(slate, rng)
    return xs, ys


def test_random_points_on_a_square_reach_the_edge_strip():
    """About 12% of points on a 2 mm square lie within 60 um of the edge, so drawing is over the whole slate."""
    xs, ys = draw_points(SQUARE, 4000, 1)
    assert np.all(geometry.edge_distance(SQUARE, xs, ys) >= 0)
    share = np.mean(geometry.edge_distance(SQUARE, xs, ys) < 60.0)
    assert 0.09 < share < 0.14


def test_random_points_on_a_circle_are_uniform_in_area():
    """All points lie inside the circle and the share within R/sqrt(2) of the centre is near one half."""
    xs, ys = draw_points(CIRCLE, 4000, 2)
    assert np.all(geometry.edge_distance(CIRCLE, xs, ys) >= 0)
    share = np.mean(np.sqrt(xs * xs + ys * ys) <= 1000.0 / math.sqrt(2.0))
    assert 0.46 < share < 0.54


def test_random_points_on_a_rectangle_fill_the_corners():
    """All points lie inside a 400 x 100 um rectangle and some are far out in both directions."""
    xs, ys = draw_points(RECTANGLE, 4000, 3)
    assert np.all(np.abs(xs) <= 200.0)
    assert np.all(np.abs(ys) <= 50.0)
    assert np.any((np.abs(xs) > 150.0) & (np.abs(ys) > 40.0))


def test_random_points_are_repeatable_for_one_seed():
    """The same seed gives the same points."""
    first = draw_points(SQUARE, 5, 9)
    second = draw_points(SQUARE, 5, 9)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])


def imported_module_names(path):
    """Return the top-level module name of every import in a Python file."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        if isinstance(node, ast.ImportFrom) and node.level == 0:
            names.add(node.module.split(".")[0])
    return names


def test_every_simulator_file_imports_only_the_allowed_modules():
    """Simulator files import only the standard library, numpy, scipy and the two packages; tests may add pytest."""
    allowed = {"numpy", "scipy", "synthetic_data_generation", "placement_simulation"}
    files = sorted(SIMULATOR_FOLDER.rglob("*.py"))
    assert SIMULATOR_FOLDER / "geometry.py" in files
    for path in files:
        is_test = "tests" in path.relative_to(SIMULATOR_FOLDER).parts
        for name in imported_module_names(path):
            if name in sys.stdlib_module_names or name in allowed:
                continue
            assert is_test and name == "pytest", str(path) + " imports " + name


def test_import_scan_would_refuse_a_forbidden_module(tmp_path):
    """The scan reads real imports: a file importing cv2 and pandas reports both names."""
    sample = tmp_path / "sample.py"
    sample.write_text("import cv2\nfrom pandas import DataFrame\nimport os.path\n", encoding="utf-8")
    assert imported_module_names(sample) == {"cv2", "pandas", "os"}


def test_test_files_are_named_for_the_simulator_and_have_no_init_file():
    """Every test file is named test_placement_*.py and the tests folder has no __init__.py."""
    tests_folder = SIMULATOR_FOLDER / "tests"
    names = sorted(path.name for path in tests_folder.glob("test_*.py"))
    assert "test_placement_geometry.py" in names
    for name in names:
        assert name.startswith("test_placement_"), name
    assert not (tests_folder / "__init__.py").exists()
