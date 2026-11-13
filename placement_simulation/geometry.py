# This file holds the slate shapes and the one fit test the simulator uses.
# The slate is a plain dictionary with its centre at (0, 0), in micrometres:
# a square (side_um), a rectangle (width_um and height_um) or a circle
# (diameter_um). A particle is a circle and must lie entirely inside the slate;
# it may touch the edge. Two particles may touch, and the optional buffer is
# the smallest gap allowed between their surfaces. The function "clearance" is
# the only place that writes the distance arithmetic, so every fit decision in
# the package agrees with it to the last rounding step.

import math

import numpy as np

SLATE_KEYS = {
    "square": ["side_um"],
    "rectangle": ["width_um", "height_um"],
    "circle": ["diameter_um"],
}


def check_dimension(shape, key, value):
    """Return the dimension as a float, or raise ValueError when it is not a number above 0."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("slate " + shape + " " + key + " must be a number, not a boolean")
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError("slate " + shape + " " + key + " must be a number, got " + repr(value))
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("slate " + shape + " " + key + " must be finite, got " + repr(value))
    if not number > 0:
        raise ValueError("slate " + shape + " " + key + " must be above 0, got " + repr(value))
    return number


def check_slate(slate):
    """Validate a slate dictionary and return a new one with float dimensions."""
    if not isinstance(slate, dict):
        raise ValueError("the slate must be a dictionary, got " + repr(slate))
    shape = slate.get("shape")
    if not isinstance(shape, str) or shape not in SLATE_KEYS:
        raise ValueError("the slate shape must be square, rectangle or circle, got " + repr(shape))
    keys = SLATE_KEYS[shape]
    for key in keys:
        if key not in slate:
            raise ValueError("the " + shape + " slate is missing the key " + key)
    for key in slate:
        if key != "shape" and key not in keys:
            raise ValueError("the " + shape + " slate has an unexpected key " + repr(key))
    checked = {"shape": shape}
    for key in keys:
        checked[key] = check_dimension(shape, key, slate[key])
    return checked


def half_extents(slate):
    """Return the half width and half height of a square or rectangle slate."""
    if slate["shape"] == "square":
        half = slate["side_um"] / 2.0
        return half, half
    return slate["width_um"] / 2.0, slate["height_um"] / 2.0


def slate_area(slate):
    """Return the slate area in square micrometres."""
    if slate["shape"] == "square":
        return slate["side_um"] * slate["side_um"]
    if slate["shape"] == "rectangle":
        return slate["width_um"] * slate["height_um"]
    radius = slate["diameter_um"] / 2.0
    return math.pi * radius * radius


def edge_distance(slate, x, y):
    """Return the distance from a point to the slate edge, positive inside (floats or arrays)."""
    if slate["shape"] == "circle":
        radius = slate["diameter_um"] / 2.0
        return radius - np.sqrt(x * x + y * y)
    half_width, half_height = half_extents(slate)
    return np.minimum(half_width - np.abs(x), half_height - np.abs(y))


def clearance(x, y, placed_x, placed_y, placed_radius, buffer_um):
    """Return the gap between a point and the buffered surface of placed particles."""
    dx = x - placed_x
    dy = y - placed_y
    return np.sqrt(dx * dx + dy * dy) - placed_radius - buffer_um


def classify_candidate(slate, x, y, radius, placed_x, placed_y, placed_radius, buffer_um):
    """Return None when the circle fits, "edge" when it crosses the edge, else "overlap"."""
    if edge_distance(slate, x, y) < radius:
        return "edge"
    gaps = clearance(x, y, placed_x, placed_y, placed_radius, buffer_um)
    if np.any(gaps < radius):
        return "overlap"
    return None


def random_point(slate, rng):
    """Return a point (x, y) uniform over the whole slate, whatever the particle size."""
    if slate["shape"] == "circle":
        radius = slate["diameter_um"] / 2.0
        distance = radius * math.sqrt(rng.random())
        angle = 2.0 * math.pi * rng.random()
        return distance * math.cos(angle), distance * math.sin(angle)
    half_width, half_height = half_extents(slate)
    x = float(rng.uniform(-half_width, half_width))
    y = float(rng.uniform(-half_height, half_height))
    return x, y
