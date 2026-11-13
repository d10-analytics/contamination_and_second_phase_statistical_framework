# This file runs one placement: it reads the particle sizes, checks every
# setting, prints the requested coverage and places the particles one at a
# time. The result is an in-memory record; nothing is written to disk here.
# The input step (particles to records plus an identity that lets a run be
# repeated) and the settings check are separate functions so that replaying a
# saved run uses exactly the same code. Random placement uses two independent
# random streams made from one seed: the first decides the order, the second
# draws the candidate locations, so changing the order never moves a location.
# Centre placement makes one search per particle on a clearance map (see
# centre_search.py) and never uses the location stream.
# Coordinates are in micrometres, the origin at the slate centre, x to the
# right and y up.

import hashlib
import math
import os
import secrets
import sys
from dataclasses import dataclass

import numpy as np

from placement_simulation import centre_search, geometry
from synthetic_data_generation import dataset, outputs

MODES = ("random", "centre")
ORDERS = ("as_given", "shuffled")


@dataclass(frozen=True)
class Particle:
    """One input particle: its 1-based data row, its labels and its true size."""

    source_index: int
    object_id: str
    feature_label: str
    equivalent_diameter_um: float
    radius_um: float


@dataclass(frozen=True)
class PlacementRow:
    """The outcome for one particle; the field names are the placements-file columns."""

    placement_order: int
    source_index: int
    object_id: str
    feature_label: str
    equivalent_diameter_um: float
    placed: bool
    x_um: object
    y_um: object
    tries: int
    edge_rejections: int
    overlap_rejections: int


@dataclass(frozen=True)
class PlacementResult:
    """A finished run: the settings (named as in the summary's settings block), the coverages and one row per particle."""

    slate: dict
    mode: str
    max_tries: int
    buffer_um: float
    pixel_um: float
    order: str
    seed: int
    input: dict
    requested_coverage: float
    achieved_coverage: float
    rows: tuple


def make_particles(rows):
    """Return a Particle for each row dictionary, numbering the rows from 1."""
    particles = []
    for position, row in enumerate(rows, start=1):
        diameter = float(row["equivalent_diameter_um"])
        particles.append(
            Particle(
                source_index=position,
                object_id=row["object_id"],
                feature_label=row["feature_label"],
                equivalent_diameter_um=diameter,
                radius_um=diameter / 2.0,
            )
        )
    return particles


def read_particles(particles):
    """Return (the particles as records, the input identity) for a generator result or a measurements CSV path.

    A CSV is read once: the bytes that are hashed are the bytes that are parsed.
    Anything that is neither raises ValueError.
    """
    if isinstance(particles, dataset.DatasetResult):
        digest = hashlib.sha256(outputs.serialize_csv(particles).encode("utf-8")).hexdigest()
        identity = {
            "kind": "generator",
            "generator_parameters": outputs.reproduction_parameters(particles),
            "measurements_sha256": digest,
        }
        rows = [{"object_id": row.object_id, "feature_label": row.feature_label,
                 "equivalent_diameter_um": row.equivalent_diameter_um} for row in particles.rows]
        return make_particles(rows), identity
    if isinstance(particles, (str, os.PathLike)):
        if not isinstance(os.fspath(particles), str):
            raise ValueError("particles must be a generator result or the path of a measurements CSV file, got " + repr(particles))
        data, rows = outputs.load_measurements(particles)
        identity = {
            "kind": "csv",
            "file_name": os.path.basename(os.fspath(particles)),
            "measurements_sha256": hashlib.sha256(data).hexdigest(),
        }
        return make_particles(rows), identity
    raise ValueError("particles must be a generator result or the path of a measurements CSV file, got " + repr(particles))


def check_number(name, value, allow_zero):
    """Return value as a float when it is a finite number (above 0, or 0 or more when allow_zero); else raise ValueError."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(name + " must be a number, got " + repr(value))
    try:
        number = float(value)
    except OverflowError:
        number = math.inf
    if not math.isfinite(number):
        raise ValueError(name + " must be finite, got " + repr(value))
    if number < 0 or (number == 0 and not allow_zero):
        limit = "0 or more" if allow_zero else "above 0"
        raise ValueError(name + " must be " + limit + ", got " + repr(value))
    return number


def check_settings(slate, mode, max_tries, buffer_um, pixel_um, order, seed):
    """Check every setting and return them as a dictionary named as in the summary's settings block.

    A seed of None is passed through unchanged; the caller draws one. Raises
    ValueError naming the first setting that is refused.
    """
    checked_slate = geometry.check_slate(slate)
    if not isinstance(mode, str) or mode not in MODES:
        raise ValueError("mode must be one of " + ", ".join(MODES) + ", got " + repr(mode))
    if isinstance(max_tries, bool) or not isinstance(max_tries, int) or max_tries < 1:
        raise ValueError("max_tries must be a whole number of 1 or more, not " + repr(max_tries))
    if mode == "centre" and max_tries > 1:
        raise ValueError("mode centre makes one search per particle, so max_tries must be 1, got " + str(max_tries))
    checked_buffer = check_number("buffer_um", buffer_um, True)
    checked_pixel = check_number("pixel_um", pixel_um, False)
    if not isinstance(order, str) or order not in ORDERS:
        raise ValueError("order must be one of " + ", ".join(ORDERS) + ", got " + repr(order))
    checked_seed = None if seed is None else dataset.validate_seed(seed)
    return {
        "slate": checked_slate,
        "mode": mode,
        "max_tries": max_tries,
        "buffer_um": checked_buffer,
        "pixel_um": checked_pixel,
        "order": order,
        "seed": checked_seed,
    }


def particle_area(diameter):
    """Return the true area of a circle of this diameter."""
    return math.pi * diameter * diameter / 4.0


def requested_coverage(particles, slate):
    """Return the total particle area divided by the slate area (0.0 with no particles)."""
    total = 0.0
    for particle in particles:
        total += particle_area(particle.equivalent_diameter_um)
    return total / geometry.slate_area(slate)


def order_particles(particles, order, order_stream):
    """Return the particles in placement order; only a shuffle draws from the order stream."""
    if order == "as_given":
        return list(particles)
    permutation = order_stream.permutation(len(particles))
    return [particles[int(position)] for position in permutation]


def place_randomly(ordered, slate, max_tries, buffer_um, location_stream):
    """Place each particle at its first fitting random candidate, up to max_tries tries; return the rows."""
    count = len(ordered)
    # The placed particles are the record the fit test reads; only the filled part is passed on.
    placed_x = np.zeros(count)
    placed_y = np.zeros(count)
    placed_radius = np.zeros(count)
    filled = 0
    rows = []
    for number, particle in enumerate(ordered, start=1):
        edge_rejections = 0
        overlap_rejections = 0
        found = None
        for _ in range(max_tries):
            x, y = geometry.random_point(slate, location_stream)
            reason = geometry.classify_candidate(
                slate,
                x,
                y,
                particle.radius_um,
                placed_x[:filled],
                placed_y[:filled],
                placed_radius[:filled],
                buffer_um,
            )
            if reason is None:
                found = (x, y)
                break
            if reason == "edge":
                edge_rejections += 1
            else:
                overlap_rejections += 1
        if found is not None:
            placed_x[filled], placed_y[filled], placed_radius[filled] = found[0], found[1], particle.radius_um
            filled += 1
        rows.append(
            PlacementRow(
                placement_order=number,
                source_index=particle.source_index,
                object_id=particle.object_id,
                feature_label=particle.feature_label,
                equivalent_diameter_um=particle.equivalent_diameter_um,
                placed=found is not None,
                x_um=None if found is None else float(found[0]),
                y_um=None if found is None else float(found[1]),
                tries=edge_rejections + overlap_rejections + (0 if found is None else 1),
                edge_rejections=edge_rejections,
                overlap_rejections=overlap_rejections,
            )
        )
    return rows


def place_centrally(ordered, slate, buffer_um, pixel_um):
    """Place each particle at the grid point nearest the slate centre where it fits; return the rows.

    One search per particle. A particle with no fitting grid point is not
    placed and the run goes on. Every row has one try and no rejections.
    """
    max_radius = 0.0
    for particle in ordered:
        max_radius = max(max_radius, particle.radius_um)
    searcher = centre_search.ClearanceSearch(slate, buffer_um, pixel_um, max_radius)
    rows = []
    for number, particle in enumerate(ordered, start=1):
        found = searcher.find(particle.radius_um)
        x = None
        y = None
        if found is not None:
            searcher.add(found[0], found[1], particle.radius_um)
            x = float(found[0] * pixel_um)
            y = float(found[1] * pixel_um)
        rows.append(
            PlacementRow(
                placement_order=number,
                source_index=particle.source_index,
                object_id=particle.object_id,
                feature_label=particle.feature_label,
                equivalent_diameter_um=particle.equivalent_diameter_um,
                placed=found is not None,
                x_um=x,
                y_um=y,
                tries=1,
                edge_rejections=0,
                overlap_rejections=0,
            )
        )
    return rows


def achieved_coverage(rows, slate):
    """Return the placed particles' true area divided by the slate area."""
    total = 0.0
    for row in rows:
        if row.placed:
            total += particle_area(row.equivalent_diameter_um)
    return total / geometry.slate_area(slate)


def place_records(records, identity, settings):
    """Place already-read particle records and return a PlacementResult; nothing is written to disk.

    settings is the dictionary check_settings returns, with a seed drawn. The
    requested coverage is printed on stderr before anything is placed.
    """
    requested = requested_coverage(records, settings["slate"])
    print(f"requested coverage: {requested!r} (total particle area / slate area)", file=sys.stderr)
    order_child, location_child = np.random.SeedSequence(settings["seed"]).spawn(2)
    ordered = order_particles(records, settings["order"], np.random.default_rng(order_child))
    if settings["mode"] == "centre":
        rows = place_centrally(ordered, settings["slate"], settings["buffer_um"], settings["pixel_um"])
    else:
        rows = place_randomly(
            ordered, settings["slate"], settings["max_tries"], settings["buffer_um"], np.random.default_rng(location_child)
        )
    return PlacementResult(
        slate=settings["slate"],
        mode=settings["mode"],
        max_tries=settings["max_tries"],
        buffer_um=settings["buffer_um"],
        pixel_um=settings["pixel_um"],
        order=settings["order"],
        seed=settings["seed"],
        input=identity,
        requested_coverage=requested,
        achieved_coverage=achieved_coverage(rows, settings["slate"]),
        rows=tuple(rows),
    )


def place_particles(particles, slate, *, mode="random", max_tries=1, buffer_um=0.0, pixel_um=1.0, order="as_given", seed=None):
    """Place the particles on the slate and return a PlacementResult; nothing is written to disk.

    particles is a generator DatasetResult or the path of a measurements CSV.
    Every setting is checked before the requested coverage is printed on
    stderr and before anything is placed. With no seed one is drawn and
    recorded. Raises ValueError for a refused setting or input.
    """
    settings = check_settings(slate, mode, max_tries, buffer_um, pixel_um, order, seed)
    records, identity = read_particles(particles)
    if settings["seed"] is None:
        settings["seed"] = secrets.randbits(dataset.SEED_BITS)
    return place_records(records, identity, settings)
