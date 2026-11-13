# This file finds the grid point nearest the slate centre where a particle
# fits. The grid points are (i * pixel_um, j * pixel_um) for whole numbers i
# and j. A square window of cells around the centre holds the clearance map:
# each cell is min(Rmax, distance to the slate edge, gap to every placed
# particle's buffered surface), where Rmax is the largest radius in the input.
# A particle of radius r fits at a cell whose value is at least r. After a
# particle is placed, every cell within Rmax + r + buffer_um of it is lowered;
# a cell farther away cannot fall below Rmax. The window only covers the region
# the cluster has reached: it is kept at least as wide as the cluster's outer
# reach plus buffer_um + Rmax + pixel_um, and it grows to the larger of that
# bound and twice its width, never beyond the slate. Cells added by growth start
# at min(Rmax, edge distance), which is exact because they lie beyond every
# earlier particle's reach. The nearest fit is chosen by the integer i*i + j*j,
# then the largest j, then the smallest i, so the choice depends on the
# coordinates and never on the array layout or the window size.

import math

import numpy as np

from placement_simulation import geometry

FIRST_BOX_CELLS = 16


class ClearanceSearch:
    """The clearance map over a growing window and the nearest-fit search on it.

    window_half_widths lists the window's (x, y) half-extents in micrometres
    after clipping to the slate: the starting window first, then one entry per
    growth.
    """

    def __init__(self, slate, buffer_um, pixel_um, max_radius):
        """Start the window at buffer_um + max_radius + pixel_um, rounded outward to whole cells and clipped."""
        self.slate = slate
        self.buffer_um = buffer_um
        self.pixel_um = pixel_um
        self.max_radius = max_radius
        if slate["shape"] == "circle":
            self.half_x = slate["diameter_um"] / 2.0
            self.half_y = self.half_x
        else:
            self.half_x, self.half_y = geometry.half_extents(slate)
        self.limit_x = int(math.floor(self.half_x / pixel_um))
        self.limit_y = int(math.floor(self.half_y / pixel_um))
        self.reach = 0.0
        self.width_cells = int(math.ceil((buffer_um + max_radius + pixel_um) / pixel_um))
        self.window_half_widths = []
        self.cells_x = 0
        self.cells_y = 0
        self.values = None
        self.resize()

    def resize(self):
        """Set the window to width_cells clipped to the slate and rebuild the map, keeping the old cells."""
        new_x = min(self.width_cells, self.limit_x)
        new_y = min(self.width_cells, self.limit_y)
        if self.values is not None and new_x == self.cells_x and new_y == self.cells_y:
            return
        xs = np.arange(-new_x, new_x + 1) * self.pixel_um
        ys = np.arange(-new_y, new_y + 1) * self.pixel_um
        values = np.minimum(self.max_radius, geometry.edge_distance(self.slate, xs[:, None], ys[None, :]))
        if self.values is not None:
            low_x = new_x - self.cells_x
            low_y = new_y - self.cells_y
            values[low_x : low_x + 2 * self.cells_x + 1, low_y : low_y + 2 * self.cells_y + 1] = self.values
        self.values = values
        self.cells_x = new_x
        self.cells_y = new_y
        width_um = self.width_cells * self.pixel_um
        self.window_half_widths.append((min(width_um, self.half_x), min(width_um, self.half_y)))

    def find(self, radius):
        """Return the grid offsets (i, j) of the nearest fit for a particle of this radius, or None.

        The search looks at a box of growing size around the centre. A fit
        is accepted once its i*i + j*j is within the box's own radius,
        because every grid point that near lies inside the box.
        """
        reach = FIRST_BOX_CELLS
        while True:
            box_x = min(reach, self.cells_x)
            box_y = min(reach, self.cells_y)
            whole = box_x == self.cells_x and box_y == self.cells_y
            block = self.values[
                self.cells_x - box_x : self.cells_x + box_x + 1, self.cells_y - box_y : self.cells_y + box_y + 1
            ]
            fits = block >= radius
            if fits.any():
                squares = (np.arange(-box_x, box_x + 1, dtype=np.int64) ** 2)[:, None] + (
                    np.arange(-box_y, box_y + 1, dtype=np.int64) ** 2
                )[None, :]
                nearest = int(squares[fits].min())
                if nearest <= reach * reach or whole:
                    return self.break_tie(fits & (squares == nearest), box_x, box_y)
            if whole:
                return None
            reach = reach * 2

    def break_tie(self, tied, box_x, box_y):
        """Return (i, j) for the tied cell with the largest j, then the smallest i."""
        found_x, found_y = np.nonzero(tied)
        best = None
        for index_x, index_y in zip(found_x, found_y):
            i = int(index_x) - box_x
            j = int(index_y) - box_y
            key = (-j, i)
            if best is None or key < best[0]:
                best = (key, i, j)
        return best[1], best[2]

    def add(self, i, j, radius):
        """Record a particle placed at grid offsets (i, j): grow the window if needed, then lower the cells it reaches."""
        x = i * self.pixel_um
        y = j * self.pixel_um
        self.reach = max(self.reach, math.hypot(x, y) + radius)
        bound = self.reach + self.buffer_um + self.max_radius + self.pixel_um
        if bound > self.width_cells * self.pixel_um:
            self.width_cells = max(int(math.ceil(bound / self.pixel_um)), 2 * self.width_cells)
            self.resize()
        self.lower(x, y, radius)

    def lower(self, x, y, radius):
        """Lower every cell within max_radius + radius + buffer_um of the particle (rounded outward) to its gap."""
        span = self.max_radius + radius + self.buffer_um
        low_i = max(int(math.floor((x - span) / self.pixel_um)) - 1, -self.cells_x)
        high_i = min(int(math.ceil((x + span) / self.pixel_um)) + 1, self.cells_x)
        low_j = max(int(math.floor((y - span) / self.pixel_um)) - 1, -self.cells_y)
        high_j = min(int(math.ceil((y + span) / self.pixel_um)) + 1, self.cells_y)
        if low_i > high_i or low_j > high_j:
            return
        xs = np.arange(low_i, high_i + 1) * self.pixel_um
        ys = np.arange(low_j, high_j + 1) * self.pixel_um
        block = self.values[
            low_i + self.cells_x : high_i + self.cells_x + 1, low_j + self.cells_y : high_j + self.cells_y + 1
        ]
        gaps = geometry.clearance(xs[:, None], ys[None, :], x, y, radius, self.buffer_um)
        np.minimum(block, gaps, out=block)
