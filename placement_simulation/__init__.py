# This folder holds the particle placement simulator. It places the particles
# of a measured or generated size list on a slate (a square, a rectangle or a
# circle, in micrometres), saves the result and replays it. Later work adds
# the placement modes, the saved files and the replay to this package. The
# public interface so far is place_particles, which returns an in-memory result.

from placement_simulation.placement import place_particles
