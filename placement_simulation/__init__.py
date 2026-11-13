# This folder holds the particle placement simulator. It places the particles
# of a measured or generated size list on a slate (a square, a rectangle or a
# circle, in micrometres), saves the result and replays it. Later work adds
# the placement modes, the saved files and the replay to this package. The
# public interface is place_particles, which returns an in-memory result, the
# two functions that check for and write the saved files, and replay_placement,
# which repeats a saved run from its summary.

from placement_simulation.placement import place_particles
from placement_simulation.saving import check_placement_destinations, write_placement_outputs
from placement_simulation.replay import replay_placement
