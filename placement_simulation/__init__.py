# This folder holds the particle placement simulator. It places the particles
# of a measured or generated size list on a slate (a square, a rectangle or a
# circle, in micrometres), either at random or each at the most central free
# spot, saves the result and replays it. The public names are:
# place_particles, which returns an in-memory result and writes nothing;
# check_placement_destinations, which checks the two output paths before a
# long run; write_placement_outputs, which writes the placements CSV and the
# summary together; and replay_placement, which repeats a saved run from its
# summary.

from placement_simulation.placement import place_particles
from placement_simulation.saving import check_placement_destinations, write_placement_outputs
from placement_simulation.replay import replay_placement
