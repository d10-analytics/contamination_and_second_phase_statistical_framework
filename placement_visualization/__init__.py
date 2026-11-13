# This folder draws pictures of a saved particle placement run: still images of
# the slate as it fills, and later a time-lapse video and a packing map. Each
# function reads the two files placement_simulation saved and writes new files
# beside them; it never changes the saved files. The public names so far are
# write_still_images, which writes one PNG for every chosen step, and
# write_packing_map, which writes one PNG showing how much of the slate the
# placed particles cover.

from placement_visualization.stills import write_still_images
from placement_visualization.packing_map import write_packing_map
