# This folder draws pictures of a saved particle placement run: still images of
# the slate as it fills, and later a time-lapse video and a packing map. Each
# function reads the two files placement_simulation saved and writes new files
# beside them; it never changes the saved files. The public name so far is
# write_still_images, which writes one PNG for every chosen step.

from placement_visualization.stills import write_still_images
