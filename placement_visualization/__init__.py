# This folder draws pictures of a saved particle placement run: still images of
# the slate as it fills, a time-lapse video and a packing map. Each
# function reads the two files placement_simulation saved and writes new files
# beside them; it never changes the saved files. The public names so far are
# write_still_images, which writes one PNG for every chosen step,
# write_video, which writes an mp4 time-lapse of the same steps in a slate-only
# or a dashboard layout, write_packing_map, which writes one PNG showing how
# much of the slate the placed particles cover, and write_packing_map_series,
# which writes that map at several blur widths and a grid of them all.

from placement_visualization.stills import write_still_images
from placement_visualization.packing_map import write_packing_map, write_packing_map_series
from placement_visualization.video import write_video
