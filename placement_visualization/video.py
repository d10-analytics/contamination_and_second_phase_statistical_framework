# This file writes the time-lapse video of a saved placement run. Each shown
# step (the same steps the still images use) is drawn once at 1920 x 1080. Its
# frame bytes are sent to matplotlib's ffmpeg writer, which encodes at the
# chosen frame rate, once for every frame the step is held. The holds are whole
# frames worked out here, so the video has exactly round(video_seconds x fps)
# frames (a half rounds up), shared as evenly as whole frames allow, and no
# step is drawn more than once. Everything that can be refused is refused
# before the ffmpeg lookup and before any output: the arguments, the saved
# files, the destination and a video too short to give every shown step one
# frame. ffmpeg is looked up when the video is written, never when the folder
# is imported, so a missing ffmpeg stops only this function. The video is
# encoded to a hidden temporary file and renamed into place, and the temporary
# file is always removed.
import io
import math
import os
import shutil
import sys
from pathlib import Path

import matplotlib
from matplotlib.animation import FFMpegWriter
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from placement_simulation import placement
from placement_visualization import dashboard, picture_files, slate_drawing, stills
from synthetic_data_generation import outputs

SCHEDULE_TOLERANCE = 1e-9
LAYOUTS = ("slate", "dashboard")
FRAME_WIDTH_PX = 1920
FRAME_HEIGHT_PX = 1080
FRAME_DPI = 100
BITRATE_KBPS = 8000
CODEC = "libopenh264"
LEGEND_ROOM = 0.1
INSTALL_GUIDANCE = (
    "ffmpeg was not found next to the running Python or on the PATH; install ffmpeg "
    "(with the libopenh264 encoder), for example with conda install ffmpeg, then try again"
)


def video_path(base_path, layout):
    """Return the destination <base>_<layout>.mp4 beside base_path."""
    base_path = Path(base_path)
    return base_path.parent / f"{base_path.name}_{layout}.mp4"


def frame_count(video_seconds, fps):
    """Return the number of frames the video has: video_seconds x fps rounded, a half rounding up."""
    return math.floor(video_seconds * fps + 0.5)


def frame_holds(shown_count, total_frames):
    """Return how many frames each shown step is held for: the k-th step ends at frame floor(k x total / shown), so the holds add up to total_frames and differ by at most one."""
    holds = []
    previous_end = 0
    for position in range(1, shown_count + 1):
        end = position * total_frames // shown_count
        holds.append(end - previous_end)
        previous_end = end
    return holds


def shown_step_count(particle_count, every):
    """Return how many shown steps every gives for the particle count."""
    return -(-particle_count // every)


def holds_one_frame_each(shown_count, frames_available):
    """Say whether frames_available (video_seconds x fps) gives every shown step a whole frame, ignoring float rounding error at an exact ratio."""
    return frames_available >= shown_count * (1.0 - SCHEDULE_TOLERANCE)


def smallest_passing_every(particle_count, video_seconds, fps):
    """Return the smallest every whose shown steps each get at least one frame, or None when none does."""
    frames_available = video_seconds * fps
    if not holds_one_frame_each(1, frames_available):
        return None
    candidate = max(1, -(-particle_count // max(1, frame_count(video_seconds, fps))))
    while candidate > 1 and holds_one_frame_each(shown_step_count(particle_count, candidate - 1), frames_available):
        candidate -= 1
    while not holds_one_frame_each(shown_step_count(particle_count, candidate), frames_available):
        candidate += 1
    return candidate


def check_schedule(shown, particle_count, every, video_seconds, fps):
    """Raise ValueError when a shown step would be held for less than one frame, naming the smallest every that passes."""
    frames_available = video_seconds * fps
    if holds_one_frame_each(len(shown), frames_available):
        return
    if not holds_one_frame_each(1, frames_available):
        raise ValueError(
            f"video_seconds={video_seconds!r} at fps={fps!r} is shorter than one frame, so no every can show a step; choose a longer video or a higher fps"
        )
    smallest = smallest_passing_every(particle_count, video_seconds, fps)
    raise ValueError(
        f"every={every} gives {len(shown)} shown steps, too many for {frames_available:g} frames (a shown step needs at least one); "
        f"use every={smallest} or a longer video"
    )


def find_ffmpeg(interpreter_folder, search_path):
    """Return the ffmpeg beside the interpreter when it is there, else the one on search_path; raise RuntimeError with install guidance when neither exists."""
    name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    beside = Path(interpreter_folder) / name
    if beside.is_file():
        return str(beside)
    found = shutil.which("ffmpeg", path=search_path)
    if found is not None:
        return found
    raise RuntimeError(INSTALL_GUIDANCE)


def build_slate_frame(result, step):
    """Return the slate-only 1920 x 1080 figure for one shown step: the slate, filled to the step, and the legend."""
    colours = slate_drawing.class_colours(result.rows)
    figure = Figure(figsize=(FRAME_WIDTH_PX / FRAME_DPI, FRAME_HEIGHT_PX / FRAME_DPI), dpi=FRAME_DPI, facecolor=slate_drawing.BACKGROUND)
    FigureCanvasAgg(figure)
    axes = figure.add_axes([0.02, LEGEND_ROOM + 0.02, 0.96, 1.0 - LEGEND_ROOM - 0.04])
    half_width, half_height = slate_drawing.slate_extent(result.slate)
    axes.set_xlim(-half_width, half_width)
    axes.set_ylim(-half_height, half_height)
    axes.set_aspect("equal")
    axes.set_axis_off()
    slate_drawing.draw_slate(axes, result.slate, result.rows[:step], colours)
    slate_drawing.draw_legend(figure, colours, LEGEND_ROOM * FRAME_HEIGHT_PX)
    return figure


class SlateFrames:
    """Builds the slate-only figure of a shown step."""

    def __init__(self, result):
        """Keep the run the frames are drawn from."""
        self.result = result

    def figure_for(self, step):
        """Return the figure for the shown step."""
        return build_slate_frame(self.result, step)


class DashboardFrames:
    """Builds the dashboard figure of a shown step from numbers worked out in advance."""

    def __init__(self, result, data):
        """Keep the run and the panel numbers for all shown steps."""
        self.result = result
        self.data = data

    def figure_for(self, step):
        """Return the figure for the shown step."""
        return dashboard.build_dashboard_figure(self.result, self.data, step, FRAME_WIDTH_PX, FRAME_HEIGHT_PX)


def make_frames(result, layout, shown):
    """Return the object that builds each shown step's figure for the layout."""
    if layout == "slate":
        return SlateFrames(result)
    return DashboardFrames(result, dashboard.panel_data(result, shown))


class HeldFrameWriter(FFMpegWriter):
    """The matplotlib ffmpeg writer with one addition: a drawn frame can be sent several times."""

    def grab_frame_held(self, hold):
        """Draw the current figure once, as grab_frame does, and send its bytes to ffmpeg hold times."""
        self.fig.set_size_inches(self._w, self._h)
        buffer = io.BytesIO()
        self.fig.savefig(buffer, format=self.frame_format, dpi=self.dpi)
        frame_bytes = buffer.getvalue()
        for _ in range(hold):
            self._proc.stdin.write(frame_bytes)


def encode(frames, shown, holds, fps, ffmpeg, temporary):
    """Draw each shown step once and send it to ffmpeg for its held frames; ffmpeg writes the mp4 at the temporary name."""
    with matplotlib.rc_context({"animation.ffmpeg_path": ffmpeg}):
        writer = HeldFrameWriter(
            fps=fps,
            codec=CODEC,
            bitrate=BITRATE_KBPS,
            extra_args=["-pix_fmt", "yuv420p", "-f", "mp4"],
        )
        first = frames.figure_for(shown[0])
        with writer.saving(first, str(temporary), FRAME_DPI):
            writer.grab_frame_held(holds[0])
            for step, hold in zip(shown[1:], holds[1:]):
                writer.fig = frames.figure_for(step)
                writer.grab_frame_held(hold)


def write_video(base_path, every, video_seconds, *, fps=30, layout="slate", overwrite=False):
    """Write <base>_<layout>.mp4 and return its path.

    The video lasts video_seconds and has round(video_seconds x fps) frames,
    a half rounding up; each shown step is held for an equal share, as near as
    whole frames allow (the holds differ by at most one frame). Arguments, the saved files, the
    destination, the schedule and the ffmpeg lookup are all checked before the
    first frame, each refusal a ValueError except a missing ffmpeg, which is a
    RuntimeError. A failure while encoding leaves no temporary file and any
    earlier video untouched.
    """
    every = picture_files.check_whole_number(every, "every")
    video_seconds = placement.check_number("video_seconds", video_seconds, False)
    fps_number = placement.check_number("fps", fps, False)
    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}, got {layout!r}")
    result = picture_files.load_placement_result(base_path)
    particle_count = len(result.rows)
    destination = video_path(base_path, layout)
    picture_files.check_destinations([destination], overwrite)
    shown = stills.shown_steps(particle_count, every)
    check_schedule(shown, particle_count, every, video_seconds, fps_number)
    ffmpeg = find_ffmpeg(Path(sys.executable).parent, os.environ.get("PATH", ""))
    frames = frame_count(video_seconds, fps_number)
    print(
        f"video: {video_seconds!r} s, {len(shown)} shown steps, {fps_number!r} fps, {frames} frames",
        file=sys.stderr,
    )
    holds = frame_holds(len(shown), frames)
    builder = make_frames(result, layout, shown)
    temporary = outputs.temporary_path_for(destination)
    try:
        encode(builder, shown, holds, fps, ffmpeg, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
