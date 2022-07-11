"""
A script to playback demonstrations (using visual observations and the pygame renderer)
in order to allow a user to annotate portions of the demonstrations. This is useful
to indicate important keyframes to the TAMP planner.

Args:
    dataset (str): path to hdf5 dataset

    filter_key (str): if provided, use the subset of trajectories
        in the file that correspond to this filter key

    n (int): if provided, stop after n trajectories are processed

    video_skip (int): render frames to screen every @video_skip steps

    render_image_names (str or [str]): camera name(s) / image observation(s) to 
        use for rendering on-screen or to video
"""

import os
import sys
import time
import json
import h5py
import argparse
import imageio
import numpy as np
from collections import deque, OrderedDict
from contextlib import contextmanager

# for rendering images on-screen
import cv2
import pygame

import robomimic
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
from robomimic.envs.env_base import EnvBase, EnvType


# Define default cameras to use for each env type
DEFAULT_CAMERAS = {
    EnvType.ROBOSUITE_TYPE: ["agentview_image"],
    # EnvType.IG_MOMART_TYPE: ["rgb"],
    EnvType.GYM_TYPE: ValueError("No camera names supported for gym type env!"),
}

# scaling size for images when rendering to screen
# IMAGE_SCALE = 5
IMAGE_SCALE = 1


class Grid(object):
    """
    Keep track of a list of values, and point to a single value at a time.
    """
    def __init__(self, values, initial_ind=0):
        self.values = list(values)
        self.ind = initial_ind
        self.n = len(self.values)

    def get(self):
        return self.values[self.ind]

    def next(self):
        self.ind = min(self.ind + 1, self.n - 1)
        return self.get()

    def prev(self):
        self.ind = max(self.ind - 1, 0)
        return self.get()


# Grid of timing rates
RATE_GRID = Grid(
    values=[1, 5, 10, 20, 40],
    initial_ind=0,
)


class Timer(object):
    """
    A simple timer.
    """
    def __init__(self, history=100, ignore_first=False):
        """
        Args:
            history (int): number of recent timesteps to record for reporting statistics
        """
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.last_diff = 0.
        self.average_time = 0.
        self.min_diff = float("inf")
        self.max_diff = 0.
        self._measurements = deque(maxlen=history)
        self._enabled = True
        self.ignore_first = ignore_first
        self._had_first = False

    def enable(self):
        """
        Enable measurements with this timer.
        """
        self._enabled = True

    def disable(self):
        """
        Disable measurements with this timer.
        """
        self._enabled = False

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self):
        if self._enabled:

            if self.ignore_first and (self.start_time > 0. and not self._had_first):
                self._had_first = True
                return time.time() - self.start_time

            self.last_diff = time.time() - self.start_time
            self.total_time += self.last_diff
            self.calls += 1
            self.average_time = self.total_time / self.calls
            self.min_diff = min(self.min_diff, self.last_diff)
            self.max_diff = max(self.max_diff, self.last_diff)
            self._measurements.append(self.last_diff)
        last_diff = self.last_diff
        return last_diff

    @contextmanager
    def timed(self):
        self.tic()
        yield
        self.toc()

    def report_stats(self, verbose=False):
        stats = OrderedDict()
        stats["global"] = OrderedDict(
            mean=self.average_time,
            min=self.min_diff,
            max=self.max_diff,
            num=self.calls,
        )
        num = len(self._measurements)
        stats["local"] = OrderedDict()
        if num > 0:
            stats["local"] = OrderedDict(
                mean=np.mean(self._measurements),
                std=np.std(self._measurements),
                min=np.min(self._measurements),
                max=np.max(self._measurements),
                num=num,
            )
        if verbose:
            stats["local"]["values"] = list(self._measurements)
        return stats


class Rate(object):
    """
    Convenience class for enforcing rates in loops. Modeled after rospy.Rate.

    See http://docs.ros.org/en/jade/api/rospy/html/rospy.timer-pysrc.html#Rate.sleep
    """
    def __init__(self, hz):
        """
        Args:
            hz (int): frequency to enforce
        """
        self.update_hz(hz)

    def update_hz(self, hz):
        """
        Update rate to enforce.
        """
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = (1. / hz)

    def _remaining(self, curr_time):
        """
        Calculate time remaining for rate to sleep.
        """
        assert curr_time >= self.last_time, "time moved backwards!"
        elapsed = curr_time - self.last_time
        return self.sleep_duration - elapsed

    def sleep(self):
        """
        Attempt to sleep at the specified rate in hz, by taking the time
        elapsed since the last call to this function into account.
        """
        curr_time = time.time()
        remaining = self._remaining(curr_time)
        if remaining > 0:
            time.sleep(remaining)

        # assume successful rate sleeping
        self.last_time = self.last_time + self.sleep_duration

        # NOTE: this commented line is what we used to do, but this enforces a slower rate
        # self.last_time = time.time()

        # detect time jumping forwards (e.g. loop is too slow)
        if curr_time - self.last_time > self.sleep_duration * 2:
            # we didn't sleep at all
            self.last_time = curr_time


class RateMeasure(object):
    """
    Measure approximate time intervals of code execution by calling @measure
    """
    def __init__(self, name=None, history=100, freq_threshold=None):
        self._timer = Timer(history=history, ignore_first=True)
        self._timer.tic()
        self.name = name
        self.freq_threshold = freq_threshold
        self._enabled = True
        self._first = False
        self.sum = 0.
        self.calls = 0

    def enable(self):
        """
        Enable measurements.
        """
        self._timer.enable()
        self._enabled = True

    def disable(self):
        """
        Disable measurements.
        """
        self._timer.disable()
        self._enabled = False

    def measure(self):
        """
        Take a measurement of the time elapsed since the last @measure call
        and also return the time elapsed.
        """
        interval = self._timer.toc()
        self._timer.tic()
        self.sum += (1. / interval)
        self.calls += 1
        if self._enabled and (self.freq_threshold is not None) and ((1. / interval) < self.freq_threshold):
            print("WARNING: RateMeasure {} violated threshold {} hz with measurement {} hz".format(self.name, self.freq_threshold, (1. / interval)))
            return (interval, True)
        return (interval, False)

    def report_stats(self, verbose=False):
        """
        Report statistics over measurements, converting timer measurements into frequencies.
        """
        stats = self._timer.report_stats(verbose=verbose)
        stats["name"] = self.name
        if stats["global"]["num"] > 0:
            stats["global"] = OrderedDict(
                mean=(self.sum / float(self.calls)),
                min=(1. / stats["global"]["max"]),
                max=(1. / stats["global"]["min"]),
                num=stats["global"]["num"],
            )
        if len(stats["local"]) > 0:
            measurements = [1. / x for x in self._timer._measurements]
            stats["local"] = OrderedDict(
                mean=np.mean(measurements),
                std=np.std(measurements),
                min=np.min(measurements),
                max=np.max(measurements),
                num=stats["local"]["num"],
            )
        return stats

    def __str__(self):
        stats = self.report_stats(verbose=False)
        return json.dumps(stats, indent=4)


def print_keyboard_commands():
    """
    Helper function to print keyboard annotation commands.
    """
    def print_command(char, info):
        char += " " * (11 - len(char))
        print("{}\t{}".format(char, info))

    print("")
    print_command("Keys", "Command")
    print_command("up-down", "increase / decrease playback speed")
    print_command("left-right", "seek left / right by N frames")
    print_command("spacebar", "hold down to annotate intervention")
    print_command("f", "next demo and save annotations")
    print_command("r", "repeat demo and clear annotations")
    print("")


def make_pygame_screen(
    traj_grp,
    image_names,
):
    # grab first image from all image modalities to infer size of window
    im = [traj_grp["obs/{}".format(k)][0] for k in image_names]
    frame = np.concatenate(im, axis=1)
    width, height = frame.shape[:2]
    width *= IMAGE_SCALE
    height *= IMAGE_SCALE
    screen = pygame.display.set_mode((width, height))
    return screen


def handle_pygame_events(
    frame_ind,
    int_start,
    interventions,
    rate_obj,
    need_repeat,
    annotation_done,
):
    """
    Reads events from pygame window in order to provide the
    following keyboard annotation functionality:

        up-down     | increase / decrease playback speed
        left-right  | seek left / right by N frames
        spacebar    | hold down to annotate intervention
        f           | next demo and save annotations
        r           | repeat demo and clear annotations
    """

    seek = 0
    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            sys.exit()

        if event.type == pygame.KEYDOWN:
            # print("pressed key {}".format(event.key))
            if event.key == pygame.K_SPACE:
                # intervention start
                assert int_start is None
                int_start = frame_ind

        if event.type == pygame.KEYUP:
            # print("released key {}".format(event.key))
            if event.key == pygame.K_SPACE:
                # intervention end
                assert int_start is not None
                interventions.append((int_start, frame_ind))
                int_start = None
            elif event.key == pygame.K_UP:
                # speed up traversal
                rate_obj.update_hz(RATE_GRID.next())
                print("cmd: playback rate increased to {} hz".format(rate_obj.hz))
            elif event.key == pygame.K_DOWN:
                # slow down traversal
                rate_obj.update_hz(RATE_GRID.prev())
                print("cmd: playback rate decreased to {} hz".format(rate_obj.hz))
            elif event.key == pygame.K_LEFT:
                # seek left
                seek = -10
                print("cmd: seek {} frames".format(seek))
            elif event.key == pygame.K_RIGHT:
                # seek right
                seek = 10
                print("cmd: seek {} frames".format(seek))
            elif event.key == pygame.K_r:
                # repeat demo
                need_repeat = True
                print("cmd: repeat demo")
            elif event.key == pygame.K_f:
                # next demo
                annotation_done = True
                print("cmd: next demo")

    return int_start, need_repeat, annotation_done, seek


def annotate_interventions_in_trajectory(
    ep,
    traj_grp,
    screen,
    video_skip, 
    image_names,
):
    """
    This function reads all "rgb" observations in the dataset trajectory and
    writes them into a video.

    Args:
        ep (str): name of hdf5 group for this demo
        traj_grp (hdf5 file group): hdf5 group which corresponds to the dataset trajectory to playback
        screen: pygame screen
        video_skip (int): determines rate at which environment frames are written to video
        image_names (list): determines which image observations are used for rendering. Pass more than
            one to output a video with multiple image observations concatenated horizontally.
    """
    assert image_names is not None, "error: must specify at least one image observation to use in @image_names"

    traj_len = traj_grp["actions"].shape[0]

    rate_obj = Rate(hz=RATE_GRID.get())
    rate_measure = RateMeasure(name="rate_measure")

    # repeat this demonstration until we have permission to move on
    annotation_done = False
    while not annotation_done:
        print("Starting annotation for demo: {}".format(ep))
        print_keyboard_commands()

        need_repeat = False
        interventions = []
        int_start = None

        # keep looping through the video, reading user input from keyboard, until
        # user indicates that demo is done being annotated
        frame_ind = 0
        while (not need_repeat) and (not annotation_done):

            # maybe render frame to screen
            if frame_ind % video_skip == 0:
                # concatenate image obs together
                im = [traj_grp["obs/{}".format(k)][frame_ind] for k in image_names]
                frame = np.concatenate(im, axis=1)
                # upscale frame to appropriate resolution
                frame = cv2.resize(frame, 
                    dsize=(frame.shape[0] * IMAGE_SCALE, frame.shape[1] * IMAGE_SCALE), 
                    interpolation=cv2.INTER_CUBIC)
                # add red border during intervention annotation
                if int_start is not None:
                    border_size_h = round(0.02 * frame.shape[0])
                    border_size_w = round(0.02 * frame.shape[1])
                    frame[:border_size_h, :, :] = [255., 0., 0.]
                    frame[-border_size_h:, :, :] = [255., 0., 0.]
                    frame[:, :border_size_w, :] = [255., 0., 0.]
                    frame[:, -border_size_w:, :] = [255., 0., 0.]
                # write frame to window
                frame = frame.transpose((1, 0, 2))
                pygame.pixelcopy.array_to_surface(screen, frame)
                pygame.display.update()

            int_start, need_repeat, annotation_done, seek = handle_pygame_events(
                frame_ind=frame_ind,
                int_start=int_start,
                interventions=interventions,
                rate_obj=rate_obj,
                need_repeat=need_repeat,
                annotation_done=annotation_done,
            )

            # try to enforce rate
            rate_obj.sleep()
            rate_measure.measure()

            # increment frame index appropriately (either by 1 or by seek amount), then
            # clamp within bounds
            mask = int(seek != 0)
            frame_ind += (1 - mask) * 1 + mask * seek
            frame_ind = max(min(frame_ind, traj_len - 1), 0)

        # if we don't need to repeat the demo, we're done
        annotation_done = annotation_done or (not need_repeat)

        # handle case where last intervention did not have an end
        if int_start is not None:
            interventions.append((int_start, i))

    # write interventions to hdf5 group
    interventions_arr = np.zeros(traj_len).astype(int).astype(bool)
    for (start, end) in interventions:
        interventions_arr[start : end] = True
    if "interventions" in traj_grp:
        del traj_grp["interventions"]
    traj_grp["interventions"] = interventions_arr

    # report rate measurements
    print("\nFrame Rate (Hz) Statistics for demo {} annotation".format(ep))
    print(rate_measure)


def annotate_interventions(args):
    # Auto-fill camera rendering info if not specified
    if args.render_image_names is None:
        # We fill in the automatic values
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
        env_type = EnvUtils.get_env_type(env_meta=env_meta)
        args.render_image_names = DEFAULT_CAMERAS[env_type]

    # Open the file in read-write mode to write "interventions" dataset per trajectory
    f = h5py.File(args.dataset, "a")

    # list of all demonstration episodes (sorted in increasing number order)
    if args.filter_key is not None:
        print("using filter key: {}".format(args.filter_key))
        demos = [elem.decode("utf-8") for elem in np.array(f["mask/{}".format(args.filter_key)])]
    else:
        demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    # maybe start with some offset
    if args.start is not None:
        demos = demos[args.start:]

    # maybe reduce the number of demonstrations to playback
    if args.n is not None:
        demos = demos[:args.n]

    # make pygame screen first
    screen = make_pygame_screen(traj_grp=f["data/{}".format(demos[0])], image_names=args.render_image_names)

    for ind in range(len(demos)):
        ep = demos[ind]
        print("Annotating episode: {}".format(ep))

        annotate_interventions_in_trajectory(
            ep=ep,
            traj_grp=f["data/{}".format(ep)],
            screen=screen, 
            video_skip=args.video_skip,
            image_names=args.render_image_names,
        )

    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="path to hdf5 dataset",
    )
    parser.add_argument(
        "--filter_key",
        type=str,
        default=None,
        help="(optional) filter key, to select a subset of trajectories in the file",
    )

    # number of trajectories to playback. If omitted, playback all of them.
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="(optional) stop after n trajectories are played",
    )

    # start after this many trajectories in the dataset
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="(optional) start after this many trajectories in the dataset",
    )

    # How often to write frames on-screen during the playback
    parser.add_argument(
        "--video_skip",
        type=int,
        default=1,
        help="render frames on-screen every n steps",
    )

    # camera names to render, or image observations to use for writing to video
    parser.add_argument(
        "--render_image_names",
        type=str,
        nargs='+',
        default=None,
        help="(optional) camera name(s) / image observation(s) to use for rendering on-screen or to video. Default is"
             "None, which corresponds to a predefined camera for each env type",
    )

    args = parser.parse_args()
    annotate_interventions(args)