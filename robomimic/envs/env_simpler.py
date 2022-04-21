"""
This file contains the simpler environment wrapper that is used
to provide a standardized environment API for training policies and interacting
with metadata present in datasets.
"""
import json
import time
import argparse
import numpy as np
from copy import deepcopy

import isaacgym

import simpler_app_utils
from simpler_app_utils.task_env import SimPLERTask

import robomimic.utils.obs_utils as ObsUtils
import robomimic.envs.env_base as EB


class EnvSimPLER(EB.EnvBase):
    """Wrapper class for SimPLER environments."""
    def __init__(
        self, 
        env_name, 
        render=False, 
        render_offscreen=False, 
        use_image_obs=False, 
        postprocess_visual_obs=True, 
        namespace_args=None,
        usd_file_str=None,
        **kwargs,
    ):
        """
        Args:
            env_name (str): name of environment. Only needs to be provided if making a different
                environment from the one in @env_meta.

            render (bool): if True, environment supports on-screen rendering

            render_offscreen (bool): if True, environment supports off-screen rendering. This
                is forced to be True if @env_meta["use_images"] is True.

            use_image_obs (bool): if True, environment is expected to render rgb image observations
                on every env.step call. Set this to False for efficiency reasons, if image
                observations are not required.

            postprocess_visual_obs (bool): if True, postprocess image observations
                to prepare for learning. This should only be False when extracting observations
                for saving to a dataset (to save space on RGB images for example).

            namespace_args (dict or argparse.Namespace): if provided, pass the provided arguments to the
                wrapped SimPLER env

            usd_file_str (str): if provided, interpret the string as a usd file and pass it to the 
                environment
        """
        self.postprocess_visual_obs = postprocess_visual_obs

        self._env_name = env_name

        if namespace_args is None:
            # empty args
            namespace_args = argparse.Namespace()
        elif isinstance(namespace_args, dict):
            # load args from dict
            namespace_args = argparse.Namespace(**namespace_args)
        assert isinstance(namespace_args, argparse.Namespace)

        # prepare usd file
        if usd_file_str is None:
            # read usd file string from usd file - it will be saved as metadata
            f = open(namespace_args.usd, "r")
            usd_file_str = f.read()
            f.close()
        else:
            ### TODO: is there a cleaner way to do this? ###

            # assume we need to set args.usd to point to this string - make a file in /tmp
            tmp_usd_file = "/tmp/{}.usda".format(str(time.time()).replace(".", "_"))
            f = open(tmp_usd_file, "w")
            f.write(usd_file_str)
            f.close()
            namespace_args.usd = tmp_usd_file

        # there is a chance that this was saved as an int during serialization, so convert back
        namespace_args.physics_engine = isaacgym.gymapi.SimType(namespace_args.physics_engine)

        # set rendering settings
        assert not (render and render_offscreen)
        namespace_args.headless = (not render)
        self._headless = namespace_args.headless

        # update kwargs based on passed arguments
        kwargs = deepcopy(kwargs)
        update_kwargs = dict(
            use_camera_obs=use_image_obs,
        )
        kwargs.update(update_kwargs)

        # cache the env kwargs and namespace object as a dict, to save as metadata
        self._init_kwargs = deepcopy(kwargs)
        namespace_args_copy = deepcopy(namespace_args)
        namespace_args_copy.physics_engine = int(namespace_args_copy.physics_engine) # need int for json serialization
        self._namespace_args_dict = vars(namespace_args_copy)
        self._usd_file_str = usd_file_str

        self.env = SimPLERTask(args=namespace_args, **kwargs)

    def step(self, action):
        """
        Step in the environment with an action.

        Args:
            action (np.array): action to take

        Returns:
            observation (dict): new observation dictionary
            reward (float): reward for this step
            done (bool): whether the task is done
            info (dict): extra information
        """
        obs, r, done, info = self.env.step(action)
        obs = self.get_observation(obs)
        return obs, r, self.is_done(), info

    def reset(self):
        """
        Reset environment.

        Returns:
            observation (dict): initial observation dictionary.
        """
        di = self.env.reset()
        return self.get_observation(di)

    def reset_to(self, state):
        """
        Reset to a specific simulator state.

        Args:
            state (dict): current simulator state that contains one or more of:
                - states (dict): state of the simulator environment (usually consisting
                    of str : np.array pairs)
        
        Returns:
            observation (dict): observation dictionary after setting the simulator state (only
                if "states" is in @state)
        """
        should_ret = False
        if "states" in state:
            self.env.set_state(state["states"])
            should_ret = True

        if "goal" in state:
            self.set_goal(**state["goal"])
        if should_ret:
            # only return obs if we've done a forward call - otherwise the observations will be garbage
            return self.get_observation()
        return None

    def render(self, mode="human", height=None, width=None, camera_name="default"):
        """
        Render from simulation to either an on-screen window or off-screen to RGB array.

        Args:
            mode (str): pass "human" for on-screen rendering or "rgb_array" for off-screen rendering
            height (int): height of image to render - only used if mode is "rgb_array"
            width (int): width of image to render - only used if mode is "rgb_array"
            camera_name (str): camera name to use for rendering
        """

        ### TODO: support setting image size in this function ###
        if mode == "human":
            assert not self._headless
            self.env.update_gui()
        elif mode == "rgb_array":
            return self.env.render(name=camera_name, height=height, width=width)
        else:
            raise NotImplementedError("mode={} is not implemented".format(mode))

    def get_observation(self, di=None):
        """
        Get current environment observation dictionary.

        Args:
            di (dict): current raw observation dictionary from robosuite to wrap and provide 
                as a dictionary. If not provided, will be queried from robosuite.
        """
        if di is None:
            di = self.env.compute_observations()
        ret = {}
        for k in di:
            # handle rgb image observations
            if (k in ObsUtils.OBS_KEYS_TO_MODALITIES) and ObsUtils.key_is_obs_modality(key=k, obs_modality="rgb"):
                ret[k] = di[k][::-1]
                if self.postprocess_visual_obs:
                    ret[k] = ObsUtils.process_obs(obs=ret[k], obs_key=k)

        ### TODO: filter out whatever we want for learning ###
        for k in di:
            if k not in ret:
                ret[k] = np.array(di[k])
        return ret

    def get_state(self):
        """
        Get current environment simulator state as a dictionary. Should be compatible with @reset_to.
        """

        # TODO: consider saving usd scene string here, especially if it is liable to change from episode to episode

        # get state of all objects
        state_dict = self.env.get_state()
        return dict(states=state_dict)

    def get_reward(self):
        """
        Get current reward.
        """
        return self.env.reward()

    def get_goal(self):
        """
        Get goal observation. Not all environments support this.
        """
        raise NotImplementedError

    def set_goal(self, **kwargs):
        """
        Set goal observation with external specification. Not all environments support this.
        """
        raise NotImplementedError

    def is_done(self):
        """
        Check if the task is done (not necessarily successful).
        """

        # SimPLER envs are assumed to rollout to a fixed maximum horizon.
        return False

    def is_success(self):
        """
        Check if the task condition(s) is reached. Should return a dictionary
        { str: bool } with at least a "task" key for the overall task success,
        and additional optional keys corresponding to other task criteria.
        """
        succ = self.env.check_success()
        if isinstance(succ, dict):
            assert "task" in succ
            return succ
        return { "task" : succ }

    @property
    def action_dimension(self):
        """
        Returns dimension of actions (int).
        """
        return self.env.action_dimension

    @property
    def name(self):
        """
        Returns name of environment name (str).
        """
        return self._env_name

    @property
    def type(self):
        """
        Returns environment type (int) for this kind of environment.
        This helps identify this env class.
        """
        return EB.EnvType.SIMPLER_TYPE

    def serialize(self):
        """
        Save all information needed to re-instantiate this environment in a dictionary.
        This is the same as @env_meta - environment metadata stored in hdf5 datasets,
        and used in utils/env_utils.py.
        """
        env_kwargs = deepcopy(self._init_kwargs)
        env_kwargs["namespace_args"] = deepcopy(self._namespace_args_dict)
        env_kwargs["usd_file_str"] = self._usd_file_str
        return dict(env_name=self.name, type=self.type, env_kwargs=env_kwargs)

    @classmethod
    def create_for_data_processing(
        cls, 
        env_name, 
        camera_names, 
        camera_height, 
        camera_width, 
        reward_shaping, 
        **kwargs,
    ):
        """
        Create environment for processing datasets, which includes extracting
        observations, labeling dense / sparse rewards, and annotating dones in
        transitions. 

        Args:
            env_name (str): name of environment
            camera_names (list of str): list of camera names that correspond to image observations
            camera_height (int): camera height for all cameras
            camera_width (int): camera width for all cameras
            reward_shaping (bool): if True, use shaped environment rewards, else use sparse task completion rewards
        """
        assert not reward_shaping, "TODO: no support for reward shaping flag yet"

        # add in appropriate camera kwargs
        has_camera = (len(camera_names) > 0)
        new_kwargs = dict()
        if has_camera:
            new_kwargs["camera_names"] = list(camera_names)
            new_kwargs["camera_heights"] = camera_height
            new_kwargs["camera_widths"] = camera_width
        kwargs.update(new_kwargs)

        # also initialize obs utils so it knows which modalities are image modalities
        image_modalities = ["{}_image".format(cn) for cn in camera_names]
        obs_modality_specs = {
            "obs": {
                "low_dim": [], # technically unused, so we don't have to specify all of them
                "rgb": image_modalities,
            }
        }
        ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs)

        # note that @postprocess_visual_obs is False since this env's images will be written to a dataset
        return cls(
            env_name=env_name,
            render=False, 
            render_offscreen=has_camera, 
            use_image_obs=has_camera, 
            postprocess_visual_obs=False,
            **kwargs,
        )

    @property
    def rollout_exceptions(self):
        """
        Return tuple of exceptions to except when doing rollouts. This is useful to ensure
        that the entire training run doesn't crash because of a bad policy that causes unstable
        simulation computations.
        """
        return tuple()

    def __repr__(self):
        """
        Pretty-print env description.
        """
        return self.name + "\n" + json.dumps(self._init_kwargs, sort_keys=True, indent=4)
