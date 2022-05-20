"""
Modified version of robosuite wrapper to have a TAMP planner execute behind-the-scenes during
    env.step and env.reset calls.
"""
import numpy as np

import robosuite
from robosuite.utils.mjcf_utils import postprocess_model_xml

import robomimic.envs.env_base as EB
from robomimic.envs.env_robosuite import EnvRobosuite


class EnvRobosuiteTAMP(EnvRobosuite):
    def __init__(
        self, 
        env_name, 
        render=False, 
        render_offscreen=False, 
        use_image_obs=False, 
        postprocess_visual_obs=True, 
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
        """
        super(EnvRobosuiteTAMP, self).__init__(
            env_name=env_name, 
            render=render, 
            render_offscreen=render_offscreen, 
            use_image_obs=use_image_obs, 
            postprocess_visual_obs=postprocess_visual_obs, 
            **kwargs,
        )

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
        obs, r, done, info = super(EnvRobosuiteTAMP, self).step(action=action)

        # agent needs to know when additional time passed in the world due to the TAMP planner executing,
        # so that it can reset its internal state and do other housekeeping
        info["planner_executed"] = False

        ### TODO: insert logic here to execute TAMP plan if necessary, and set info variable above ###

        return obs, r, done, info

    def reset(self, tamp_init=False):
        """
        Reset environment.

        Returns:
            observation (dict): initial observation dictionary.
        """
        di = super(EnvRobosuiteTAMP, self).reset()

        ### TODO: consider moving this logic to hitl-tamp repo - with "TAMP" wrappers for each robosuite env we care about. ###
        ###       Then, we use that same environment during Teleop data collection and here.                                ###
        if tamp_init:
            self.do_tamp()
            # need to re-retrieve observation after TAMP execution
            return self.get_observation()

        return di

    def do_tamp(self):
        """
        A placeholder for TAMP logic.
        """
        ### TODO: add TAMP logic here, for beginning of episode and possibly later on ###
        pass

    def reset_to(self, state):
        """
        Reset to a specific simulator state.

        Args:
            state (dict): current simulator state that contains one or more of:
                - states (np.ndarray): initial state of the mujoco environment
                - model (str): mujoco scene xml
        
        Returns:
            observation (dict): observation dictionary after setting the simulator state (only
                if "states" is in @state)
        """
        should_ret = False
        if "model" in state:
            self.reset(tamp_init=False)
            xml = postprocess_model_xml(state["model"])
            self.env.reset_from_xml_string(xml)
            self.env.sim.reset()
            if not self._is_v1:
                # hide teleop visualization after restoring from model
                self.env.sim.model.site_rgba[self.env.eef_site_id] = np.array([0., 0., 0., 0.])
                self.env.sim.model.site_rgba[self.env.eef_cylinder_id] = np.array([0., 0., 0., 0.])
        if "states" in state:
            self.env.sim.set_state_from_flattened(state["states"])
            self.env.sim.forward()
            should_ret = True

        if "goal" in state:
            self.set_goal(**state["goal"])

        if "model" in state:
            # after the reset, run TAMP planner if necessary
            self.do_tamp()

        if should_ret:
            # only return obs if we've done a forward call - otherwise the observations will be garbage
            return self.get_observation()
        return None

    @property
    def type(self):
        """
        Returns environment type (int) for this kind of environment.
        This helps identify this env class.
        """
        return EB.EnvType.ROBOSUITE_TAMP_TYPE
