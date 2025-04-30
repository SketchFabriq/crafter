import numpy as np
from panda_gym.envs.core import RobotTaskEnv
from panda_gym.pybullet import PyBullet

from visioncraft.envs.widowx import WidowX

import numpy as np
from panda_gym.envs.core import RobotTaskEnv
from panda_gym.pybullet import PyBullet

from gymnasium import spaces

from .custom_task import (
    TouchTask,
    GraspTask,
    LiftTask,
    PickAndPlaceTask,
)
from visioncraft.envs.widowx import WidowX


class BaseWidowXEnv(RobotTaskEnv):
    OBS_DIM = 19
    ACTION_DIM = 4
    GOAL_DIM = 3

    def __init__(self, robot_kwargs, task_instance, **kwargs):
        # render_distance: float = 1.4,
        # render_yaw: float = 45,
        # render_pitch: float = -30,
        # render_roll: float = 0,
        super().__init__(robot=robot_kwargs["robot"], task=task_instance, render_distance=0.7,render_yaw=-90.0, render_pitch=-65, **kwargs)
        self.observation_space = spaces.Dict(
            dict(
                observation=spaces.Box(
                    -10.0, 10.0, shape=(self.OBS_DIM,), dtype=np.float32
                ),
                desired_goal=spaces.Box(
                    -10.0, 10.0, shape=(self.GOAL_DIM,), dtype=np.float32
                ),
                achieved_goal=spaces.Box(
                    -10.0, 10.0, shape=(self.GOAL_DIM,), dtype=np.float32
                ),
            )
        )
        self.action_space = spaces.Box(
            -1.0, 1.0, shape=(self.ACTION_DIM,), dtype=np.float32
        )


class WidowXTouchEnv(BaseWidowXEnv):
    def __init__(self, render_mode="rgb_array", control_type="ee", **kwargs):
        sim = PyBullet(render_mode=render_mode, renderer="Tiny")
        robot = WidowX(
            sim,
            block_gripper=False,
            base_position=np.array([-0.3, 0.0, 0.0]),
            control_type=control_type,
        )
        task = TouchTask(sim, robot, **kwargs)
        robot_kwargs = {"robot": robot}
        super().__init__(robot_kwargs=robot_kwargs, task_instance=task)
        BaseWidowXEnv.GOAL_DIM = task.get_goal().shape[0]


class WidowXGraspEnv(BaseWidowXEnv):
    def __init__(self, render_mode="rgb_array", control_type="ee", **kwargs):
        sim = PyBullet(render_mode=render_mode, renderer="Tiny")
        robot = WidowX(
            sim,
            block_gripper=False,
            base_position=np.array([-0.3, 0.0, 0.0]),
            control_type=control_type,
        )
        task = GraspTask(sim, robot, **kwargs)
        robot_kwargs = {"robot": robot}
        super().__init__(robot_kwargs=robot_kwargs, task_instance=task)
        BaseWidowXEnv.GOAL_DIM = task.get_goal().shape[0]


class WidowXLiftEnv(BaseWidowXEnv):
    def __init__(self, render_mode="rgb_array", control_type="ee", **kwargs):
        sim = PyBullet(
            render_mode=render_mode,
            renderer="Tiny",
            background_color=np.array([100.0, 100.0, 100.0]),
        )
        robot = WidowX(
            sim,
            block_gripper=False,
            base_position=np.array([-0.3, 0.0, 0.0]),
            control_type=control_type,
        )
        task = LiftTask(sim, robot)
        robot_kwargs = {"robot": robot}
        super().__init__(robot_kwargs=robot_kwargs, task_instance=task)
        BaseWidowXEnv.GOAL_DIM = task.get_goal().shape[0]


class WidowXPickPlaceEnv(BaseWidowXEnv):
    def __init__(self, render_mode="rgb_array", control_type="ee", **kwargs):
        sim = PyBullet(render_mode=render_mode, renderer="Tiny")
        robot = WidowX(
            sim,
            block_gripper=False,
            base_position=np.array([-0.3, 0.0, 0.0]),
            control_type=control_type,
        )
        task = PickAndPlaceTask(sim, robot)
        robot_kwargs = {"robot": robot}
        super().__init__(robot_kwargs=robot_kwargs, task_instance=task, **kwargs)
        BaseWidowXEnv.GOAL_DIM = task.get_goal().shape[0]
