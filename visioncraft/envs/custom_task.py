from operator import is_
from typing import Any, Dict
import numpy as np
from panda_gym.envs.core import Task
from panda_gym.pybullet import PyBullet
from panda_gym.utils import distance


class TouchTask(Task):
    """Stage 1: Reach near the object (2D axis)"""

    def __init__(
        self,
        sim: PyBullet,
        robot: Any,
        reward_type: str = "dense",
        threshold: float = 0.045,
        randomize: bool = True,
        goal_xy_range=0.3,
        obj_xy_range: float = 0.3,
    ):
        super().__init__(sim)
        self.robot = robot
        self.reward_type = reward_type
        self.threshold = threshold
        self.randomize = randomize
        self.object_size = 0.03
        self.obj_range_low = np.array([-obj_xy_range / 2, -obj_xy_range / 2, 0])
        self.obj_range_high = np.array([obj_xy_range / 2, obj_xy_range / 2, 0])
        self.goal_range_low = np.array([-goal_xy_range / 2, -goal_xy_range / 2, 0])
        self.goal_range_high = np.array([goal_xy_range / 2, goal_xy_range / 2, 0])

        with self.sim.no_rendering():
            self._create_scene()

    def _create_scene(self) -> None:
        """Create the scene."""
        self.sim.create_box(
            body_name="plane",
            half_extents=np.array([5.0, 5.0, 0.01]),
            mass=0.0,
            position=np.array([0.0, 0.0, -0.4 - 0.01]),
            specular_color=np.zeros(3),
            rgba_color=np.array([134 / 255, 96 / 255, 72 / 255, 1]),
        )

        self.sim.create_box(
            body_name="table",
            half_extents=np.array([1.1, 0.7, 0.4]) / 2,
            mass=0.0,
            position=np.array([-0.3, 0.0, -0.4 / 2]),
            specular_color=np.zeros(3),
            rgba_color=np.array([211 / 255, 183 / 255, 152 / 255, 1]),
            lateral_friction=None,
            spinning_friction=None,
        )

        self.sim.create_box(
            body_name="object",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=0.5,
            position=np.array([0.0, 0.0, self.object_size / 2]),
            rgba_color=np.array([1.0, 0.2, 0.2, 1.0]),
        )

    def get_obs(self) -> np.ndarray:
        # position, rotation of the object
        object_position = self.sim.get_base_position("object")
        object_rotation = self.sim.get_base_rotation("object")
        object_velocity = self.sim.get_base_velocity("object")
        object_angular_velocity = self.sim.get_base_angular_velocity("object")
        observation = np.concatenate([object_position, object_rotation, object_velocity, object_angular_velocity])
        return observation

    def get_achieved_goal(self) -> np.ndarray:
        object_position = np.array(self.robot.get_ee_position())
        return object_position
    
    def get_desired_goal(self) -> np.ndarray:
        # Current object position
        return np.array(self.sim.get_base_position("object"))

    def reset(self) -> None:
        object_position = self._sample_object()
        self.goal = object_position
        self.sim.set_base_pose("object", object_position, np.array([0.0, 0.0, 0.0, 1.0]))


    def _sample_goal(self) -> np.ndarray:
        """Sample a goal."""
        goal = np.array([0.0, 0.0, self.object_size / 2])  # z offset for the cube center
        noise = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        if self.np_random.random() < 0.3:
            noise[2] = 0.0
        goal += noise
        return goal

    def _sample_object(self) -> np.ndarray:
        """Randomize start position of object."""
        object_position = np.array([0.0, 0.0, self.object_size / 2])
        noise = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
        object_position += noise
        return object_position
    
    def check_collision(self, body_a: str, body_b: str) -> bool:
        contacts = self.sim.physics_client.getContactPoints(
            bodyA=self.sim._bodies_idx[body_a],
            bodyB=self.sim._bodies_idx[body_b]
        )
        return len(contacts) > 0

    

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any] = {}) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        # check contact between two pybullet object
        contact = self.check_collision('robot_arm', "object")

        return contact

    def compute_reward(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any]
    ) -> np.ndarray:

        d = distance(achieved_goal, desired_goal)

        if self.reward_type == "sparse":
            return -np.array(d > self.threshold, dtype=np.float32)
        else:
            return -d.astype(np.float32)


class GraspTask(TouchTask):
    """Stage 3: Grasping the object successfully"""

    def is_object_grasped(self, achieved_goal: np.ndarray) -> bool:
        gripper_width = self.robot.get_fingers_width()
        obj_pos = self.sim.get_base_position("object")
        ee_pos = achieved_goal

        corrected_gripper_width = gripper_width - 0.02

        # Object is between fingers and close to EE
        is_closed = corrected_gripper_width < self.object_size * 1.1 and corrected_gripper_width > self.object_size * 0.9
        is_aligned = distance(ee_pos, obj_pos) < self.threshold

        if is_closed and is_aligned:
            print("success!!!!")

        return is_closed and is_aligned

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any] = {}) -> np.ndarray:
        return np.array(self.is_object_grasped(achieved_goal), dtype=bool)

    def compute_reward(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any]
    ) -> np.ndarray:
        if self.is_object_grasped(achieved_goal):
            return np.array(10.0, dtype=np.float32)  # Success

        # If not grasped, provide shaping reward based on distance
        d = distance(achieved_goal, desired_goal)
        if self.reward_type == "sparse":
            return np.array(-1.0, dtype=np.float32)  # Failure
        else:
            return -d.astype(np.float32)  # Dense reward


class LiftTask(GraspTask):
    """Stage 4: Lifting object a few cm above table"""

    def __init__(
        self,
        sim: PyBullet,
        robot: Any,
        reward_type: str = "dense",
        threshold: float = 0.03,
        lift_height: float = 0.05,
        randomize: bool = True,
        obj_xy_range: float = 0.3,
    ):
        super().__init__(sim, robot, reward_type, threshold, randomize, obj_xy_range)
        self.lift_height = lift_height
        self.initial_z = 0.0

    def _create_scene(self) -> None:
        """Create the scene."""
        super()._create_scene()
        # Create a visible target marker
        self.sim.create_box(
            body_name="target",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=0.0,
            ghost=True,
            position=np.array([0.0, 0.0, self.object_size / 2]),
            rgba_color=np.array([0.1, 0.9, 0.1, 0.1]),
        )

    def reset(self) -> None:
        object_position = self._sample_object()
        self.goal = object_position + np.array([0.0, 0.0, self.lift_height])
        self.sim.set_base_pose("object", object_position, np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("target", self.goal, np.array([0.0, 0.0, 0.0, 1.0]))
  
    def get_achieved_goal(self) -> np.ndarray:
        object_position = np.array(self.sim.get_base_position("object"))
        return object_position

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any] = {}) -> np.ndarray:
        is_grasped = super().is_object_grasped(achieved_goal)
        d = distance(achieved_goal, desired_goal)
        is_lifted = d < self.threshold
        return np.array(is_grasped and is_lifted, dtype=bool)

    def compute_reward(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any]
    ) -> np.ndarray:
        is_grasped = self.is_object_grasped(achieved_goal)
        d = distance(achieved_goal, desired_goal)

        if is_grasped and d < self.threshold:
            return np.array(0.0, dtype=np.float32)  # Success

        if self.reward_type == "sparse":
            return np.array(-1.0, dtype=np.float32)  # Failure
        else:
            if not is_grasped:
                reward = -distance(achieved_goal, np.array(self.robot.get_ee_position())) -0.5
            else:
                reward = -distance(achieved_goal, desired_goal)
            return reward.astype(np.float32)


class PickAndPlaceTask(LiftTask):
    """Stage 5: Pick the object and move it to a goal zone"""

    def reset(self) -> None:
        object_position = self._sample_object()
        self.goal = self._sample_goal()
        self.sim.set_base_pose("object", object_position, np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("target", self.goal, np.array([0.0, 0.0, 0.0, 1.0]))


    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any] = {}) -> np.ndarray:
        return np.array(distance(achieved_goal, desired_goal) < self.threshold, dtype=bool)

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any] = {}) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        if self.reward_type == "sparse":
            return -np.array(d > self.threshold, dtype=np.float32)
        else:
            return -d.astype(np.float32)