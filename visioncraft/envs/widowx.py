from typing import Optional

import numpy as np
from gymnasium import spaces

from panda_gym.envs.core import PyBulletRobot
from panda_gym.pybullet import PyBullet
import os


class WidowX(PyBulletRobot):
    """WidowX robot in PyBullet.

    Args:
        sim (PyBullet): Simulation instance.
        block_gripper (bool, optional): Whether the gripper is blocked. Defaults to False.
        base_position (np.ndarray, optional): Position of the base of the robot, as (x, y, z). Defaults to (0, 0, 0).
        control_type (str, optional): "ee" to control end-effector displacement or "joints" to control joint angles.
            Defaults to "ee".
    """

    def __init__(
        self,
        sim: PyBullet,
        block_gripper: bool = False,
        base_position: Optional[np.ndarray] = None,
        control_type: str = "ee",
        binary_gripper: bool = True,
    ) -> None:
        base_position = base_position if base_position is not None else np.zeros(3)
        self.block_gripper = block_gripper
        self.control_type = control_type
        n_action = (
            3 if self.control_type == "ee" else 6
        )  # control (x, y z) if "ee", else control the 6 arm joints
        n_action += 0 if self.block_gripper else 1
        action_space = spaces.Box(-1.0, 1.0, shape=(n_action,), dtype=np.float32)
        self.use_binary_gripper = binary_gripper

        # Joint indices from URDF: waist, shoulder, elbow, wrist_angle, wrist_rotate, gripper, left_finger, right_finger
        joint_indices = np.array([0, 1, 2, 3, 4, 5, 9, 10])

        # Approximate joint forces based on motor specifications
        joint_forces = np.array(
            [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 1000.0, 1000.0]
        )

        # Get the path to the WidowX URDF file (three level up, then assets/widowx/urdf)
        urdf_path = os.path.join(
            os.path.dirname(__file__), "assets", "widowx", "urdf", "wx250.urdf"
        )

        super().__init__(
            sim,
            body_name="robot_arm",
            file_name=urdf_path,
            base_position=base_position,
            action_space=action_space,
            joint_indices=joint_indices,
            joint_forces=joint_forces,
        )

        self.fingers_indices = np.array([9, 10])  # left_finger and right_finger joints
        self.neutral_joint_values = np.array(
            [0.0, 0.0, 0.0, -np.pi/2, 0.0, 0.0, 0.026, -0.026]
        )  # Neutral pose
        self.ee_link = 11  # ee_gripper_link index

        # Set friction for fingers
        self.sim.set_lateral_friction(
            self.body_name, self.fingers_indices[0], lateral_friction=1.0
        )
        self.sim.set_lateral_friction(
            self.body_name, self.fingers_indices[1], lateral_friction=1.0
        )
        self.sim.set_spinning_friction(
            self.body_name, self.fingers_indices[0], spinning_friction=0.001
        )
        self.sim.set_spinning_friction(
            self.body_name, self.fingers_indices[1], spinning_friction=0.001
        )

    def set_action(self, action: np.ndarray) -> None:
        """Execute the action."""
        action = action.copy()  # ensure action don't change
        action = np.clip(action, self.action_space.low, self.action_space.high)

        if self.control_type == "ee":
            ee_displacement = action[:3]
            target_arm_angles = self.ee_displacement_to_target_arm_angles(
                ee_displacement
            )
        else:
            arm_joint_ctrl = action[:6]
            target_arm_angles = self.arm_joint_ctrl_to_target_arm_angles(arm_joint_ctrl)

        if self.block_gripper:
            target_fingers_width = 0.03  # Default closed position
        else:
            # convert action from [-1, 1] to [0.03, 0.074]
            action_range = [0.03, 0.074]
            converted_action = (action[-1] + 1) / 2 * (
                action_range[1] - action_range[0]
            ) + action_range[0]
            target_fingers_width = converted_action

            if self.use_binary_gripper:
                if action[-1] > 0:
                    target_fingers_width = action_range[1]
                else:
                    target_fingers_width = action_range[0]

        target_angles = np.concatenate(
            (target_arm_angles, [target_fingers_width / 2, -target_fingers_width / 2])
        )
        self.control_joints(target_angles=target_angles)

    def inverse_kinematics(
        self, link: int, position: np.ndarray, orientation: np.ndarray
    ) -> np.ndarray:
        """Override the inverse kinematics for WidowX specific implementation.

        Args:
            link (int): The link.
            position (x, y, z): Desired position of the link.
            orientation (x, y, z, w): Desired orientation of the link.

        Returns:
            np.ndarray: Joint values.
        """
        return self.sim.inverse_kinematics(
            body=self.body_name, link=link, position=position, orientation=orientation
        )

    def ee_displacement_to_target_arm_angles(
        self, ee_displacement: np.ndarray
    ) -> np.ndarray:
        """Compute the target arm angles from the end-effector displacement.

        Args:
            ee_displacement (np.ndarray): End-effector displacement, as (dx, dy, dy).

        Returns:
            np.ndarray: Target arm angles, as the angles of the 6 arm joints.
        """
        ee_displacement = ee_displacement * 0.01  # limit maximum change in position
        ee_position = self.get_ee_position()
        target_ee_position = ee_position + ee_displacement
        target_ee_position[2] = np.max((0.01, target_ee_position[2]))

        # Get current orientation
        current_orientation = self.sim.get_link_orientation(
            self.body_name, self.ee_link
        )
        current_euler = self.sim.physics_client.getEulerFromQuaternion(
            current_orientation
        )

        # Only keep yaw (around z-axis), set roll and pitch to 0
        roll = 0
        pitch = np.pi / 2
        yaw = 0

        # Build quaternion with only yaw preserved
        limited_orientation = self.sim.physics_client.getQuaternionFromEuler(
            [roll, pitch, yaw]
        )

        # Calculate IK with current orientation
        target_arm_angles = self.inverse_kinematics(
            link=self.ee_link,
            position=target_ee_position,
            orientation=limited_orientation,
        )

        return target_arm_angles[:6]  # remove fingers angles

    def arm_joint_ctrl_to_target_arm_angles(
        self, arm_joint_ctrl: np.ndarray
    ) -> np.ndarray:
        """Compute the target arm angles from the arm joint control.

        Args:
            arm_joint_ctrl (np.ndarray): Control of the 6 joints.

        Returns:
            np.ndarray: Target arm angles, as the angles of the 6 arm joints.
        """
        arm_joint_ctrl = arm_joint_ctrl * 0.05  # limit maximum change in position
        current_arm_joint_angles = np.array(
            [self.get_joint_angle(joint=i) for i in range(6)]
        )
        target_arm_angles = current_arm_joint_angles + arm_joint_ctrl
        return target_arm_angles

    def get_obs(self) -> np.ndarray:
        """Get observation of robot state.

        Returns:
            np.ndarray: Observation array containing end-effector position, velocity and gripper width if not blocked
        """
        ee_position = self.get_ee_position()
        ee_velocity = self.get_ee_velocity()

        if not self.block_gripper:
            fingers_width = self.get_fingers_width()
            observation = np.concatenate((ee_position, ee_velocity, [fingers_width]))
        else:
            observation = np.concatenate((ee_position, ee_velocity))
        return observation

    def reset(self) -> None:
        """Reset the robot to its neutral pose."""
        self.set_joint_neutral()

    def set_joint_neutral(self) -> None:
        """Set the robot to its neutral pose."""
        self.set_joint_angles(self.neutral_joint_values)

    def get_fingers_width(self) -> float:
        """Get the distance between the fingers."""
        finger1 = self.sim.get_joint_angle(self.body_name, self.fingers_indices[0])
        finger2 = self.sim.get_joint_angle(self.body_name, self.fingers_indices[1])
        return abs(finger1 - finger2)  # Distance between fingers

    def get_ee_position(self) -> np.ndarray:
        """Returns the position of the end-effector as (x, y, z)"""
        return self.get_link_position(self.ee_link)

    def get_ee_velocity(self) -> np.ndarray:
        """Returns the velocity of the end-effector as (vx, vy, vz)"""
        return self.get_link_velocity(self.ee_link)
