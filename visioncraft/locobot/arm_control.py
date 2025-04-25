import numpy as np
import rospy
import sys
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
import time
import actionlib
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal


class LocobotArmControl:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('locobot_arm_control', anonymous=True)

        self._traj_client = actionlib.SimpleActionClient(
            '/locobot/arm_controller/follow_joint_trajectory',
            FollowJointTrajectoryAction
        )
        self._traj_client.wait_for_server()

        # Create publisher for gripper controller
        self.gripper_pub = rospy.Publisher(
            '/locobot/gripper_controller/command',
            JointTrajectory,
            queue_size=10
        )

        # Define joint names (based on your rostopic list output)
        self.arm_joint_names = ['waist', 'shoulder', 'elbow',
                                'forearm_roll', 'wrist_angle', 'wrist_rotate']
        self.gripper_joint_names = ['left_finger', 'right_finger']

        # Subscribe to joint states to get current position
        self.joint_states = None
        rospy.Subscriber('/locobot/joint_states', JointState,
                         self.joint_states_callback)

        # Wait for publisher connections
        rospy.sleep(1.0)
        print("Locobot arm control initialized")

    def joint_states_callback(self, data):
        """Store the latest joint states data"""
        self.joint_states = data

    def get_current_joint_positions(self):
        """Get current joint positions if available"""
        if self.joint_states is None:
            print("Warning: No joint states received yet")
            return None

        positions = {}
        for name, pos in zip(self.joint_states.name, self.joint_states.position):
            positions[name] = pos

        return positions

    def print_current_positions(self):
        """Print current joint positions"""
        positions = self.get_current_joint_positions()
        if positions:
            print("\nCurrent Joint Positions:")
            for joint in self.arm_joint_names:
                if joint in positions:
                    print(f"  {joint}: {positions[joint]:.4f}")
                else:
                    print(f"  {joint}: unknown")

            for joint in self.gripper_joint_names:
                if joint in positions:
                    print(f"  {joint}: {positions[joint]:.4f}")
                else:
                    print(f"  {joint}: unknown")
        else:
            print("Could not retrieve current joint positions")

    def move_arm(self, positions, duration=2.0):
        from trajectory_msgs.msg import JointTrajectoryPoint

        # build the JointTrajectory inside a goal
        goal = FollowJointTrajectoryGoal()
        goal.trajectory.joint_names = self.arm_joint_names
        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start = rospy.Duration(duration)
        goal.trajectory.points = [point]

        # send and wait
        self._traj_client.send_goal(goal)
        self._traj_client.wait_for_result()   # <-- blocks until done
        status = self._traj_client.get_state()
        if status != actionlib.GoalStatus.SUCCEEDED:
            rospy.logwarn(f"Arm trajectory failed with status {status}")

    def move_gripper(self, width, duration=1.0):
        width = max(0.0, min(0.05, width))

        finger_pos = width / 2.0

        # Create trajectory message
        traj = JointTrajectory()
        traj.joint_names = self.gripper_joint_names

        # Create trajectory point
        point = JointTrajectoryPoint()
        point.positions = [finger_pos, finger_pos]
        point.time_from_start = rospy.Duration(duration)

        # Add point to trajectory
        traj.points = [point]

        # Publish gripper message
        traj.header.stamp = rospy.Time.now()
        self.gripper_pub.publish(traj)

        # Wait for motion to complete
        rospy.sleep(duration + 0.5)


if __name__ == '__main__':
    arm_control = LocobotArmControl()

    # Define preset positions
    positions = {
        'home': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        'up': [0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
        'forward': [0.0, 0.0, -1.5, 0.0, 0.0, 0.0],
        'side': [1.5, 0.0, 0.0, 0.0, 0.0, 0.0],
        'ready': [0.0, -0.5, -0.5, 0.0, 0.0, 0.0]
    }

    arm_control.move_arm(positions['side'])

    arm_control.move_gripper(0.05)  # Fully open
    arm_control.move_gripper(0.0)   # Fully closed

    arm_control.print_current_positions()

    trajectory_example = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.3, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.3, 0.1, 0.0, 0.0, 0.0, 0.0],
        [0.3, 0.2, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.5, 0.0, 0.0, 0.0, 0.0],
    ]

    speed = 0.5
    for i, traj in enumerate(trajectory_example):
        if i >= 1:
            past_traj = trajectory_example[i-1]
        else:
            past_traj = [1.5, 0.0, 0.0, 0.0, 0.0, 0.0]

        time_needed = np.linalg.norm(np.array(traj)-np.array(past_traj))/speed
        arm_control.move_arm(traj, time_needed)
