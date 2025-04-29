#!/usr/bin/env python3
import sys
import rospy
import actionlib
import moveit_commander
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose
import numpy as np
from tf.transformations import quaternion_from_euler

class LocobotArmControl:
    def __init__(self,
                 joint_action_topic: str = '/locobot/arm_controller/follow_joint_trajectory',
                 gripper_topic: str = '/locobot/gripper_controller/command',
                 joint_group_names=None,
                 gripper_joint_names=None,
                 moveit_ns: str = 'locobot',
                 moveit_group: str = 'interbotix_arm',
                 planning_time: float = 5.0):
        # --- ROS init ---
        if not rospy.core.is_initialized():
            rospy.init_node('locobot_arm_control', anonymous=True)

        # --- Joint‐space action client ---
        self._traj_client = actionlib.SimpleActionClient(
            joint_action_topic,
            FollowJointTrajectoryAction
        )
        rospy.loginfo(f"Waiting for joint trajectory action on {joint_action_topic}…")
        self._traj_client.wait_for_server()
        rospy.loginfo("Joint‐space action server ready")

        # --- Gripper publisher ---
        self.gripper_pub = rospy.Publisher(
            gripper_topic,
            JointTrajectory,
            queue_size=10
        )

        # --- Joint names ---
        self.arm_joint_names = joint_group_names or [
            'waist', 'shoulder', 'elbow',
            'forearm_roll', 'wrist_angle', 'wrist_rotate'
        ]
        self.gripper_joint_names = gripper_joint_names or ['left_finger', 'right_finger']

        # --- Joint states subscriber (for feedback) ---
        self.joint_states = None
        rospy.Subscriber('/locobot/joint_states',
                         JointState,
                         self._joint_states_cb)

        # --- MoveIt! setup for IK mode ---
        moveit_commander.roscpp_initialize(sys.argv)
        self.robot = moveit_commander.RobotCommander(
            robot_description=f"/{moveit_ns}/robot_description",
            ns=moveit_ns
        )
        self.scene = moveit_commander.PlanningSceneInterface(ns=moveit_ns)
        self.group = moveit_commander.MoveGroupCommander(
            moveit_group,
            robot_description=f"/{moveit_ns}/robot_description",
            ns=moveit_ns
        )
        self.group.set_planning_time(planning_time)
        self.group.allow_replanning(True)

        rospy.sleep(1.0)
        rospy.loginfo("LocobotArmControl initialized")

    def _joint_states_cb(self, msg: JointState):
        self.joint_states = msg

    def get_current_joint_positions(self):
        if not self.joint_states:
            rospy.logwarn("No joint states yet")
            return None
        return {n: p for n, p in zip(self.joint_states.name,
                                     self.joint_states.position)}

    def move_arm_joints(self, positions, duration: float = 2.0):
        """Send a FollowJointTrajectoryGoal to move each arm joint."""
        if len(positions) != len(self.arm_joint_names):
            rospy.logwarn(f"Expected {len(self.arm_joint_names)} joints, got {len(positions)}")
        goal = FollowJointTrajectoryGoal()
        goal.trajectory.joint_names = self.arm_joint_names

        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start = rospy.Duration(duration)
        goal.trajectory.points = [point]

        self._traj_client.send_goal(goal)
        self._traj_client.wait_for_result()
        status = self._traj_client.get_state()
        if status != actionlib.GoalStatus.SUCCEEDED:
            rospy.logwarn(f"Joint trajectory failed: {status}")

    def go_to_pose(self, target_pose: Pose, wait: bool = True):
        """Use MoveIt to plan & execute to a Cartesian end-effector pose."""
        self.group.set_pose_target(target_pose)
        plan = self.group.plan()
        # plan is a tuple (success_flag, plan_msg, planning_time, error_code)
        # for newer MoveIt versions, plan itself is a RobotTrajectory
        if hasattr(plan, 'joint_trajectory'):
            traj = plan
        else:
            success, traj, _, _ = plan
            if not success:
                rospy.logwarn("MoveIt planning failed")
                return False
#        print("Going to pose. Traj calculated : ", traj)
        self.group.execute(traj, wait=wait)
        self.group.stop()
        self.group.clear_pose_targets()
        return True

    def move_gripper(self, width: float, duration: float = 1.0):
        """Open/close gripper to given width (0.0 closed, ~0.05 open)."""
        width = max(0.0, min(0.1, width))
        pos = width / 2.0
        traj = JointTrajectory()
        traj.joint_names = self.gripper_joint_names

        pt = JointTrajectoryPoint()
        pt.positions = [pos, -pos]
        pt.time_from_start = rospy.Duration(duration)
        traj.points = [pt]
        traj.header.stamp = rospy.Time.now()

        self.gripper_pub.publish(traj)
        rospy.sleep(duration + 0.2)

    def pick(self, coordinate: list, size: int = 0.03):
        # Create target poses
        target_pose = Pose()
        target_pose.position.x = coordinate[0]
        target_pose.position.y = coordinate[1]
        target_pose.position.z = max(0.1, coordinate[2] + 0.2) # offset to be on top of the block

        # Create a quaternion for (roll=0, pitch=-90°, yaw=0)
        q = quaternion_from_euler(0, np.pi/2, 0)  # pinch down
        target_pose.orientation.x = q[0]
        target_pose.orientation.y = q[1]
        target_pose.orientation.z = q[2]
        target_pose.orientation.w = q[3]

        # Create the grab pose
        grab_pose = Pose()
        grab_pose.position.x = coordinate[0]
        grab_pose.position.y = coordinate[1]
        grab_pose.position.z = max(0.015, min(0.4, coordinate[2]))
        grab_pose.orientation = target_pose.orientation

        grabbed_pose = Pose()
        grabbed_pose.position.x = coordinate[0]
        grabbed_pose.position.y = coordinate[1]
        grabbed_pose.position.z = 0.2
        grabbed_pose.orientation = target_pose.orientation
        # Execute the pick sequence
        self.move_gripper(size * 2)  # Open gripper
        print("Trying to go to target pick")
        self.go_to_pose(target_pose)  # Move to approach position
        print("Trying to go to target grab")
        self.go_to_pose(grab_pose)  # Move down to grab
        self.move_gripper(size * 0.9)  # Close gripper
        print("Trying to go to target grabbed")
        self.go_to_pose(grabbed_pose) # Lift grabbed object
        return grabbed_pose

    def place(self, coordinate: list, size: int = 0.03):
        print("Trying to place cube at coordinates : ", coordinate)
        # Create target poses
        target_pose = Pose()
        target_pose.position.x = coordinate[0]
        target_pose.position.y = coordinate[1]
        target_pose.position.z = max(0.1, coordinate[2] + 0.2)
        # Create a quaternion for (roll=0, pitch=-90°, yaw=0)
        q = quaternion_from_euler(0, np.pi/2, 0)
        target_pose.orientation.x = q[0]
        target_pose.orientation.y = q[1]
        target_pose.orientation.z = q[2]
        target_pose.orientation.w = q[3]

        print("target pose : ", target_pose)

        # Create the drop pose
        drop_pose = Pose()
        drop_pose.position.x = coordinate[0]
        drop_pose.position.y = coordinate[1]
        drop_pose.position.z = max(0.05, min(0.4, coordinate[2]))
        drop_pose.orientation = target_pose.orientation
        
        # Create the lifted pose
        lifted_pose = Pose()
        lifted_pose.position.x = coordinate[0]
        lifted_pose.position.y = coordinate[1]
        lifted_pose.position.z = 0.2
        lifted_pose.orientation = target_pose.orientation
        # Execute the place sequence
        print("Trying to go to target place")
        self.go_to_pose(target_pose)
        print("Trying to go to drop place")
        self.go_to_pose(drop_pose)
        self.move_gripper(size * 2)  # Open gripper
        print("trying to go to lifted pose")
        self.go_to_pose(lifted_pose)
        
    def shutdown(self):
        moveit_commander.roscpp_shutdown()


if __name__ == '__main__':
    arm = LocobotArmControl()

    home = [0, 0, 0, 0, 0, 0]
    arm.move_arm_joints(home)

    target = Pose()
    target.position.x = 0.3
    target.position.y = 0.0
    target.position.z = 0.4
    target.orientation.w = 1.0
    arm.go_to_pose(target)

    arm.move_gripper(0.05)
    arm.move_gripper(0.0)

    arm.shutdown()
