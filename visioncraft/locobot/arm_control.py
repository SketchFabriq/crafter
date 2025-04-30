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
from interbotix_xs_msgs.msg import JointSingleCommand

class LocobotArmControl:

    JOINT_HOME = [0, 0, 0, 0, 0, 0]
    JOINT_SLEEP = [0.0015339808305725455, -1.1090681552886963, 1.5646604299545288, -0.0015339808305725455, 0.5016117095947266, -0.0015339808305725455]

    offset_real = [0, 0.02, 0.0]

    def __init__(self,
                 use_simulation: bool = True,
                 joint_action_topic: str = '/locobot/arm_controller/follow_joint_trajectory',
                 gripper_topic: str = '/locobot/gripper_controller/command',
                 moveit_ns: str = 'locobot',
                 moveit_group: str = 'interbotix_arm',
                 planning_time: float = 5.0):

        self.use_simulation = use_simulation
        self.is_holding = False
        self.effort_thresh = 280
        self.gripper_effort = 0.0
        self.joint_states = None
        self.efforts = None
        
        if not rospy.core.is_initialized():
            rospy.init_node('locobot_arm_control', anonymous=True)


        # Joint action client
        self._traj_client = actionlib.SimpleActionClient(
            joint_action_topic,
            FollowJointTrajectoryAction
        )
        rospy.loginfo(f"Waiting for joint trajectory action on {joint_action_topic}…")
        self._traj_client.wait_for_server()
        rospy.loginfo("Joint‐space action server ready")

        # Gripper single action client
        if self.use_simulation:
            self.gripper_pub = rospy.Publisher(
                    "/locobot/gripper_controller/command",
                    JointTrajectory,
                    queue_size=1
                )
            self.gripper_joint_names = ['left_finger', 'right_finger']
        else:
            self.gripper_pub = rospy.Publisher(
                "/locobot/commands/joint_single",
                JointSingleCommand,
                queue_size=1
            )
            self.gripper_joint_names = ['gripper']


    
        self.arm_joint_names = [
            'waist', 'shoulder', 'elbow',
            'forearm_roll', 'wrist_angle', 'wrist_rotate'
        ]


        # Initialize MoveIt
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

        # Motor effort callback
        if not use_simulation:
            rospy.Subscriber(
                "/locobot/dynamixel/joint_states",
                JointState,
                self._joint_states_cb,
                queue_size=1
            )

        rospy.sleep(0.5)
        rospy.loginfo("LocobotArmControl initialized")



    def _joint_states_cb(self, msg: JointState):
        self.joint_states = msg
        self.efforts = dict(zip(msg.name, msg.effort))
        self.gripper_effort =  abs(self.efforts.get("gripper", 0.0))



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
        success, traj, _, _ = plan
        if not success:
            rospy.logwarn("MoveIt planning failed")
            return False

        self.group.execute(traj, wait=wait)
        self.group.stop()
        self.group.clear_pose_targets()
        return True

    def move_gripper(self, width: float, duration: float = 1.0):
        """Open/close gripper to given width"""
        max_width = 1.1
        width = max(0.0, min(max_width, width)) # Empiracal limits

        if self.use_simulation:
            traj = JointTrajectory()
            pt = JointTrajectoryPoint ()

            traj.joint_names = self.gripper_joint_names
            pt.positions = [width, -width]
            pt.time_from_start = rospy.Duration(duration)
            traj.points = [pt]
            traj.header.stamp = rospy.Time.now()
        
            self.gripper_pub.publish(traj)
            rospy.sleep(duration)
        else:
            self.is_holding = False
            if width != 0.0:
                cmd = JointSingleCommand(name="gripper", cmd=width)
                self.gripper_pub.publish(cmd)
                rospy.sleep(0.1)
            else:
                target_width = self.joint_states.position[6]
                actual_width = self.joint_states.position[6]
                while actual_width >= 0.0: # Close to desired width
                    target_width -= 0.1
                    cmd = JointSingleCommand(name="gripper", cmd=target_width)
                    self.gripper_pub.publish(cmd)
                    
                    while actual_width > (target_width + 0.02):
                        rospy.sleep(0.05)
                        actual_width = self.joint_states.position[6]
                        # print("Gripper closing")
                        # print(self.gripper_effort)

                        if self.gripper_effort > self.effort_thresh:
                            print("object detected")
                            self.is_holding = True
                            return True

                rospy.logwarn("❌ No object detected before fully closed")
        return False
        
    
    def open_gripper(self):
        self.move_gripper(2.0)

    def close_gripper(self):
        self.move_gripper(0.0)           


    def pick(self, coordinate: list, size: int = 0.03):
        # Create target poses
        if not self.use_simulation:
            coordinate += np.array(self.offset_real)

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
        grab_pose.position.z = max(0.03, min(0.4, coordinate[2]))
        grab_pose.orientation = target_pose.orientation

        # Execute the pick sequence
        self.open_gripper()  # Open gripper
        self.go_to_pose(target_pose)  # Move to approach position
        self.go_to_pose(grab_pose)  # Move down to grab
        self.close_gripper() # Close gripper

    def place(self, coordinate: list, size: int = 0.03):
        # Create target poses
        if not self.use_simulation:
            coordinate += np.array(self.offset_real)
    
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

        # Create the drop pose
        drop_pose = Pose()
        drop_pose.position.x = coordinate[0]
        drop_pose.position.y = coordinate[1]
        drop_pose.position.z = max(0.01, min(0.4, coordinate[2]))
        drop_pose.orientation = target_pose.orientation
        
        # Execute the place sequence
        self.go_to_pose(target_pose)
        self.go_to_pose(drop_pose)
        self.move_gripper(size * 2)  # Open gripper

    def shutdown(self):
        moveit_commander.roscpp_shutdown()


if __name__ == '__main__':
    arm = LocobotArmControl()

    # sleep = 
    arm.move_arm_joints(LocobotArmControl.JOINT_SLEEP)

    # target = Pose()
    # target.position.x = 0.3
    # target.position.y = 0.0
    # target.position.z = 0.4
    # target.orientation.w = 1.0
    # arm.go_to_pose(target)

    # arm.move_gripper(0.05)
    # arm.move_gripper(0.0)

    arm.shutdown()