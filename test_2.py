#!/usr/bin/env python3
import rospy
from interbotix_xs_msgs.msg import JointSingleCommand
from control_msgs.msg import JointTrajectoryControllerState
from sensor_msgs.msg import JointState

class GraspDetector:
    def __init__(self):

        rospy.init_node("grasp_detector")
        self.pub = rospy.Publisher(
            "/locobot/commands/joint_single",
            JointSingleCommand,
            queue_size=1
        )

        self.effort_thresh = 250
        self.curr_effort = 0.0
        rospy.Subscriber(
            "/locobot/dynamixel/joint_states",
            JointState,
            self._joint_states_cb,
            queue_size=1
        )

        rospy.sleep(0.5)

    def _joint_states_cb(self, msg: JointState):
        # map names→effort
        efforts = dict(zip(msg.name, msg.effort))
        # take the max of the two fingers
        self.curr_effort =  abs(efforts.get("gripper", 0.0))
        if self.curr_effort > self.effort_thresh:
            pass
            #print(self.curr_effort)
        

    def _state_cb(self, msg: JointTrajectoryControllerState):
        self.state = msg

    def close_until_grasp(self,
                         max_open=1.5,      # now your open position
                         min_closed=0.0,   # your closed position
                         step=0.1,          # tune step size
                         error_thresh=0.02, # tune threshold (rad)
                         pause=0.3):        # allow time to move
        width = max_open
        while width >= min_closed:
            # command the next smaller opening
            cmd = JointSingleCommand(name="gripper", cmd=width)
            self.pub.publish(cmd)
            rospy.sleep(pause)

            width -= step
            if self.curr_effort > self.effort_thresh:
                print("object detected")
                return True

        rospy.logwarn("❌ No object detected before fully closed")
        return False

if __name__ == "__main__":
    gd = GraspDetector()
    # 1) Open fully to 1.5 rad
    open_cmd = JointSingleCommand(name="gripper", cmd=1.5)
    gd.pub.publish(open_cmd)
    rospy.sleep(1.0)

    # 2) Now close down until we hit something
    gd.close_until_grasp(
        max_open=1.5,
        min_closed=0.05,
        step=0.05,
        error_thresh=0.02,
        pause=0.1
    )
