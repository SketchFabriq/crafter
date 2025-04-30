#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from visioncraft.locobot.camera import LocobotCamera
from visioncraft.locobot.arm_control import LocobotArmControl
import rospy
from visioncraft.vlm.vlm_pipeline import VLMPipeline
import os
from geometry_msgs.msg import Pose

use_latte = False
is_local = False

if is_local:
    os.environ["ROS_IP"] = "127.0.0.1"
    os.environ["ROS_MASTER_URI"] = "http://127.0.0.1:11311"
else:
    os.environ["ROS_IP"] = "192.168.0.17"
    os.environ["ROS_MASTER_URI"] = "http://192.168.0.17:11311"

x_start, x_step, x_count = 0.1, 0.05, 10
y_start, y_step, y_count = -0.1, -0.05, 10
z_val = 0.01

placing_grid = [
    [[round(x_start + i * x_step, 2), round(y_start + j * y_step, 2), z_val]
     for j in range(y_count)]
    for i in range(x_count)
]    

def pick_block_action(action, cam, arm_control):
    print(f"Pick block Action: {action}")    
    pos_init = action['pos_init']
    height, width = cam.image.shape[:2]
    x = int(pos_init["x"]/100 * width)
    y = int(pos_init["y"]/100 * height)
    point_cam = cam.get_point_at_pixel(x, y)
    point_base = cam.get_target_coordinate_from_camera(point_cam)
    arm_control.pick(point_base)


def place_block_action(action, cam, arm_control):
    pos_finale = action['pos_finale']
    x = int(pos_finale["x"])
    y = int(pos_finale["y"])
    arm_control.place(placing_grid[x][y])


cam = LocobotCamera(use_simulation=False)
arm_control = LocobotArmControl(use_simulation=False)

arm_control.move_arm_joints(LocobotArmControl.JOINT_SLEEP)

cam.pan(0.5)
cam.tilt(0.8)
img = cam.get_image()
    
# Visualize depth map and image
plt.imsave('./resources/rgb_image.png', img)

vlm_pipeline = VLMPipeline(model_name="gpt-4o", openai_api_key=os.environ.get("OPENAI_API_KEY"))

vlm_pipeline.set_current_image("./resources/rgb_image.png")

vlm_pipeline.set_goal_image("./resources/Ldraw.png")

json_answer = vlm_pipeline.run_pipeline() # In this pipeline is where we can add the own verification of pointer and some other upgrades
step = 0
for actions in json_answer:
        for action in json_answer[actions]:
            print(f"Step {step}")
            # Each action to accomplish
            pick_block_action(action, cam, arm_control)
            # latte_trajectory(action, latest_pose, arm_control) # Comment this if latte doesnt end up working
            place_block_action(action, cam, arm_control)


arm_control.move_arm_joints(LocobotArmControl.JOINT_SLEEP)