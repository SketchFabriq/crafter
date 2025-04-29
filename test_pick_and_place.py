import os
os.environ["ROS_IP"] = "192.168.0.190"
os.environ["ROS_MASTER_URI"] = "http://192.168.0.17:11311"

from visioncraft.locobot.camera import LocobotCamera
from visioncraft.locobot.arm_control import LocobotArmControl
import numpy as np
import matplotlib.pyplot as plt


cam = LocobotCamera()
arm_control = LocobotArmControl(use_simulation=False)

arm_control.open_gripper()

arm_control.close_gripper()

height, width = cam.get_image().shape[:2]
center_x = 250
center_y = 340

plt.imshow(cam.get_image())
plt.scatter(center_x, center_y, color='red')
plt.show()
plt.imshow(cam.depth)
plt.scatter(center_x, center_y, color='red')
plt.show()

point_camera = cam.depth[center_x, center_y]
print(point_camera)

point_cam = cam.depth_to_xyz(center_x, center_y)
print(point_cam)

point_base = cam.get_target_coordinate_from_camera(point_cam)
print(point_base)

print(f"Point in base frame: {point_base}")

arm_control.pick(point_base, size=0.03)

point_base_2 = point_base + np.array([0.1, 0.1, 0.03])
arm_control.place(point_base_2, size=0.03)
arm_control.open_gripper()


