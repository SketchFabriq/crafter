#!/usr/bin/env python3
import os
import time
import pybullet as p
import pybullet_data
import pkg_resources

# X11 MIT-SHM fix
os.environ['QT_X11_NO_MITSHM'] = '1'

# Start GUI
p.connect(p.GUI)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

# Add plane and gravity
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("plane.urdf")
p.setGravity(0, 0, -9.81)

# Get the path to the URDF file from the visioncraft package
urdf_path = pkg_resources.resource_filename(
    'visioncraft',
    'envs/assets/widowx/urdf/wx250.urdf'
)
urdf_path = pkg_resources.resource_filename(
    'visioncraft',
    'envs/assets/locobot/locobot_full.urdf'
)



print(f"Loading URDF from: {urdf_path}")

# Add the URDF’s directory to the search path
urdf_dir = os.path.dirname(urdf_path)
p.setAdditionalSearchPath(urdf_dir)

# Load the robot with fixed base (so wheels don't move)
robot_id = p.loadURDF(
    "locobot_full.urdf",
    basePosition=[0, 0, 0],
    useFixedBase=True
)
print(f"Successfully loaded robot with ID: {robot_id}\n")

# Helper to map joint type constants to human-readable
joint_type_map = {
    p.JOINT_REVOLUTE:    "REVOLUTE",
    p.JOINT_PRISMATIC:   "PRISMATIC",
    p.JOINT_SPHERICAL:   "SPHERICAL",
    p.JOINT_PLANAR:      "PLANAR",
    p.JOINT_FIXED:       "FIXED",
    p.JOINT_POINT2POINT: "POINT2POINT",
    p.JOINT_GEAR:        "GEAR",
}

# Print out every joint
num_joints = p.getNumJoints(robot_id)
print(f"Number of joints: {num_joints}\n")
for i in range(num_joints):
    info = p.getJointInfo(robot_id, i)
    idx        = info[0]
    name       = info[1].decode('utf-8')
    jtype_code = info[2]
    jtype      = joint_type_map.get(jtype_code, f"UNKNOWN({jtype_code})")
    print(f"  Joint {idx:2d}: name = {name:20s} | type = {jtype}")

# Run simulation
while True:
    p.stepSimulation()
    time.sleep(1/240.)
