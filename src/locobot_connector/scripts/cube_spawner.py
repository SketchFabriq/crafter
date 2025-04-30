#!/usr/bin/env python3

import rospy
import random
import ast
from gazebo_msgs.srv import SpawnModel, DeleteModel
from geometry_msgs.msg import Pose

class CubeSpawner:
    def __init__(self):
        rospy.init_node('cube_spawner', anonymous=True)

        # wait for Gazebo services to be available
        rospy.wait_for_service('/gazebo/spawn_sdf_model')
        rospy.wait_for_service('/gazebo/delete_model')

        self.spawn_model = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        self.delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)

        # SDF template with internal pose zeroed out
        self.cube_template = """
        <sdf version="1.5">
        <model name="cube_{0}">
          <pose>0 0 0 0 0 0</pose>
          <link name="link">
            <inertial>
              <mass>0.01</mass>
              <inertia>
                <ixx>0.0001</ixx>
                <iyy>0.0001</iyy>
                <izz>0.0001</izz>
                <ixy>0.0</ixy>
                <ixz>0.0</ixz>
                <iyz>0.0</iyz>
              </inertia>
            </inertial>
            <visual name="visual">
              <geometry>
                <box>
                  <size>0.03 0.03 0.03</size>
                </box>
              </geometry>
              <material>
                <ambient>{1}</ambient>
                <diffuse>{2}</diffuse>
                <specular>0.1 0.1 0.1 1</specular>
              </material>
            </visual>
            <collision name="collision">
              <geometry>
                <box>
                  <size>0.03 0.03 0.03</size>
                </box>
              </geometry>
            </collision>
          </link>
        </model>
      </sdf>
      """
        
        self.left_side_scenario = [
            ((0.30, 0.20, 0.015),  "0 0 1 1"),        # blue
            ((0.40, 0.30, 0.015),  "0.5 0.5 0.5 1"),  # grey
            ((0.30, 0.30, 0.015),  "0 0 0 1"),        # black
            ((0.40, 0.20, 0.015),  "1 1 1 1"),        # white
        ]

    def generate_random_color(self):
        r = random.random()
        g = random.random()
        b = random.random()
        return f"{r} {g} {b} 1"

    def spawn_cubes(self, num_cubes, pattern=None, positions=None):
        # Delete any old cubes
        for i in range(100):
            try:
                self.delete_model(f"cube_{i}")
            except:
                pass
        # Special case for short demonstration
        if pattern == "scenario":
            for i, (pos, colour) in enumerate(self.left_side_scenario):
                model_xml = self.cube_template.format(i, colour, colour)

                spawn_pose = Pose()
                spawn_pose.position.x, spawn_pose.position.y, spawn_pose.position.z = pos

                try:
                    self.spawn_model(f"cube_{i}", model_xml, "", spawn_pose, "world")
                    rospy.loginfo(f"[scenario] cube_{i}: {colour} at {pos}")
                except Exception as e:
                    rospy.logerr(f"Failed to spawn cube_{i}: {e}")
            return  # Skip everything else

        # Spawn new cubes
        for i in range(num_cubes):
            # determine target position
            if positions and i < len(positions):
                # Use provided position
                pos = positions[i]
                x = pos[0]
                y = pos[1]
                z = pos[2] if len(pos) > 2 else 0.015
            elif pattern == "line":
                x = 0.5
                y = -0.2 + (i * 0.1)
                z = 0.015
            elif pattern == "grid":
                x = 0.5 + ((i % 3) * 0.1)
                y = -0.2 + ((i // 3) * 0.1)
                z = 0.015
            else:  # random or any other
                x = 0.5 + random.uniform(-0.3, 0.3)
                y = random.uniform(-0.3, 0.3)
                z = 0.015

            # pick color
            if pattern == "rainbow":
                color = self.generate_random_color()
            else:
                color = "0.8 0.2 0.2 1"  # default red

            # fill the SDF (internal pose zeroed)
            model_xml = self.cube_template.format(i, color, color)

            # build the actual spawn pose
            spawn_pose = Pose()
            spawn_pose.position.x = x
            spawn_pose.position.y = y
            spawn_pose.position.z = z
            # orientation quaternion defaults to (0,0,0,1)

            try:
                self.spawn_model(
                    model_name=f"cube_{i}",
                    model_xml=model_xml,
                    robot_namespace="",
                    initial_pose=spawn_pose,
                    reference_frame="world"
                )
                rospy.loginfo(f"Spawned cube {i} at ({x:.2f}, {y:.2f}, {z:.2f})")
            except Exception as e:
                rospy.logerr(f"Failed to spawn cube {i}: {e}")

if __name__ == '__main__':
    try:
        spawner = CubeSpawner()
        num_cubes = rospy.get_param('~num_cubes', 1)
        pattern = rospy.get_param('~pattern', 'random')
        positions_str = rospy.get_param('~positions', '[]')
        
        # Convert string representation of list to actual list
        try:
            positions = ast.literal_eval(positions_str)
        except:
            rospy.logwarn("Invalid positions parameter. Using default positions.")
            positions = None
            
        spawner.spawn_cubes(num_cubes, pattern, positions)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
