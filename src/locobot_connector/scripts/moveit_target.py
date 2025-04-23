#!/usr/bin/env python3

import rospy
import sys
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
import time

class LocobotArmControl:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('locobot_arm_control', anonymous=True)
        
        # Create publisher for arm controller
        self.arm_pub = rospy.Publisher(
            '/locobot/arm_controller/command',
            JointTrajectory,
            queue_size=10
        )
        
        # Create publisher for gripper controller
        self.gripper_pub = rospy.Publisher(
            '/locobot/gripper_controller/command',
            JointTrajectory,
            queue_size=10
        )
        
        # Define joint names (based on your rostopic list output)
        self.arm_joint_names = ['waist', 'shoulder', 'elbow', 'forearm_roll', 'wrist_angle', 'wrist_rotate']
        self.gripper_joint_names = ['left_finger', 'right_finger']
        
        # Subscribe to joint states to get current position
        self.joint_states = None
        rospy.Subscriber('/locobot/joint_states', JointState, self.joint_states_callback)
        
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
        """Move arm to specified joint positions
        
        Args:
            positions: List of joint positions [waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate]
            duration: Time to reach target position in seconds
        """
        # Validate positions
        if len(positions) != len(self.arm_joint_names):
            print(f"Warning: Expected {len(self.arm_joint_names)} joint positions, got {len(positions)}")
            if len(positions) < len(self.arm_joint_names):
                # Pad with zeros
                positions.extend([0.0] * (len(self.arm_joint_names) - len(positions)))
            else:
                # Truncate
                positions = positions[:len(self.arm_joint_names)]
                
        # Create trajectory message
        traj = JointTrajectory()
        traj.joint_names = self.arm_joint_names
        
        # Create trajectory point
        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start = rospy.Duration(duration)
        
        # Add point to trajectory
        traj.points = [point]
        
        # Set header timestamp
        traj.header.stamp = rospy.Time.now()
        
        # Publish trajectory
        print(f"Moving arm to positions: {positions}")
        self.arm_pub.publish(traj)
        
        # Wait for motion to complete
        rospy.sleep(duration + 0.5)
        
    def move_gripper(self, width, duration=1.0):
        """Move gripper to specified width
        
        Args:
            width: Width between fingers (0.0 is closed, 0.05 is fully open)
            duration: Time to reach target position in seconds
        """
        # Ensure width is in valid range
        width = max(0.0, min(0.05, width))
        
        # For Locobot gripper, both fingers move symmetrically
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
        
        # Set header timestamp
        traj.header.stamp = rospy.Time.now()
        
        # Publish trajectory
        print(f"Moving gripper to width: {width} (finger positions: {finger_pos})")
        self.gripper_pub.publish(traj)
        
        # Wait for motion to complete
        rospy.sleep(duration + 0.5)
        

def main():
    try:
        # Create arm control object
        arm_control = LocobotArmControl()
        
        # Define preset positions
        positions = {
            'home': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            'up': [0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
            'forward': [0.0, 0.0, -1.5, 0.0, 0.0, 0.0],
            'side': [1.5, 0.0, 0.0, 0.0, 0.0, 0.0],
            'ready': [0.0, -0.5, -0.5, 0.0, 0.0, 0.0]
        }
        
        # Main control loop
        while not rospy.is_shutdown():
            print("\n===== Locobot Arm Control =====")
            print("1: Move to home position")
            print("2: Move to up position")
            print("3: Move to forward position")
            print("4: Move to side position")
            print("5: Move to ready position")
            print("6: Move to custom position")
            print("7: Open gripper")
            print("8: Close gripper")
            print("9: Set custom gripper width")
            print("0: Show current joint positions")
            print("q: Quit")
            
            choice = input("\nEnter your choice: ")
            
            if choice == '1':
                arm_control.move_arm(positions['home'])
            elif choice == '2':
                arm_control.move_arm(positions['up'])
            elif choice == '3':
                arm_control.move_arm(positions['forward'])
            elif choice == '4':
                arm_control.move_arm(positions['side'])
            elif choice == '5':
                arm_control.move_arm(positions['ready'])
            elif choice == '6':
                print("\nEnter 6 joint positions for [waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate]")
                print("Example: 0.0 -0.5 -0.5 0.0 0.0 0.0")
                try:
                    pos_input = input("> ")
                    joint_pos = [float(x) for x in pos_input.split()]
                    arm_control.move_arm(joint_pos)
                except ValueError:
                    print("Invalid input. Please enter numbers separated by spaces.")
            elif choice == '7':
                arm_control.move_gripper(0.05)  # Fully open
            elif choice == '8':
                arm_control.move_gripper(0.0)   # Fully closed
            elif choice == '9':
                try:
                    width = float(input("Enter gripper width (0.0 to 0.05): "))
                    arm_control.move_gripper(width)
                except ValueError:
                    print("Invalid input. Please enter a number.")
            elif choice == '0':
                arm_control.print_current_positions()
            elif choice.lower() == 'q':
                print("Exiting...")
                break
            else:
                print("Invalid choice")
                
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        print("\nProgram terminated by user")

if __name__ == '__main__':
    main()