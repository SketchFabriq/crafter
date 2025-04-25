import os
import time
import numpy as np
import gymnasium as gym
import pybullet as p
import visioncraft


def widowx_teleop():
    # Create the environment
    env = gym.make(
        "WidowXReach-v0",
        render_mode="human",
        max_episode_steps=1000,
        control_type="ee",
        reward_type="dense",
    )
    obs, _ = env.reset()

    # Get the PyBullet client ID

    # Movement parameters - increase lateral step size
    lateral_step = 0.03  # Larger step size for X/Y movement
    vertical_step = 0.01  # Keep original step size for Z movement
    gripper_step = 0.01  # Gripper movement amount

    # Key mappings (using PyBullet key constants)
    # WASD for X/Y movement
    KEY_1 = ord("1")
    KEY_2 = ord("2")
    KEY_3 = ord("3")
    KEY_4 = ord("4")
    # QE for Z movement
    KEY_5 = ord("5")
    KEY_6 = ord("6")
    # GH for gripper
    KEY_7 = ord("7")
    KEY_8 = ord("8")
    # R for reset
    KEY_9 = ord("9")
    # P for printing position
    KEY_0 = ord("0")
    # ESC for exit
    KEY_Z = ord("z")

    # Instructions
    print("\nWidowX Teleoperation Controls:")
    print("------------------------------")
    print("Position Control:")
    print("  1/2: Forward/Backward (+/- Y)")
    print("  3/4: Left/Right (+/- X)")
    print("  5/6: Up/Down (+/- Z)")
    print("Gripper Control:")
    print("  7/8: Close/Open gripper")
    print("Other:")
    print("  9: Reset robot position")
    print("  0: Print current end-effector position")
    print("  z: Exit program")
    print("------------------------------")

    # Main loop
    running = True
    last_print_time = 0
    last_reset_time = 0

    while running:
        # Get keyboard events
        keys = p.getKeyboardEvents(physicsClientId=0)

        # Default action: no movement
        action = np.zeros(4)  # [x, y, z, gripper]

        # Check keyboard input for movement with different step sizes
        if KEY_1 in keys and keys[KEY_1] & p.KEY_IS_DOWN:  # Forward
            action[1] = lateral_step
        if KEY_2 in keys and keys[KEY_2] & p.KEY_IS_DOWN:  # Backward
            action[1] = -lateral_step
        if KEY_3 in keys and keys[KEY_3] & p.KEY_IS_DOWN:  # Left
            action[0] = -lateral_step
        if KEY_4 in keys and keys[KEY_4] & p.KEY_IS_DOWN:  # Right
            action[0] = lateral_step
        if KEY_5 in keys and keys[KEY_5] & p.KEY_IS_DOWN:  # Up
            action[2] = vertical_step
        if KEY_6 in keys and keys[KEY_6] & p.KEY_IS_DOWN:  # Down
            action[2] = -vertical_step

        # Check keyboard input for gripper
        if KEY_7 in keys and keys[KEY_7] & p.KEY_IS_DOWN:  # Close gripper
            action[3] = -gripper_step
        if KEY_8 in keys and keys[KEY_8] & p.KEY_IS_DOWN:  # Open gripper
            action[3] = gripper_step

        # Reset position (with cooldown to prevent multiple resets)
        current_time = time.time()
        if (
            KEY_9 in keys
            and keys[KEY_9] & p.KEY_IS_DOWN
            and current_time - last_reset_time > 0.5
        ):
            obs, _ = env.reset()
            print("Robot reset to initial position")
            last_reset_time = current_time

        # Print current position (with cooldown to prevent multiple prints)
        if (
            KEY_0 in keys
            and keys[KEY_0] & p.KEY_IS_DOWN
            and current_time - last_print_time > 0.5
        ):
            ee_pos = obs["observation"][:3]
            object_pos = obs["achieved_goal"]
            target_pos = obs["desired_goal"]
            print(f"End-effector position: {ee_pos}")
            print(f"Object position: {object_pos}")
            print(f"Target position: {target_pos}")
            print(f"Distance to object: {np.linalg.norm(ee_pos - object_pos)}")
            print(f"Distance to target: {np.linalg.norm(ee_pos - target_pos)}")
            print("------")
            last_print_time = current_time

        # Exit program
        if KEY_Z in keys and keys[KEY_Z] & p.KEY_IS_DOWN:
            print("Exiting teleoperation...")
            running = False

        # Execute the action
        obs, reward, terminated, truncated, info = env.step(action)

        # Add small delay to reduce CPU usage
        time.sleep(0.01)

        # Check if episode is done
        if terminated or truncated:
            obs, _ = env.reset()

    env.close()


if __name__ == "__main__":
    widowx_teleop()
