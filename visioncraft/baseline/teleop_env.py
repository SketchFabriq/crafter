#!/usr/bin/env python3
import pygame
import gymnasium as gym
import numpy as np
import sys
import os
import visioncraft

def main():

    env_name = "LocobotTouch-v0" # WidowXGrasp-v0, WidowXLift-v0, WidowXPickPlace-v0

    # Initialize Pygame
    pygame.init()
    pygame.display.set_caption(f"Robot Arm Control ({env_name})")
    screen_width, screen_height = 1024, 768
    screen = pygame.display.set_mode((screen_width, screen_height))
    font = pygame.font.SysFont(None, 36)
    clock = pygame.time.Clock()

    # Create environment with rgb_array rendering
    env = gym.make(env_name, render_mode="rgb_array", max_episode_steps=10000)
    observation, info = env.reset()

    # Initialize command-based action vector [x, y, z, pinch]
    # 1 = positive direction, -1 = negative direction, 0 = no movement
    action = np.zeros(4)

    # Control status
    running = True
    paused = False

    # Help text
    control_text = [
        "W/S - Y axis movement (forward/backward)",
        "A/D - X axis movement (left/right)",
        "Q/E - Z axis movement (up/down)",
        "Space - Pinch command (hold for close)",
        "R - Reset environment",
        "P - Pause/Resume",
        "Esc - Quit",
    ]

    # Get first render to determine dimensions
    render_img = env.render()
    render_height, render_width = render_img.shape[:2]

    # Calculate scaling factor to fit in the top portion of the window
    max_render_height = screen_height - 250  # Reserve space for controls
    scale_factor = min(screen_width / render_width, max_render_height / render_height)
    scaled_width = int(render_width * scale_factor)
    scaled_height = int(render_height * scale_factor)

    # Main game loop
    while running:
        screen.fill((0, 0, 0))

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_p:
                    paused = not paused
                elif event.key == pygame.K_r:
                    # Reset environment
                    observation, info = env.reset()
                    action = np.zeros(4)

        if not paused:
            # Reset action vector each frame
            action = np.zeros(4)

            # Get current key states for command input
            keys = pygame.key.get_pressed()

            # Set command values based on key presses
            # X-axis commands
            if keys[pygame.K_d]:  # Right/positive X
                action[1] = -1.0
            elif keys[pygame.K_a]:  # Left/negative X
                action[1] = 1.0

            # Y-axis commands
            if keys[pygame.K_w]:  # Forward/positive Y
                action[0] = 1.0
            elif keys[pygame.K_s]:  # Backward/negative Y
                action[0] = -1.0

            # Z-axis commands
            if keys[pygame.K_q]:  # Up/positive Z
                action[2] = -1.0
            elif keys[pygame.K_e]:  # Down/negative Z
                action[2] = 1.0

            # Pinch commands
            if keys[pygame.K_SPACE]:  # Close pincher
                action[3] = 1.0
            else:  # Open pincher
                action[3] = -1.0

            # Take a step in the environment
            observation, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                observation, info = env.reset()
                action = np.zeros(4)

        # Render the environment and display on screen
        render_img = env.render()

        # Convert numpy array to pygame surface
        render_surface = pygame.surfarray.make_surface(render_img.swapaxes(0, 1))

        # Scale the render image to fit in the window
        render_surface = pygame.transform.scale(
            render_surface, (scaled_width, scaled_height)
        )

        # Display the render image centered at the top of the screen
        render_x = (screen_width - scaled_width) // 2
        screen.blit(render_surface, (render_x, 20))

        # Display control information
        y_offset = scaled_height + 40
        for line in control_text:
            text_surface = font.render(line, True, (255, 255, 255))
            screen.blit(text_surface, (20, y_offset))
            y_offset += 30

        # Display current command values
        command_text = f"Commands: X={action[0]:.1f}, Y={action[1]:.1f}, Z={action[2]:.1f}, Pinch={action[3]:.1f}"
        command_surface = font.render(command_text, True, (0, 255, 0))
        screen.blit(command_surface, (20, y_offset + 10))

        # Display pause status if paused
        if paused:
            pause_surface = font.render("PAUSED - Press P to resume", True, (255, 0, 0))
            pause_rect = pause_surface.get_rect(
                center=(screen_width // 2, screen_height // 2)
            )
            screen.blit(pause_surface, pause_rect)

        pygame.display.flip()
        clock.tick(30)  # 30 FPS

    # Clean up
    env.close()
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()