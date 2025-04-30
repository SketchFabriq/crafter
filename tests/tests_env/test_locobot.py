
import unittest
import visioncraft
import gymnasium as gym


class TestLocobot(unittest.TestCase):
    def test_locobot_sample_space(self):
        """Test locobot action space"""
        env = gym.make("LocobotTouch-v0", render_mode="rgb_array")

        env.reset()
        action_space = env.action_space
        self.assertEqual(action_space.shape, (4,))