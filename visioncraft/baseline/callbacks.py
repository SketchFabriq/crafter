from stable_baselines3.common.callbacks import BaseCallback

class SuccessThresholdCallback(BaseCallback):
    def __init__(
        self, success_threshold=0.8, check_freq=100, window_size=100, verbose=0
    ):
        super().__init__(verbose)
        self.success_threshold = success_threshold
        self.check_freq = check_freq
        self.window_size = window_size
        self.successes = []

    def _on_step(self) -> bool:
        if self.locals.get("infos") is not None:
            for info in self.locals["infos"]:
                if "is_success" in info and "episode" in info:
                    self.successes.append(info["is_success"])

        if len(self.successes) >= self.window_size:
            recent_successes = self.successes[-self.window_size :]
            avg_success = sum(recent_successes) / self.window_size

            if self.verbose > 0:
                print(
                    f"[Callback] Avg success rate over last {self.window_size} episodes: {avg_success:.2f}"
                )

            if avg_success >= self.success_threshold:
                print(
                    f"[Callback] Success threshold reached: {avg_success:.2f} >= {self.success_threshold}"
                )
                return False

        return True


from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np


class NoiseDecayCallback(BaseCallback):
    def __init__(self, n_actions, total_steps,
                 sigma_start=0.2, sigma_end=0.05, verbose=0):
        super().__init__(verbose)
        self.n_actions = n_actions
        self.total_steps = total_steps
        self.sigma_start = sigma_start
        self.sigma_end = sigma_end

    def _on_step(self) -> bool:
        frac = min(self.model.num_timesteps / self.total_steps, 1.0)
        sigma_now = (1 - frac) * self.sigma_start + frac * self.sigma_end

        # Build a fresh noise object and swap it in
        self.model.action_noise = NormalActionNoise(
            mean=np.zeros(self.n_actions, dtype=np.float32),
            sigma=np.full(self.n_actions, sigma_now, dtype=np.float32),
        )
        return True
