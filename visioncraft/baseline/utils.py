import numpy as np

def exponential_schedule(initial_value: float, decay_rate: float = 0.9):
    """
    Exponential decay schedule.
    """
    def scheduler(progress_remaining: float):
        return initial_value * (decay_rate ** (1 - progress_remaining))
    return scheduler


def shaped_dist(x: np.ndarray, y: np.ndarray, scale: float = 0.30) -> np.ndarray:
    """Normalised Euclidean distance in [0, 1]."""
    x, y = np.asarray(x), np.asarray(y)
    d = np.linalg.norm(x - y, axis=-1) / scale
    return np.clip(d, 0.0, 1.0).astype(np.float32)

