from gymnasium.envs.registration import register
from importlib import import_module

ROBOTS = {
    "Locobot": "visioncraft.envs.custom_env_locobot",
    "WidowX": "visioncraft.envs.custom_env_widowx",
}

TASKS = {
    "Touch": ("TouchEnv", 50),
    "Grasp": ("GraspEnv", 50),
    "Lift": ("LiftEnv", 50),
    "PickPlace": ("PickPlaceEnv", 100),
}

REWARD_VARIANTS = {
    "dense": {"reward_type": "dense"},
    "sparse": {"reward_type": "sparse"},
}

def _class_exists(module_path: str, class_name: str) -> bool:
    """Return True if *class_name* exists inside *module_path*."""
    try:
        mod = import_module(module_path)
        return hasattr(mod, class_name)
    except ModuleNotFoundError:
        return False


def _register_all() -> None:
    """Register every (robot, task, reward) combination with Gymnasium."""
    for robot, module in ROBOTS.items():
        for task, (cls_suffix, max_steps) in TASKS.items():
            env_class = f"{robot}{cls_suffix}"

            if not _class_exists(module, env_class):
                raise ImportError(
                    f"{env_class} not found in {module}. Did you spell it correctly?"
                )

            for rew_name, kwargs in REWARD_VARIANTS.items():
                is_sparse = rew_name == "sparse"
                id_suffix = "Sparse" if is_sparse else ""
                gym_id = f"{robot}{task}{id_suffix}-v0"

                register(
                    id=gym_id,
                    entry_point=f"{module}:{env_class}",
                    max_episode_steps=max_steps,
                    kwargs=(kwargs or None),
                )

_register_all()
