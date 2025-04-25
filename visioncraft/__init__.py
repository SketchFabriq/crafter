from gymnasium.envs.registration import register


register(
    id="WidowXTouch-v0",
    entry_point="visioncraft.envs.custom_widowx_env:WidowXTouchEnv",
    max_episode_steps=50,
)

register(
    id="WidowXTouchSparse-v0",
    entry_point="visioncraft.envs.custom_widowx_env:WidowXTouchEnv",
    max_episode_steps=50,
    kwargs={"reward_type": "sparse"},
)

register(
    id="WidowXGrasp-v0",
    entry_point="visioncraft.envs.custom_widowx_env:WidowXGraspEnv",
    max_episode_steps=50,
)

register(
    id="WidowXGraspSparse-v0",
    entry_point="visioncraft.envs.custom_widowx_env:WidowXGraspEnv",
    max_episode_steps=50,
    kwargs={"reward_type": "sparse"},
)

register(
    id="WidowXLift-v0",
    entry_point="visioncraft.envs.custom_widowx_env:WidowXLiftEnv",
    max_episode_steps=50,
)

register(
    id="WidowXPickPlace-v0",
    entry_point="visioncraft.envs.custom_widowx_env:WidowXPickPlaceEnv",
    max_episode_steps=100,
)
