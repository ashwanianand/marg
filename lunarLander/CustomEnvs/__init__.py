from gymnasium.envs.registration import register



register(
    id="CustomLunarLander-v0",
    entry_point="CustomEnvs.envs.lunar_lander:LunarLander",
    max_episode_steps=10000,
    reward_threshold=200,
)

register(
    id="CustomLunarLanderContinuous-v0",
    entry_point="CustomEnvs.envs.lunar_lander:LunarLander",
    kwargs={"continuous": True},
    max_episode_steps=4500,
    reward_threshold=200,
)