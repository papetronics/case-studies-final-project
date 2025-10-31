"""
Yahtzee Gym: A Gymnasium environment for Yahtzee
"""

from gymnasium.envs.registration import register

# Register the environment
register(
    id='Yahtzee-v0',
    entry_point='yahtzee_gym.envs:YahtzeeEnv',
    max_episode_steps=13,  # 13 rounds in Yahtzee
)

__version__ = "0.1.0"