"""
Yahtzee Gym: A Gymnasium environment for Yahtzee
"""

__version__ = "0.1.0"

# Register the environment when gymnasium is available
try:
    from gymnasium.envs.registration import register
    
    register(
        id='Yahtzee-v0',
        entry_point='yahtzee_gym.envs:YahtzeeEnv',
        max_episode_steps=13,  # 13 rounds in Yahtzee
    )
except ImportError:
    # Gymnasium not available during installation
    pass