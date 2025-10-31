# Yahtzee Gym

A Gymnasium environment for the classic dice game Yahtzee.

## Installation

```bash
pip install -e .
```

## Usage

```python
import gymnasium as gym
import yahtzee_gym

env = gym.make('Yahtzee-v0')
observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # Random action
    observation, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        observation, info = env.reset()

env.close()
```

## Development

To install in development mode with dev dependencies:

```bash
pip install -e ".[dev]"
```

## License

MIT License