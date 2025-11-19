# 🎲 Learning to Play Yahtzee with Reinforcement Learning

> **Final Project** | *Case Studies in Machine Learning* | UT Austin MS in Artificial Intelligence


## 📄 Read the Paper
**The complete research paper with full technical details, experiments, and results:**
> ### **[📖 View Paper (PDF)](./paper/csml_paper.pdf)**


## 🎯 Project Overview

This project explores how well reinforcement learning agents can master Yahtzee through **pure RL training** — without hand-crafted strategies or game-specific heuristics. We investigate two key approaches:

### 🔹 Single-Turn Learning
Training agents to maximize score on **individual turns** (roll → roll → roll → score), then evaluating how well this generalizes to full games.

### 🔹 Full-Game Learning  
Training agents to play **complete 13-round games**, learning long-term strategy and category selection across an entire game.

---

## 🚀 Quick Start

### Run Single-Turn RL Training
```bash
./single_turn_rl.sh
```

Trains an agent on isolated single-turn gameplay. The agent learns optimal dice-keeping and scoring strategies for maximizing points in a single turn.

### Run Full-Game RL Training
```bash
./full_game_rl.sh
```

Trains an agent to play complete Yahtzee games. The agent must learn strategic category selection, timing, and trade-offs across all 13 turns.

---

## 📊 Performance Overview

| Approach | Average Score | Key Insight |
|----------|---------------|-------------|
| **Random Policy** | ~49 | Baseline performance |
| **One-turn Expectimax** | ~110 | Greedy single-turn expectimax |
| **Single-Turn REINFORCE** | ~200 | Strong tactical decisions |
| **Full-Game REINFORCE** | ~180-200 | Strategic category planning |
| **One-turn Expectimax (Optimal)** | ~254 | Theoretical upper bound |

---

## 🏗️ What's in the repo?

- ✅ Custom Yahtzee environment following OpenAI Gym interface
- ✅ REINFORCE policy gradient implementation with PyTorch Lightning
- ✅ Single-turn and full-game training pipelines
- ✅ Expectimax baseline for performance comparison
- ✅ Comprehensive evaluation framework
- ✅ W&B integration for experiment tracking

---

## 📚 Repository Structure

```
├── paper/              # LaTeX source and compiled PDF
├── src/                # All source code
│   ├── environments/   # Yahtzee gym environment
│   ├── yahtzee_agent/  # RL agent implementations
│   └── utilities/      # Helper functions and baselines
├── checkpoints/        # Trained model checkpoints
└── logs/              # Training logs and metrics
```

---

## 🎓 About

This work was completed as the final project for the **Case Studies in Machine Learning** course in the University of Texas at Austin's Master of Science in Artificial Intelligence program.

**For full details on methodology, architecture, experiments, and analysis, please see the [paper](./paper/csml_paper.pdf).**