# Training Diagnostics & Visualizations Guide

This document explains the diagnostic metrics logged during training and how to interpret them for debugging training issues.

## Quick Health Checklist

✅ **Healthy Training Signs:**
- `train/critic_ev`: 0.5 - 0.9 (and trending upward)
- `train/entropy_roll`: > 0.5 (5 dice × ~0.1-0.2 per die)
- `train/entropy_score`: > 1.0 (categorical over 13 categories)
- `train/kl_roll` & `train/kl_score`: > 1e-3 (policy is moving)
- `train/avg_reward`: Steadily increasing
- `train/score_top1`: < 0.5 (not collapsed)
- `diagnostics/training_health`: > 0.5

🚨 **Warning Signs:**
- `train/critic_ev` < 0: Critic is worse than mean baseline
- `train/entropy_score` < 0.5: Policy collapsing to single action
- `train/kl_roll` < 1e-4: Policy frozen, not learning
- `train/score_top1` > 0.8: Deterministic, lost exploration
- `train/adv_roll_mean` >> `train/adv_score_mean`: Phase imbalance

---

## Metric Categories

### 1. Advantage & Return Statistics

**What they measure:** How well the value function estimates expected returns and how much variance exists in episode outcomes.

**Metrics:**
- `train/adv_mean`: Average advantage across all steps (should oscillate near 0)
- `train/adv_std`: Standard deviation of advantages (shows learning signal strength)
- `train/adv_roll_mean`: Average advantage for rolling actions
- `train/adv_score_mean`: Average advantage for scoring actions
- `train/ret_std`: Variability in episode returns

**What to look for:**
- If `adv_roll_mean` and `adv_score_mean` differ significantly (>10x), one phase might be dominating learning
- High `ret_std` early in training is normal; should decrease as policy improves
- `adv_std` too low (<0.1) means weak learning signal

**Suggested plots:**
```python
# WandB dashboard
Line plot: train/adv_roll_mean vs train/adv_score_mean
Line plot: train/adv_std, train/ret_std
```

---

### 2. Critic Quality (Value Function)

**What it measures:** How well your value function predicts actual returns.

**Metrics:**
- `train/critic_ev`: Explained variance = 1 - Var(G - V) / Var(G)
- `train/v_loss`: MSE loss between predictions and returns

**Interpretation:**
- `critic_ev = 1.0`: Perfect predictions
- `critic_ev = 0.0`: Predictions as good as constant baseline (mean return)
- `critic_ev < 0.0`: Predictions worse than mean (critic is harmful!)

**What to look for:**
- Should trend upward from ~0 to 0.5-0.9
- If stuck near 0 or negative: critic learning rate too high/low, or gradient issues
- If oscillating wildly: value function overfitting to recent batches

**Suggested plots:**
```python
Line plot: train/critic_ev, train/v_loss
Alert: if critic_ev < 0 for 5+ consecutive epochs
```

---

### 3. Entropy (Exploration)

**What it measures:** How random/deterministic your policy is. High entropy = exploring, low entropy = exploiting.

**Metrics:**
- `train/entropy_roll`: Entropy of rolling head (sum of 5 Bernoulli entropies)
  - Max = 5 × 0.693 ≈ 3.47 (uniform 50/50 for each die)
  - Min = 0 (deterministic)
- `train/entropy_score`: Entropy of scoring head (categorical over 13 categories)
  - Max = log(13) ≈ 2.56 (uniform distribution)
  - Min = 0 (deterministic)

**What to look for:**
- **Premature collapse:** Entropy drops to near 0 before policy has learned
  - Fix: Increase exploration (higher temperature, entropy bonus)
- **No exploitation:** Entropy stays maximal even late in training
  - Fix: Decrease exploration, policy is not learning
- **Healthy pattern:** Gradual decrease from high to moderate levels

**Suggested plots:**
```python
Line plot: train/entropy_roll, train/entropy_score
Horizontal reference line: entropy_score = 1.0 (healthy minimum)
```

---

### 4. Action Concentration (Collapse Detection)

**What it measures:** How much probability mass is concentrated in the top actions. Early warning for policy collapse.

**Metrics:**
- `train/score_top1`: Mean probability of most likely scoring action
- `train/score_top3`: Mean total probability of top 3 scoring actions
- `train/roll_mask_diversity`: Fraction of unique rolling patterns in batch

**Interpretation:**
- Random policy: `top1 ≈ 1/13 ≈ 0.077`, `top3 ≈ 3/13 ≈ 0.23`
- Deterministic policy: `top1 ≈ 1.0`, `top3 ≈ 1.0`

**What to look for:**
- **Red flag:** `score_top1 > 0.8` or `roll_mask_diversity < 0.1` = collapsed
- **Healthy:** `score_top1` gradually increases from ~0.3 to ~0.6
- **Warning:** Sudden spike in `score_top1` = collapse event

**Suggested plots:**
```python
Line plot: train/score_top1, train/score_top3
Line plot: train/roll_mask_diversity
Alert: if score_top1 > 0.85
```

---

### 5. KL Divergence (Policy Movement)

**What it measures:** How much your policy changes between consecutive training steps. Indicates if learning is happening.

**Metrics:**
- `train/kl_roll`: KL(current || previous) for rolling head
- `train/kl_score`: KL(current || previous) for scoring head

**Interpretation:**
- KL ≈ 0: Policy not moving (frozen)
- KL ≈ 1e-3 to 1e-1: Healthy gradual updates
- KL > 1.0: Policy changing too rapidly (might be unstable)

**What to look for:**
- **Plateauing:** KL < 1e-4 and reward not improving = policy stuck
  - Fix: Increase learning rate, check gradients
- **Instability:** KL > 1.0 with oscillating rewards = learning too fast
  - Fix: Decrease learning rate, gradient clipping
- **Healthy:** KL starts high (~1e-1), gradually decreases to ~1e-2

**Suggested plots:**
```python
Line plot (log scale): train/kl_roll, train/kl_score
Horizontal reference line: 1e-4 (minimum movement threshold)
```

---

### 6. Phase Balance

**What it measures:** What fraction of training steps are rolling vs scoring actions.

**Metrics:**
- `train/frac_roll_steps`: Fraction of rolling steps (should be ≈ 0.67 for 3-step episodes with 2 rolls + 1 score)

**What to look for:**
- Should be stable around 0.66-0.67
- If drifting, check episode generation logic

---

### 7. Composite Health Metrics

**What they measure:** Combined signals for quick health assessment.

**Metrics:**
- `diagnostics/training_health`: Weighted combination of critic_ev and entropies
  - Range: 0 (bad) to 1 (perfect)
  - Healthy: > 0.5
- `diagnostics/policy_movement`: Average KL divergence
- `diagnostics/critic_ev_epoch`: Epoch-averaged explained variance

**Suggested plots:**
```python
Line plot: diagnostics/training_health (main dashboard metric)
Line plot: diagnostics/policy_movement (log scale)
```

---

## Recommended WandB Dashboard Layout

### Panel 1: Overall Progress
```
- train/avg_reward (large, prominent)
- val/mean_total_score
- train/policy_loss
```

### Panel 2: Health Indicators
```
- diagnostics/training_health
- train/critic_ev
- diagnostics/policy_movement (log scale)
```

### Panel 3: Exploration
```
- train/entropy_roll
- train/entropy_score
- train/score_top1
- train/roll_mask_diversity
```

### Panel 4: Learning Dynamics
```
- train/kl_roll (log scale)
- train/kl_score (log scale)
- train/adv_std
- lr
```

### Panel 5: Phase Balance
```
- train/adv_roll_mean vs train/adv_score_mean
- train/frac_roll_steps
```

---

## Common Issues & Fixes

### Issue: Reward Plateaus

**Symptoms:**
- `train/avg_reward` flat
- `train/kl_*` < 1e-4
- `train/critic_ev` not improving

**Possible causes:**
1. Learning rate too low → Increase LR
2. Gradient vanishing → Check gradient norms, adjust architecture
3. Policy/critic balance off → Adjust loss coefficients (e.g., `0.05 * v_loss`)

---

### Issue: Policy Collapse

**Symptoms:**
- `train/entropy_score` < 0.5
- `train/score_top1` > 0.8
- `train/roll_mask_diversity` < 0.1

**Possible causes:**
1. Insufficient exploration → Add entropy bonus to loss
2. Learning rate too high → Decrease LR
3. Bad rewards dominating → Check reward shaping

---

### Issue: Unstable Training

**Symptoms:**
- Rewards oscillating wildly
- `train/kl_*` > 1.0
- `train/critic_ev` jumping around

**Possible causes:**
1. Learning rate too high → Decrease LR
2. Large gradient updates → Enable gradient clipping
3. Batch size too small → Increase batch size

---

### Issue: Critic Not Learning

**Symptoms:**
- `train/critic_ev` stuck at 0 or negative
- `train/v_loss` not decreasing

**Possible causes:**
1. Critic LR too low → Increase critic coefficient in loss
2. Critic architecture too weak → Add layers/units
3. Value targets too noisy → Increase gamma, use TD(λ)

---

## Advanced: Custom Alerts

Set up WandB alerts for automated monitoring:

```python
# In WandB dashboard
Alert if train/critic_ev < 0 for 10 consecutive steps
Alert if train/entropy_score < 0.5
Alert if train/kl_score < 1e-5 AND train/avg_reward not improving
Alert if diagnostics/training_health < 0.3
```
