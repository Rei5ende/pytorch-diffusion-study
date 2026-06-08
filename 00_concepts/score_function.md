# Score Function & Score Matching

> The mathematical foundation behind what DDPM is actually learning
> Connects DDPM to score-based generative models

---

## What is the Score Function?

The score function is the gradient of the log probability density with respect to the data:

```
s(x) = ∇_x log p(x)
```

**Intuition:**
```
p(x)    = probability density at point x
log p(x) = log probability
∇_x     = gradient with respect to x (which direction increases p(x))

Score function = "which direction should x move to become more likely?"
```

**Visualization:**
```
Low density region          High density region
      ↓                           ↓
  p(x) small               p(x) large
  score points →            score points inward
  toward high density       toward peak
```

---

## Connection to DDPM

At first glance, DDPM predicts noise ε — not the score function.
But they are mathematically equivalent.

**Forward process at step t:**
```
x_t = sqrt(α̅_t) * x_0 + sqrt(1 - α̅_t) * ε
```

**Score of the noisy distribution:**
```
∇_xt log p(x_t) = -ε / sqrt(1 - α̅_t)
```

Therefore:
```
Predicting ε  ≡  Predicting the score (up to a scaling factor)

ε_θ(x_t, t) ≈ -sqrt(1 - α̅_t) * ∇_xt log p(x_t)
```

**What this means:**
```
DDPM noise prediction
= learning the score function of the noisy distribution
= learning "which direction to move x_t to make it more like real data"
```

---

## Score Matching

Score matching is a training objective that directly minimizes
the difference between the model's score and the true score:

```
L_SM = E[ || s_θ(x) - ∇_x log p(x) ||² ]
```

**Problem:** we don't know the true score ∇_x log p(x).

**Solution — Denoising Score Matching (DSM):**
```
Instead of matching the score of p(x),
match the score of the noisy distribution p(x_t | x_0):

∇_xt log p(x_t | x_0) = -(x_t - sqrt(α̅_t) * x_0) / (1 - α̅_t)
                       = -ε / sqrt(1 - α̅_t)
```

This is exactly what DDPM learns — DDPM is denoising score matching.

---

## Score-Based Generative Models vs DDPM

Two seemingly different frameworks that turn out to be equivalent:

| | Score-Based Models | DDPM |
|--|-------------------|------|
| Training | Learn score ∇_x log p(x_t) | Predict noise ε |
| Sampling | Langevin dynamics | Reverse diffusion |
| Key paper | Song et al. 2020 | Ho et al. 2020 |
| Equivalence | ✅ Mathematically identical | ✅ Mathematically identical |

Song et al. (2021) unified both frameworks under **Stochastic Differential Equations (SDEs)**,
showing they are special cases of the same general framework.

---

## Why the Score Function Matters for Robot Learning

```
Score function = "direction toward more likely data"

In robot learning:
  More likely data = demonstrations that look like expert behavior
  Score points toward: actions an expert would take

Diffusion Policy learns this score implicitly
→ Each denoising step moves x_t toward expert-like actions
→ Final x_0 is a plausible expert action
```

---

## Langevin Dynamics — Sampling with the Score

Once we have the score, we can generate samples using Langevin dynamics:

```
x_{t+1} = x_t + α * ∇_x log p(x_t) + sqrt(2α) * z
              ↑                              ↑
         move toward                    random exploration
         high density
```

This is essentially what the DDPM reverse process does at each step.

---

## Key Takeaway

```
DDPM noise prediction (ε)
        ≡
Score function estimation (∇_x log p(x_t))
        ≡
Denoising Score Matching

All three are the same thing expressed differently.
Understanding the score function reveals WHY diffusion works:
it learns to point toward regions of high data density,
which in robot learning means pointing toward expert behavior.
```

---

## References

- DDPM: https://arxiv.org/abs/2006.11239
- Score-Based Models (Song et al. 2020): https://arxiv.org/abs/2006.09011
- Unified SDE Framework (Song et al. 2021): https://arxiv.org/abs/2011.13456
- ddpm_math.md: ./ddpm_math.md
- deterministic_vs_stochastic.md: ./deterministic_vs_stochastic.md
