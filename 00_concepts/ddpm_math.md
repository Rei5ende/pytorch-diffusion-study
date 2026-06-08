# DDPM Core Math

> Key equations from DDPM (Ho et al. 2020), the foundation of Diffusion Policy
> Related: DiSPo (ICRA 2026), Diffusion Policy (RSS 2023)

---

## Symbol Reference

| Symbol | Meaning |
|--------|---------|
| x_0 | Original data |
| x_t | Data with noise added at step t |
| ε | Added noise (sampled from standard normal distribution) |
| β_t | Amount of noise to add at step t (noise schedule) |
| α_t | 1 - β_t (signal retention rate at step t) |
| α̅_t | α_1 × α_2 × ... × α_t (cumulative product) |
| T | Total number of noise addition steps |

---

## 1. Noise Schedule

```
β_t    = linspace(β_start, β_end, T)   noise amount per step
α_t    = 1 - β_t                        signal retention per step
α̅_t   = α_1 × α_2 × ... × α_t         cumulative product (cumprod)
```

**Intuition:**
```
Small β_t → add a little noise at that step
Large β_t → add a lot of noise at that step

Typically β starts small and increases over time
→ Slowly destroy structure at first, then rapidly at the end
```

**Code:**
```python
betas      = torch.linspace(0.01, 0.5, T)
alphas     = 1 - betas
alphas_bar = torch.cumprod(alphas, dim=0)
```

**Example values (T=10):**
```
t=1:  α̅ = 0.990  → 99.0% signal remaining
t=5:  α̅ = 0.521  → 52.1% signal remaining
t=10: α̅ = 0.042  →  4.2% signal remaining
```

---

## 2. Forward Process — Direct Sampling at Any t

```
x_t = sqrt(α̅_t) * x_0 + sqrt(1 - α̅_t) * ε
```

**What each term does:**

```
sqrt(α̅_t) * x_0       signal attenuation (scale down original)
sqrt(1 - α̅_t) * ε     noise injection
ε ~ N(0, 1)            noise sampled from standard normal
```

**Why this formulation is powerful:**
```
Naive approach: x_0 → x_1 → x_2 → ... → x_t  (t computations)
This formula:   x_0 → x_t directly              (1 computation)

→ During training, sample any random t and compute x_t immediately
→ Much faster training
```

**Effect of α̅_t:**
```
α̅_t = 0.99  →  x_t ≈ 0.995 * x_0 + 0.100 * ε  (mostly original)
α̅_t = 0.50  →  x_t ≈ 0.707 * x_0 + 0.707 * ε  (signal/noise 50-50)
α̅_t = 0.04  →  x_t ≈ 0.200 * x_0 + 0.980 * ε  (almost pure noise)
```

**Code:**
```python
def add_noise(x0, t, alphas_bar):
    sqrt_alpha_bar = torch.sqrt(alphas_bar[t])
    sqrt_one_minus = torch.sqrt(1 - alphas_bar[t])
    epsilon = torch.randn_like(x0)
    x_t = sqrt_alpha_bar * x0 + sqrt_one_minus * epsilon
    return x_t, epsilon
```

---

## 3. Training Objective — Noise Prediction

```
L = E[ || ε - ε_θ(x_t, t) ||² ]

ε      actual noise that was added
ε_θ    noise predicted by the model (parameters θ)
```

**Why predict noise instead of x_0 directly?**

Rearranging the forward process equation:

```
x_t = sqrt(α̅_t) * x_0 + sqrt(1 - α̅_t) * ε
        ↓
x_0 = (x_t - sqrt(1 - α̅_t) * ε) / sqrt(α̅_t)
```

If we know ε, we can recover x_0.
→ Predicting noise is equivalent to recovering the original data.

**Why MSELoss:**
```
Minimize the difference between predicted and actual noise
Continuous value prediction → MSELoss (not a classification problem)
```

---

## 4. Reverse Process — Denoising (Generation)

```
x_{t-1} = (1/sqrt(α_t)) * (x_t - (β_t / sqrt(1 - α̅_t)) * ε_θ(x_t, t))
           + sqrt(β_t) * z

z ~ N(0, 1)
```

**What each term does:**

```
1/sqrt(α_t)                      restore signal scale
β_t / sqrt(1 - α̅_t) * ε_θ      remove predicted noise
sqrt(β_t) * z                    inject a small amount of randomness
```

**Why add randomness (z)?**
```
Without z (fully deterministic):
→ Always produces the same output from the same noise
→ No diversity in generated samples

With z:
→ Different outputs from the same starting noise
→ For multimodal distributions, naturally converges to one of the peaks
→ This is why Diffusion solves the multimodal problem that BC cannot
```

**Code:**
```python
def reverse_step(model, x_t, t, betas, alphas, alphas_bar):
    with torch.no_grad():
        t_tensor   = torch.full((x_t.shape[0],), t, dtype=torch.long)
        pred_noise = model(x_t, t_tensor)

    coef = betas[t] / torch.sqrt(1 - alphas_bar[t])
    mean = (1 / torch.sqrt(alphas[t])) * (x_t - coef * pred_noise)

    if t == 0:
        return mean             # no noise added at final step
    else:
        z = torch.randn_like(x_t)
        return mean + torch.sqrt(betas[t]) * z
```

---

## 5. Full Algorithm Summary

```
Training:
  1. Sample x_0 from dataset
  2. Sample random t ~ Uniform(0, T)
  3. Sample noise ε ~ N(0, 1)
  4. Compute x_t = sqrt(α̅_t) * x_0 + sqrt(1-α̅_t) * ε
  5. Predict ε_θ(x_t, t) with model
  6. Compute Loss = MSE(ε, ε_θ) and backpropagate

Sampling (Generation):
  1. Start from x_T ~ N(0, 1)  (pure random noise)
  2. For t = T, T-1, ..., 1:
       x_{t-1} = reverse_step(x_t, t)
  3. Return x_0  (generated sample)
```

---

## 6. Comparison with Behavior Cloning

| | Behavior Cloning | Diffusion |
|---|---|---|
| Training target | Predict action directly from state | Predict added noise ε |
| Loss function | CrossEntropyLoss | MSELoss |
| Multimodal handling | Averages over modes (fails) | Naturally converges to one mode (succeeds) |
| Generation | Single forward pass | T iterative denoising steps |

---

## References

- DDPM paper: https://arxiv.org/abs/2006.11239
- Diffusion Policy paper: https://arxiv.org/abs/2303.04137
- DiSPo review: ../../robotics-paper-study/dispo/paper_review.md
