# Deterministic vs Stochastic Models

> The core reason why BC fails on multimodal data is not the loss function —
> it is the deterministic nature of the model itself.

---

## The Core Problem

```
Deterministic model:
  same input → always same output

Stochastic model:
  same input → different outputs depending on randomness
```

Multimodal distributions require stochastic models.
A deterministic model, regardless of which loss function is used,
will always converge to a single output for a given input.

---

## Why Changing the Loss Function Doesn't Help

A common misconception:

> "BC uses MSELoss which averages outputs — just change the loss function!"

**The real issue:**

```
MSELoss:          pushes toward the mean → outputs 0 for {-2, +2}
CrossEntropyLoss: requires discrete classes → not applicable for continuous actions
Other losses:     still the same deterministic model
                  → same input always produces same output
                  → cannot represent multiple valid outputs
```

The loss function shapes *how* the model learns, but cannot change the fact that
a deterministic model produces a single output per input.

---

## Model Comparison

### 1. Behavior Cloning (BC) — Deterministic

```
Input x → [MLP] → Single output
```

```python
class BCModel(nn.Module):
    def forward(self, x):
        return self.net(x)  # always same output for same x
```

**Multimodal result:**
```
Training data: -2 and +2
Output:        0  ← average, never exists in training data
```

---

### 2. Gaussian Mixture Model (GMM) — Stochastic

Instead of predicting a single value, predict the parameters of a mixture of Gaussians.

```
Input x → [MLP] → (μ_1, σ_1, w_1), (μ_2, σ_2, w_2), ...
                       ↓
               Sample from mixture
                       ↓
               Output: -2 or +2
```

**Multimodal result:**
```
Can represent multiple peaks explicitly
But: number of modes must be fixed in advance
     hard to scale to high-dimensional action spaces
```

---

### 3. Normalizing Flow — Stochastic

Learn an invertible mapping between a simple distribution and the target distribution.

```
Simple distribution N(0,1)
        ↓
Invertible transformation f_θ
        ↓
Complex target distribution (multimodal)
```

**Multimodal result:**
```
Can represent complex distributions exactly
But: invertibility constraint limits model expressiveness
     computationally expensive for high dimensions
```

---

### 4. VAE (Variational Autoencoder) — Stochastic

Encode input into a latent distribution, then sample and decode.

```
Input x → Encoder → μ, σ
                      ↓
               z ~ N(μ, σ)   ← sampling introduces randomness
                      ↓
           Decoder → Output
```

```python
# Randomness comes from sampling z
z = mu + std * torch.randn_like(std)
```

**Multimodal result:**
```
Can generate diverse outputs via latent sampling
But: posterior collapse problem (model ignores z)
     blurry outputs in practice
```

---

### 5. GAN (Generative Adversarial Network) — Stochastic

Train a generator to fool a discriminator.

```
Random noise z ~ N(0,1)
        ↓
Generator G_θ
        ↓
Generated sample (looks like real data)
```

**Multimodal result:**
```
Can generate sharp, realistic samples
But: mode collapse (generator ignores some modes)
     unstable training
     hard to condition on input
```

---

### 6. Diffusion Model — Stochastic

Iteratively denoise from pure random noise.

```
Random noise x_T ~ N(0,1)
        ↓
Denoising step × T
        ↓
Generated sample
```

```python
# Randomness comes from starting noise AND each reverse step
x = torch.randn(num_samples)           # random start
for t in reversed(range(T)):
    x = reverse_step(model, x, t, ...) # randomness at each step
```

**Multimodal result:**
```
Naturally converges to one of the real peaks
Stable training (unlike GAN)
Highly expressive (unlike GMM)
No posterior collapse (unlike VAE)
```

---

## Summary Comparison

| Model | Stochastic? | Multimodal | Stability | Scalability |
|-------|-------------|------------|-----------|-------------|
| BC (MLP) | ❌ | ❌ Averages modes | ✅ Stable | ✅ Simple |
| GMM | ✅ | ⚠️ Fixed # modes | ✅ Stable | ❌ Hard to scale |
| Normalizing Flow | ✅ | ✅ Exact | ✅ Stable | ❌ Invertibility constraint |
| VAE | ✅ | ⚠️ Posterior collapse | ✅ Stable | ✅ Good |
| GAN | ✅ | ⚠️ Mode collapse | ❌ Unstable | ✅ Good |
| Diffusion | ✅ | ✅ Natural | ✅ Stable | ✅ Good |

---

## Why Diffusion Wins in Robot Learning

```
Robot action spaces are:
  high-dimensional     → rules out GMM, Normalizing Flow
  continuous           → rules out CrossEntropyLoss fix
  multimodal           → rules out BC

Diffusion:
  handles high dimensions      ✅
  handles continuous actions   ✅
  handles multimodal data      ✅
  training is stable           ✅
```

This is why Diffusion Policy (Chi et al. 2023) became the standard approach,
and why DiSPo (RIRO Lab) builds on top of it.

---

## Key Takeaway

> The problem is not MSELoss.
> The problem is that BC is deterministic.
> No matter what loss function you use,
> a deterministic model cannot represent a multimodal distribution.
> The solution is to inject randomness into the model itself.

---

## References

- Diffusion Policy: https://arxiv.org/abs/2303.04137
- DDPM: https://arxiv.org/abs/2006.11239
- VAE: https://arxiv.org/abs/1312.6114
- GAN: https://arxiv.org/abs/1406.2661
- multimodal.md: ./multimodal.md
- ddpm_math.md: ./ddpm_math.md
