# Diffusion vs Diffusion Policy

> Diffusion is a general data generation technique.
> Diffusion Policy is Diffusion specialized for robot action generation.

---

## One-Line Summary

```
Diffusion        = generate plausible data (images, audio, etc.)
Diffusion Policy = generate plausible actions conditioned on robot state
```

---

## Diffusion (Original)

```
Input:  random noise
Output: image, audio, text, etc.

Condition: none, or text prompt
           "generate a photo of a cat"
```

Core goal: **generate realistic data from noise**

Key papers:
- DDPM (Ho et al. 2020)
- Stable Diffusion
- DALL-E 2

---

## Diffusion Policy

```
Input:  random noise + robot observation (state)
Output: robot action

Condition: current state is required
           "given this observation, what action should I take?"
```

Core goal: **generate the correct action for the current state**

Key paper: Diffusion Policy (Chi et al., RSS 2023)

---

## The Key Difference: Conditional Generation

```
Diffusion:
  noise → data
  (unconditional or text-conditioned)

Diffusion Policy:
  noise + state → action
  (state-conditioned action generation)
```

In terms of the noise prediction model:

```
Diffusion:        ε_θ(x_t, t)
Diffusion Policy: ε_θ(x_t, t, obs)   ← obs (observation) added
```

The model now takes the robot's current observation as an additional input,
so the generated action is always conditioned on what the robot currently sees.

---

## How Conditioning Works in Practice

```python
class DiffusionPolicyModel(nn.Module):
    def forward(self, x_t, t, obs):
        # obs: robot observation (e.g., CartPole state: 4 values)
        # x_t: noisy action at step t
        # t:   current timestep

        t_emb   = self.time_embed(t)    # timestep embedding
        obs_emb = self.obs_encoder(obs) # observation embedding

        # combine all three
        inp = torch.cat([x_t, t_emb, obs_emb], dim=-1)
        return self.net(inp)            # predict noise
```

---

## Analogy

```
Diffusion (unconditional):
  "Draw me a picture" → generates any picture

Diffusion (text-conditioned):
  "Draw me a cat" → generates a cat picture

Diffusion Policy (state-conditioned):
  "Given this CartPole state, what action should be taken?"
  → generates the appropriate action (-1 or +1)
  → different states produce different actions
```

---

## Why Conditioning on State Matters

Without conditioning:
```
noise → action
→ The generated action ignores the current situation
→ Useless for robot control
```

With state conditioning:
```
noise + state → action
→ The action is always appropriate for the current state
→ Different states → different action distributions
→ This is what makes it a policy (not just a generator)
```

---

## Comparison Table

| | Diffusion | Diffusion Policy |
|--|-----------|-----------------|
| Output | Image, audio, text | Robot action |
| Condition | None or text prompt | Robot observation (state) |
| Goal | Generate realistic data | Generate correct action for state |
| Training data | Images, audio files | Expert demonstrations (state, action) |
| Key paper | DDPM (Ho et al. 2020) | Diffusion Policy (Chi et al. 2023) |
| RIRO Lab extension | — | DiSPo (Diffusion Policy + Mamba) |

---

## Connection to DiSPo

```
Diffusion Policy
  → Solves multimodal action distribution
  → State-conditioned action generation
        ↓
DiSPo (RIRO Lab, ICRA 2026)
  = Diffusion Policy + Mamba (SSM)
  → Also handles temporal structure
  → Step-Scale Factor for motion granularity control
  → Coarse demonstrations → fine-grained actions
```

---

## References

- DDPM: https://arxiv.org/abs/2006.11239
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- DiSPo review: ../../robotics-paper-study/dispo/paper_review.md
- ddpm_math.md: ./ddpm_math.md
- deterministic_vs_stochastic.md: ./deterministic_vs_stochastic.md
