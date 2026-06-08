# 00. Concepts

Concepts and questions that came up while working through `pytorch-diffusion-study`.
Added freely as curiosity arose — not part of a fixed curriculum.

---

## File List

| File | Topic | Related Stage |
|------|-------|---------------|
| [ddpm_math.md](./ddpm_math.md) | Core DDPM equations: noise schedule, forward process, training objective, reverse process | 01 |
| [multimodal.md](./multimodal.md) | What multimodal distributions are and why BC fails on them | 01 |
| [deterministic_vs_stochastic.md](./deterministic_vs_stochastic.md) | Why changing the loss function alone doesn't fix BC — and how 6 different model types handle multimodal data | 01 |
| [score_function.md](./score_function.md) | The mathematical connection between DDPM noise prediction and score matching | 01 |
| [unet.md](./unet.md) | U-Net architecture: encoder-decoder, skip connections, time conditioning | 01, 02 |

---

## How These Concepts Connect

```
DDPM Math (ddpm_math.md)
  → Forward process, reverse process, training objective
        ↓
Score Function (score_function.md)
  → Reveals WHY noise prediction works
  → DDPM = Denoising Score Matching
        ↓
Multimodal (multimodal.md)
  → The core problem Diffusion solves
        ↓
Deterministic vs Stochastic (deterministic_vs_stochastic.md)
  → WHY BC fails — not the loss function, but the model structure
  → Comparison of 6 generative model types
        ↓
U-Net (unet.md)
  → The architecture that makes it work for images and robot states
  → Skip connections, time conditioning, attention
```

---

## Questions That Led to These Files

```
"What is DDPM actually learning mathematically?"
→ score_function.md

"Why can't we just change the loss function in BC?"
→ deterministic_vs_stochastic.md

"What architecture does DDPM use for images?"
→ unet.md

"What exactly is a multimodal distribution?"
→ multimodal.md
```

---

## To Be Added

New concepts will be added as questions arise in later stages.
