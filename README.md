# PyTorch Diffusion Policy Study

A study repo exploring Diffusion Policy as a solution to the limitations of Behavior Cloning discovered in [pytorch-bc-study](https://github.com/Rei5ende/pytorch-bc-study).

---

## Background

In `pytorch-bc-study`, two key limitations of Behavior Cloning were identified:

```
1. Distribution Shift
   BC fails when it encounters states outside its training distribution
   → The agent collapses rapidly when the pole angle exceeds ~5 degrees

2. Multimodal Action Distribution
   BC averages over multiple valid actions → ambiguous behavior
   → Diffusion Policy solves this by sampling from a learned distribution
```

This repo explores how Diffusion Policy addresses these issues.

---

## Study Order

| Stage | Topic | Status |
|-------|-------|--------|
| 00 | Concepts | 🔄 Ongoing |
| 01 | Diffusion Basics (DDPM) | ⬜ Upcoming |
| 02 | Diffusion Policy | ⬜ Upcoming |
| 03 | BC vs Diffusion Policy | ⬜ Upcoming |

---

## 00_concepts

Concepts and questions that come up during each stage.  
Added freely as curiosity arises — not part of a fixed curriculum.

---

## Connection to RIRO Lab Research

```
Behavior Cloning (pytorch-bc-study)
  → Expert demonstration → direct action imitation
        ↓  (limitation: Distribution Shift, Multimodal)
Diffusion Policy (this repo)
  → Noise → iterative denoising → action generation
  → Naturally handles multimodal action distributions
        ↓
DiSPo (RIRO Lab, ICRA 2026)
  → Diffusion Policy + Mamba (SSM)
  → Coarse-to-fine action control via Step-Scale Factor
        ↓
Inverse Constraint Learning (TCL, ILCL)
  → Learning constraints from demonstrations
```

---

## References

- RIRO Lab: https://rirolab.kaist.ac.kr
- Paper reviews: https://github.com/Rei5ende/riro-paper-study
- BC study: https://github.com/Rei5ende/pytorch-bc-study
- Diffusion Policy (Chi et al., 2023): https://arxiv.org/abs/2303.04137
- DDPM (Ho et al., 2020): https://arxiv.org/abs/2006.11239
