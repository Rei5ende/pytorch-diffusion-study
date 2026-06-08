# Multimodal Action Distribution

> A distribution where multiple correct outputs exist for the same input
> One of the core limitations of Behavior Cloning — solved by Diffusion Policy

---

## Definition

**Unimodal:** One correct answer for a given input
```
"1 + 1 = ?"
→ Only one correct answer: 2
→ Single peak in the distribution
```

**Multimodal:** Multiple correct answers for the same input
```
"What should I have for dinner?"
→ Chicken is correct. Pizza is correct. Pasta is correct.
→ Multiple peaks in the distribution
```

---

## Why Multimodal Distributions Appear in Robotics

Different experts may behave differently in the same situation — and all of them can be correct.

```
Task: pick up a cup on a table

Expert A: approach from the left
Expert B: approach from the right
Expert C: reach from above

→ All three are valid actions
→ The demonstration dataset contains all three
→ The resulting distribution has multiple peaks
```

---

## Why BC Fails on Multimodal Data

BC minimizes MSE loss, which pushes the model toward the average of all correct outputs.

```
Expert A output: -2
Expert B output: +2

BC output: (-2 + 2) / 2 = 0  ← this action does not actually exist!
```

**Directly confirmed in experiment:**
```
Ground truth: peaks at -2 and +2
BC output:    always predicts ~0
→ Outputs a value that never appears in the training data
→ Complete failure
```

---

## Why Diffusion Solves It

Diffusion does not average — it samples.

```
Start from random noise
        ↓
Iteratively remove noise
        ↓
Naturally converges to one of the real peaks (-2 or +2)
→ Never outputs the average (0)
```

The randomness injected at each reverse step (z ~ N(0,1)) is what enables this:
- Different starting noise → converges to different peaks
- Both peaks are reachable → the full multimodal structure is preserved

**Directly confirmed in experiment:**
```
Ground truth: peaks at -2 and +2
BC:           always outputs ~0         ← wrong
Diffusion:    generates both peaks      ← correct
```

---

## Unimodal vs Multimodal vs BC vs Diffusion

| | Description | Example |
|--|-------------|---------|
| Unimodal | One correct answer | 1 + 1 = 2 |
| Multimodal | Multiple correct answers | Picking up a cup |
| BC on multimodal | Averages all correct answers → wrong | Outputs 0 instead of -2 or +2 |
| Diffusion on multimodal | Samples from the distribution → correct | Outputs -2 or +2 naturally |

---

## Connection to DiSPo

DiSPo (RIRO Lab, ICRA 2026) combines Diffusion Policy with Mamba (SSM):

```
Diffusion Policy
  → Solves multimodal action distribution
        ↓
DiSPo = Diffusion Policy + Mamba
  → Also solves temporal structure problem
  → User-controllable motion granularity via Step-Scale Factor
```

---

## References

- Diffusion Policy paper: https://arxiv.org/abs/2303.04137
- DiSPo review: ../../robotics-paper-study/dispo/paper_review.md
- DDPM math: ./ddpm_math.md
