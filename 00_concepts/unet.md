# U-Net Architecture

> The backbone architecture used in the original DDPM paper
> Key structure: encoder-decoder with skip connections

---

## Why U-Net for Diffusion?

The noise prediction model ε_θ(x_t, t) needs to:

```
1. Understand the global structure of x_t
   → "what kind of data is this overall?"

2. Preserve local details
   → "what are the fine-grained features?"

3. Condition on t
   → "how much noise was added at this step?"
```

U-Net satisfies all three requirements naturally.

---

## Basic Structure

```
Input x_t
    │
    ▼
┌─────────────┐
│  Encoder    │  Downsampling path
│  (Contracting)│
│  Conv → ↓  │  Extract features, reduce spatial size
│  Conv → ↓  │
│  Conv → ↓  │
└──────┬──────┘
       │
    Bottleneck
    (deepest features)
       │
┌──────┴──────┐
│  Decoder    │  Upsampling path
│  (Expanding)│
│  ↑ → Conv  │  Reconstruct spatial size
│  ↑ → Conv  │
│  ↑ → Conv  │
└─────────────┘
    │
    ▼
Output ε̂ (predicted noise, same size as input)
```

---

## Skip Connections — The Key Feature

Skip connections directly connect encoder layers to decoder layers at the same resolution.

```
Encoder layer 1 ──────────────────────┐
    ↓                                  │ skip connection
Encoder layer 2 ────────────────┐     │
    ↓                            │     │
Encoder layer 3 ──────────┐     │     │
    ↓                      │     │     │
  Bottleneck               │     │     │
    ↓                      │     │     │
Decoder layer 3 ←──────────┘     │     │  (concat feature maps)
    ↓                            │     │
Decoder layer 2 ←────────────────┘     │
    ↓                                  │
Decoder layer 1 ←──────────────────────┘
    ↓
  Output
```

**Why skip connections matter:**
```
Without skip connections:
  Deep features lose spatial information during downsampling
  → Decoder struggles to recover fine details

With skip connections:
  Encoder features are directly passed to the decoder
  → Local details are preserved throughout
  → Sharp, detailed output
```

---

## Time Conditioning — How t is Injected

The model needs to know which timestep t it is operating at.
DDPM uses sinusoidal positional embeddings (same as Transformers):

```python
# Sinusoidal time embedding
def time_embedding(t, dim):
    half = dim // 2
    freq = torch.exp(
        -math.log(10000) * torch.arange(half) / half
    )
    args = t[:, None] * freq[None]
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    return embedding
```

The time embedding is then injected into each residual block:

```
ResBlock:
  x → Conv → GroupNorm → SiLU
                              ↑
               time_emb → Linear → (added here)
  → Conv → GroupNorm → SiLU → output
```

---

## Residual Blocks inside U-Net

Each layer in the U-Net is a residual block (from ResNet):

```
Input x
    │
    ├──────────────────────┐  (skip connection within block)
    ▼                      │
  Conv → Norm → Act        │
    ↓                      │
  + time embedding         │
    ↓                      │
  Conv → Norm → Act        │
    ▼                      │
    + ◄─────────────────────┘
    ▼
  Output
```

**Why residual connections:**
```
Easier gradient flow during backpropagation
→ Enables training very deep networks
→ Model learns the residual (what to add/remove) rather than full transformation
```

---

## Attention Mechanism

At lower resolution layers, U-Net adds self-attention:

```
Low resolution features (global context)
    ↓
Self-Attention
    ↓
Each position attends to all other positions
→ Captures long-range dependencies
→ Helps model understand global structure
```

```
High resolution: Conv only (local features, efficiency)
Low resolution:  Conv + Attention (global features)
```

---

## U-Net vs Simple MLP (used in our experiment)

| | Simple MLP | U-Net |
|--|-----------|-------|
| Input | 1D scalar | 2D image (H×W×C) |
| Structure | Linear layers | Encoder-decoder |
| Skip connections | ❌ | ✅ |
| Spatial info | ❌ Not applicable | ✅ Preserved |
| Time conditioning | Embedding layer | Sinusoidal + ResBlock injection |
| Use case | 1D toy experiment | Real image/state diffusion |

In our 1D experiment, a simple MLP was sufficient.
For image-based Diffusion Policy (robot camera input), U-Net is necessary.

---

## Connection to Diffusion Policy

Diffusion Policy (Chi et al. 2023) uses two variants:

**CNN-based (U-Net style):**
```
Observation (image) → CNN encoder
Action sequence     → 1D U-Net (temporal)
→ Fast inference
→ Good for fixed-length action sequences
```

**Transformer-based:**
```
Observation tokens + noisy action tokens → Transformer
→ Better for long sequences
→ More flexible
```

**DiSPo (RIRO Lab):**
```
Replaces Transformer with Mamba (SSM)
→ Better temporal structure understanding
→ User-controllable motion granularity
```

---

## Key Takeaway

```
U-Net = Encoder + Decoder + Skip Connections

Skip connections solve the information bottleneck problem:
  local details lost during downsampling
  → directly passed to decoder via skip connections

Time conditioning injects t at every layer:
  model always knows "how noisy is the input right now"

This architecture is why DDPM produces sharp,
detailed outputs rather than blurry averages.
```

---

## References

- Original U-Net paper: https://arxiv.org/abs/1505.04597
- DDPM (uses U-Net): https://arxiv.org/abs/2006.11239
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- DiSPo review: ../../robotics-paper-study/dispo/paper_review.md
- ddpm_math.md: ./ddpm_math.md
