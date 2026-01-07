# FluxSpace: Explained with a 32×32×3 RGB Image Example

## 🖼️ Scenario

We have a **32×32 pixel RGB image**:
- Woman's face (eye region in the center)
- Edit: "Add sunglasses"

```
Image Structure (32×32×3):
┌──────────────────────────────────┐
│  Background (blue sky)           │  Row 0-7
├──────────────────────────────────┤
│  Hair (brown)                    │  Row 8-11
├──────────────────────────────────┤
│  Forehead (skin tone)            │  Row 12-14
├──────────────────────────────────┤
│  EYES (target region)            │  Row 15-17
├──────────────────────────────────┤
│  Nose, Mouth (skin tone)         │  Row 18-24
├──────────────────────────────────┤
│  Neck, Shoulders                 │  Row 25-31
└──────────────────────────────────┘
```

---

## 📊 Step 1: Image → Latent Space

Diffusion models don't work directly on pixels. First, the image is converted to latent space using the **VAE encoder**:

```python
# Original image: 32×32×3 (H×W×C)
image = torch.randn(32, 32, 3)  # RGB values [0, 255] or [-1, 1]

# VAE Encoder: 8× downsampling
# 32/8 = 4, so latent size: 4×4×4 (or 4×4×16 for Flux)
latent = vae.encode(image)  # Shape: (4, 4, 4)
```

### Latent Grid Structure (4×4):

```
Latent patches (each represents an 8×8 pixel region):

     Col 0    Col 1    Col 2    Col 3
    ┌────────┬────────┬────────┬────────┐
Row 0│ Bg     │ Bg     │ Bg     │ Bg     │  ← Sky
    ├────────┼────────┼────────┼────────┤
Row 1│ Hair   │ Forehead│Forehead│ Hair   │  ← Hair + Forehead
    ├────────┼────────┼────────┼────────┤
Row 2│ Cheek  │LEFT EYE│RIGHT EYE│ Cheek  │  ← EYE REGION ⭐
    ├────────┼────────┼────────┼────────┤
Row 3│ Neck   │ Mouth  │ Mouth  │ Neck   │  ← Lower face
    └────────┴────────┴────────┴────────┘
```

---

## 📊 Step 2: Latent → Attention Tokens

The Transformer processes the latent as a **token sequence**:

```python
# Latent: 4×4×4 → Flatten → 16 tokens × 4 channels
# (In reality, Flux uses 3072 dim, we're simplifying here)

tokens = latent.flatten(0, 1)  # Shape: (16, 4)

# Meaning of each token:
token_map = {
    0: "bg_top_left",      1: "bg_top_mid",       2: "bg_top_mid",       3: "bg_top_right",
    4: "hair_left",        5: "forehead_left",    6: "forehead_right",   7: "hair_right",
    8: "cheek_left",       9: "LEFT_EYE ⭐",      10: "RIGHT_EYE ⭐",    11: "cheek_right",
    12: "neck_left",       13: "mouth_left",      14: "mouth_right",     15: "neck_right"
}
```

---

## 📐 Step 3: Three Attention Passes

FluxSpace computes three parallel attention passes:

### A) Base Attention (Original prompt: "a woman")

```python
# Text embedding: "a woman" → 77 tokens × 4 dim (simplified)
text_base = encode("a woman")  # Shape: (77, 4)

# Attention: image tokens × text tokens
# Q: image (16, 4), K: text (77, 4), V: text (77, 4)
attn_base = attention(Q=tokens, K=text_base, V=text_base)
# Output shape: (16, 4)
```

**Base attention output (16 tokens × 4 channels):**

```python
attn_base = [
    # Ch0   Ch1   Ch2   Ch3    ← Channels (feature dimensions)
    [0.12, 0.08, 0.15, 0.10],  # Token 0: bg_top_left
    [0.11, 0.09, 0.14, 0.11],  # Token 1: bg_mid
    [0.11, 0.09, 0.14, 0.11],  # Token 2: bg_mid
    [0.12, 0.08, 0.15, 0.10],  # Token 3: bg_top_right
    [0.25, 0.45, 0.12, 0.08],  # Token 4: hair_left (brown)
    [0.72, 0.18, 0.05, 0.03],  # Token 5: forehead (skin tone)
    [0.71, 0.19, 0.06, 0.04],  # Token 6: forehead
    [0.24, 0.46, 0.11, 0.09],  # Token 7: hair_right
    [0.70, 0.20, 0.08, 0.05],  # Token 8: cheek_left
    [0.35, 0.15, 0.42, 0.28],  # Token 9: LEFT_EYE ⭐
    [0.34, 0.16, 0.43, 0.27],  # Token 10: RIGHT_EYE ⭐
    [0.69, 0.21, 0.07, 0.06],  # Token 11: cheek_right
    [0.68, 0.22, 0.08, 0.07],  # Token 12: neck_left
    [0.75, 0.15, 0.06, 0.04],  # Token 13: mouth
    [0.74, 0.16, 0.07, 0.05],  # Token 14: mouth
    [0.67, 0.23, 0.09, 0.08],  # Token 15: neck_right
]
```

### B) Edit Attention (Edit prompt: "sunglasses")

```python
text_edit = encode("sunglasses")  # Shape: (77, 4)
attn_edit = attention(Q=tokens, K=text_edit, V=text_edit)
```

**Edit attention output:**

```python
attn_edit = [
    # Ch0   Ch1   Ch2   Ch3
    [0.10, 0.07, 0.12, 0.08],  # Token 0: bg (slightly affected)
    [0.09, 0.08, 0.11, 0.09],  # Token 1
    [0.09, 0.08, 0.11, 0.09],  # Token 2
    [0.10, 0.07, 0.12, 0.08],  # Token 3
    [0.22, 0.42, 0.10, 0.07],  # Token 4: hair (slightly affected)
    [0.68, 0.17, 0.08, 0.05],  # Token 5: forehead (slightly affected)
    [0.67, 0.18, 0.09, 0.06],  # Token 6
    [0.21, 0.43, 0.09, 0.08],  # Token 7: hair
    [0.65, 0.19, 0.12, 0.08],  # Token 8: cheek
    [0.15, 0.10, 0.85, 0.72],  # Token 9: LEFT_EYE ⭐⭐ (STRONG!)
    [0.14, 0.11, 0.86, 0.71],  # Token 10: RIGHT_EYE ⭐⭐ (STRONG!)
    [0.64, 0.20, 0.11, 0.09],  # Token 11: cheek
    [0.63, 0.21, 0.10, 0.08],  # Token 12: neck
    [0.70, 0.14, 0.08, 0.06],  # Token 13: mouth
    [0.69, 0.15, 0.09, 0.07],  # Token 14
    [0.62, 0.22, 0.11, 0.09],  # Token 15: neck
]
```

**Note:** Tokens 9 and 10 (eye regions) have very high values in Ch2 and Ch3!

### C) Null Attention (Empty prompt: "")

```python
text_null = encode("")  # Shape: (77, 4) - only padding
attn_null = attention(Q=tokens, K=text_null, V=text_null)
```

**Null attention output (image prior - preserving image direction):**

```python
attn_null = [
    # Ch0   Ch1   Ch2   Ch3
    [0.11, 0.08, 0.14, 0.09],  # Token 0
    [0.10, 0.09, 0.13, 0.10],  # Token 1
    [0.10, 0.09, 0.13, 0.10],  # Token 2
    [0.11, 0.08, 0.14, 0.09],  # Token 3
    [0.24, 0.44, 0.11, 0.08],  # Token 4: hair
    [0.71, 0.18, 0.06, 0.04],  # Token 5: forehead
    [0.70, 0.19, 0.07, 0.05],  # Token 6
    [0.23, 0.45, 0.10, 0.09],  # Token 7
    [0.69, 0.20, 0.09, 0.06],  # Token 8: cheek
    [0.34, 0.15, 0.41, 0.27],  # Token 9: LEFT_EYE
    [0.33, 0.16, 0.42, 0.26],  # Token 10: RIGHT_EYE
    [0.68, 0.21, 0.08, 0.07],  # Token 11
    [0.67, 0.22, 0.09, 0.08],  # Token 12
    [0.74, 0.15, 0.07, 0.05],  # Token 13
    [0.73, 0.16, 0.08, 0.06],  # Token 14
    [0.66, 0.23, 0.10, 0.09],  # Token 15
]
```

---

## 📐 Step 4: Orthogonal Projection (Detailed for Token 9 - Left Eye)

### Calculation for Token 9 (Left Eye):

```python
# Vectors (4-dimensional)
v_edit = [0.15, 0.10, 0.85, 0.72]  # Edit attention (sunglasses)
v_null = [0.34, 0.15, 0.41, 0.27]  # Null attention (image prior)

# Step 1: Dot product
dot_product = (0.15 × 0.34) + (0.10 × 0.15) + (0.85 × 0.41) + (0.72 × 0.27)
            = 0.051 + 0.015 + 0.349 + 0.194
            = 0.609

# Step 2: Squared norm of null vector
norm_sq = 0.34² + 0.15² + 0.41² + 0.27²
        = 0.116 + 0.023 + 0.168 + 0.073
        = 0.380

# Step 3: Projection coefficient
coef = dot_product / norm_sq = 0.609 / 0.380 = 1.603

# Step 4: Projection vector
projection = coef × v_null
           = 1.603 × [0.34, 0.15, 0.41, 0.27]
           = [0.545, 0.240, 0.657, 0.433]

# Step 5: Orthogonal direction (PURE edit direction)
v_ortho = v_edit - projection
        = [0.15, 0.10, 0.85, 0.72] - [0.545, 0.240, 0.657, 0.433]
        = [-0.395, -0.140, 0.193, 0.287]
```

### Result Interpretation (Token 9):

```
v_edit  = [0.15,  0.10,  0.85,  0.72 ]  ← "sunglasses" effect
v_null  = [0.34,  0.15,  0.41,  0.27 ]  ← Image prior (preserving eye structure)
v_ortho = [-0.40, -0.14, 0.19,  0.29 ]  ← PURE sunglasses direction
           ↓       ↓      ↓      ↓
          Skin   Hair   Frame   Lens
          tone   color
          (−)    (−)    (+)    (+)
```

**Orthogonal vector:**
- Ch0, Ch1 (skin/hair) → **Negative** = DON'T change these features
- Ch2, Ch3 (sunglasses) → **Positive** = Only add sunglasses

---

## 📐 Step 5: Orthogonal Projection for All Tokens

```python
# Calculate orthogonal direction for each token
ortho_all = []
for i in range(16):
    v_edit = attn_edit[i]
    v_null = attn_null[i]
    
    dot = sum(e * n for e, n in zip(v_edit, v_null))
    norm_sq = sum(n * n for n in v_null)
    proj = [(dot / norm_sq) * n for n in v_null]
    ortho = [e - p for e, p in zip(v_edit, proj)]
    
    ortho_all.append(ortho)
```

**Result (orthogonal directions):**

```python
ortho_all = [
    # Ch0    Ch1    Ch2    Ch3     Token    Comment
    [-0.02, -0.01,  0.00, -0.01],  # 0       Background: ~0 (unchanged)
    [-0.01, -0.01, -0.01,  0.00],  # 1       Background: ~0
    [-0.01, -0.01, -0.01,  0.00],  # 2       Background: ~0
    [-0.02, -0.01,  0.00, -0.01],  # 3       Background: ~0
    [-0.03, -0.02, -0.01, -0.01],  # 4       Hair: ~0 (preserved)
    [-0.04, -0.01,  0.02,  0.01],  # 5       Forehead: ~0
    [-0.04, -0.01,  0.02,  0.01],  # 6       Forehead: ~0
    [-0.03, -0.02, -0.01, -0.01],  # 7       Hair: ~0
    [-0.05, -0.01,  0.03,  0.02],  # 8       Cheek: ~0
    [-0.40, -0.14,  0.19,  0.29],  # 9 ⭐    LEFT EYE: STRONG!
    [-0.41, -0.13,  0.20,  0.30],  # 10 ⭐   RIGHT EYE: STRONG!
    [-0.05, -0.01,  0.03,  0.02],  # 11      Cheek: ~0
    [-0.05, -0.01,  0.02,  0.01],  # 12      Neck: ~0
    [-0.05, -0.01,  0.02,  0.01],  # 13      Mouth: ~0
    [-0.05, -0.01,  0.02,  0.01],  # 14      Mouth: ~0
    [-0.05, -0.01,  0.02,  0.01],  # 15      Neck: ~0
]
```

**Observation:** Only Tokens 9 and 10 (eye regions) have significant orthogonal directions!

---

## 📐 Step 6: Spatial Masking

### Mask Calculation via Cross-Attention:

```python
# Calculate attention between each token and "sunglasses" text token
# Q: image tokens (16, 4)
# K: "sunglasses" text token (1, 4) - simplified

text_sunglasses = [0.2, 0.1, 0.9, 0.8]  # Sunglasses embedding

attention_scores = []
for token in tokens:
    # Dot product attention
    score = sum(t * s for t, s in zip(token, text_sunglasses))
    attention_scores.append(score)
```

**Raw attention scores:**

```python
scores_raw = [
    0.12,  # Token 0: background
    0.11,  # Token 1: background
    0.11,  # Token 2: background
    0.12,  # Token 3: background
    0.18,  # Token 4: hair
    0.21,  # Token 5: forehead
    0.22,  # Token 6: forehead
    0.19,  # Token 7: hair
    0.24,  # Token 8: cheek
    0.78,  # Token 9: LEFT EYE ⭐⭐
    0.79,  # Token 10: RIGHT EYE ⭐⭐
    0.23,  # Token 11: cheek
    0.22,  # Token 12: neck
    0.19,  # Token 13: mouth
    0.20,  # Token 14: mouth
    0.21,  # Token 15: neck
]
```

### Mask Creation (3 steps):

```python
# 1. Min-max normalization
min_val = 0.11
max_val = 0.79
normalized = [(s - min_val) / (max_val - min_val) for s in scores_raw]

# normalized:
# [0.01, 0.00, 0.00, 0.01, 0.10, 0.15, 0.16, 0.12, 0.19, 0.99, 1.00, 0.18, ...]

# 2. Sigmoid sharpening (d=10)
import math
def sigmoid(x): return 1 / (1 + math.exp(-x))
sharpened = [sigmoid(10 * (n - 0.5)) for n in normalized]

# sharpened:
# [0.01, 0.01, 0.01, 0.01, 0.02, 0.03, 0.03, 0.02, 0.04, 0.99, 0.99, 0.04, ...]

# 3. Threshold (τ = 0.5)
mask = [1.0 if s >= 0.5 else 0.0 for s in sharpened]
```

**Final Mask:**

```python
mask = [
    0.0,  # Token 0: background → MASKED
    0.0,  # Token 1: background → MASKED
    0.0,  # Token 2: background → MASKED
    0.0,  # Token 3: background → MASKED
    0.0,  # Token 4: hair → MASKED
    0.0,  # Token 5: forehead → MASKED
    0.0,  # Token 6: forehead → MASKED
    0.0,  # Token 7: hair → MASKED
    0.0,  # Token 8: cheek → MASKED
    1.0,  # Token 9: LEFT EYE → EDIT WILL BE APPLIED ✓
    1.0,  # Token 10: RIGHT EYE → EDIT WILL BE APPLIED ✓
    0.0,  # Token 11: cheek → MASKED
    0.0,  # Token 12: neck → MASKED
    0.0,  # Token 13: mouth → MASKED
    0.0,  # Token 14: mouth → MASKED
    0.0,  # Token 15: neck → MASKED
]
```

**Mask Visualization (4×4 grid):**

```
     Col 0    Col 1    Col 2    Col 3
    ┌────────┬────────┬────────┬────────┐
Row 0│   0    │   0    │   0    │   0    │
    ├────────┼────────┼────────┼────────┤
Row 1│   0    │   0    │   0    │   0    │
    ├────────┼────────┼────────┼────────┤
Row 2│   0    │   1 ✓  │   1 ✓  │   0    │  ← ONLY EYE REGION!
    ├────────┼────────┼────────┼────────┤
Row 3│   0    │   0    │   0    │   0    │
    └────────┴────────┴────────┴────────┘
```

---

## 📐 Step 7: Masked Orthogonal Direction

```python
# ortho × mask
masked_ortho = []
for i in range(16):
    masked = [o * mask[i] for o in ortho_all[i]]
    masked_ortho.append(masked)
```

**Result:**

```python
masked_ortho = [
    [0.00, 0.00, 0.00, 0.00],  # Token 0: × 0 = zeroed
    [0.00, 0.00, 0.00, 0.00],  # Token 1
    [0.00, 0.00, 0.00, 0.00],  # Token 2
    [0.00, 0.00, 0.00, 0.00],  # Token 3
    [0.00, 0.00, 0.00, 0.00],  # Token 4: hair PRESERVED
    [0.00, 0.00, 0.00, 0.00],  # Token 5: forehead PRESERVED
    [0.00, 0.00, 0.00, 0.00],  # Token 6
    [0.00, 0.00, 0.00, 0.00],  # Token 7
    [0.00, 0.00, 0.00, 0.00],  # Token 8: cheek PRESERVED
    [-0.40, -0.14, 0.19, 0.29],  # Token 9: LEFT EYE → EDIT WILL BE APPLIED
    [-0.41, -0.13, 0.20, 0.30],  # Token 10: RIGHT EYE → EDIT WILL BE APPLIED
    [0.00, 0.00, 0.00, 0.00],  # Token 11: cheek PRESERVED
    [0.00, 0.00, 0.00, 0.00],  # Token 12
    [0.00, 0.00, 0.00, 0.00],  # Token 13
    [0.00, 0.00, 0.00, 0.00],  # Token 14
    [0.00, 0.00, 0.00, 0.00],  # Token 15
]
```

---

## 📐 Step 8: Final Attention Output

```python
# λ = edit strength (e.g., 3.0)
lambda_fine = 3.0

# final_attn = base_attn + λ × masked_ortho
final_attn = []
for i in range(16):
    final = [b + lambda_fine * m for b, m in zip(attn_base[i], masked_ortho[i])]
    final_attn.append(final)
```

**Calculation for Token 9 (Left Eye):**

```python
base_token_9  = [0.35, 0.15, 0.42, 0.28]
masked_ortho_9 = [-0.40, -0.14, 0.19, 0.29]

final_token_9 = [
    0.35 + 3.0 × (-0.40),  # Ch0: 0.35 - 1.20 = -0.85 → clamp → 0
    0.15 + 3.0 × (-0.14),  # Ch1: 0.15 - 0.42 = -0.27 → clamp → 0
    0.42 + 3.0 × (0.19),   # Ch2: 0.42 + 0.57 = 0.99  ← GLASSES FRAME
    0.28 + 3.0 × (0.29),   # Ch3: 0.28 + 0.87 = 1.15  ← GLASSES LENS
]
```

### Norm Preservation:

```python
# Preserve original norm (for image stability)
original_norm = sqrt(0.35² + 0.15² + 0.42² + 0.28²) = 0.59

# New vector's norm
new_norm = sqrt(0² + 0² + 0.99² + 1.15²) = 1.52

# Normalize
final_token_9_normalized = [v / new_norm * original_norm for v in final_token_9]
                         = [0.00, 0.00, 0.38, 0.45]
```

---

## 📊 Visual Comparison

### Token 9 (Left Eye) - Channel Values:

```
                Ch0(skin) Ch1(hair)  Ch2(frame)   Ch3(lens)
                
Base:            0.35      0.15       0.42         0.28
                   ↓         ↓          ↓            ↓
                 [skin]   [brown]    [eye]        [eye]
                 
Normal Edit:     0.15      0.10       0.85         0.72
                   ↓         ↓          ↓            ↓
                [faded]  [changed]  [glasses+]   [glasses+]
                ❌ Skin tone lost!
                ❌ Hair color changed!
                
FluxSpace:       0.00      0.00       0.38         0.45
                   ↓         ↓          ↓            ↓
                [zero]   [zero]     [glasses]    [glasses]
                
                But ADDED to base:
Final:           0.35      0.15       0.80         0.73
                   ↓         ↓          ↓            ↓
                [SAME!]  [SAME!]    [glasses+]   [glasses+]
                ✅ Skin tone preserved!
                ✅ Hair color preserved!
```

---

## 📊 Result on 32×32 Pixel Image

```
NORMAL EDIT                           FLUXSPACE EDIT
┌──────────────────────────────────┐  ┌──────────────────────────────────┐
│  Background: ❌ CORRUPTED        │  │  Background: ✅ SAME             │
├──────────────────────────────────┤  ├──────────────────────────────────┤
│  Hair: ❌ COLOR CHANGED          │  │  Hair: ✅ SAME                   │
├──────────────────────────────────┤  ├──────────────────────────────────┤
│  Forehead: ❌ SKIN TONE CHANGED  │  │  Forehead: ✅ SAME               │
├──────────────────────────────────┤  ├──────────────────────────────────┤
│  Eyes: ✅ GLASSES ADDED          │  │  Eyes: ✅ GLASSES ADDED          │
│  (but eye shape corrupted ❌)    │  │  (eye shape PRESERVED ✅)        │
├──────────────────────────────────┤  ├──────────────────────────────────┤
│  Mouth: ❌ COLOR CHANGED         │  │  Mouth: ✅ SAME                  │
├──────────────────────────────────┤  ├──────────────────────────────────┤
│  Neck: ❌ CORRUPTED              │  │  Neck: ✅ SAME                   │
└──────────────────────────────────┘  └──────────────────────────────────┘

Changed pixel count:                  Changed pixel count:
~800 pixels (entire image)            ~64 pixels (only eye region 8×8)
```

---

## 🧮 Summary: FluxSpace by the Numbers

| Metric | Normal Edit | FluxSpace |
|--------|-------------|-----------|
| Changed token count | 16/16 (all) | 2/16 (only eyes) |
| Changed pixel count | ~1024 (32×32) | ~128 (8×16) |
| Background change | 45% | 0% |
| Hair change | 38% | 0% |
| Skin tone change | 52% | 0% |
| Eye region change | 100% (target) | 100% (target) |

**Conclusion:** FluxSpace changes only the eye region when adding sunglasses, while normal edit corrupts the entire 32×32 image.
