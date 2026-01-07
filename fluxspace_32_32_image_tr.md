# FluxSpace: 32×32×3 RGB Image Örneği ile Açıklama

## 🖼️ Senaryo

Bir **32×32 pixel RGB görüntümüz** var:
- Kadın yüzü (göz bölgesi ortada)
- Edit: "Gözlük ekle"

```
Görüntü Yapısı (32×32×3):
┌──────────────────────────────────┐
│  Arka plan (mavi gökyüzü)        │  Row 0-7
├──────────────────────────────────┤
│  Saç (kahverengi)                │  Row 8-11
├──────────────────────────────────┤
│  Alın (ten rengi)                │  Row 12-14
├──────────────────────────────────┤
│  GÖZLER (hedef bölge)            │  Row 15-17
├──────────────────────────────────┤
│  Burun, Ağız (ten rengi)         │  Row 18-24
├──────────────────────────────────┤
│  Boyun, Omuz                     │  Row 25-31
└──────────────────────────────────┘
```

---

## 📊 Adım 1: Image → Latent Space

Diffusion modelleri doğrudan pixel üzerinde çalışmaz. Önce **VAE encoder** ile latent space'e çevrilir:

```python
# Orijinal görüntü: 32×32×3 (H×W×C)
image = torch.randn(32, 32, 3)  # RGB değerleri [0, 255] veya [-1, 1]

# VAE Encoder: 8× downsampling
# 32/8 = 4, yani latent boyutu: 4×4×4 (veya 4×4×16 Flux için)
latent = vae.encode(image)  # Shape: (4, 4, 4)
```

### Latent Grid Yapısı (4×4):

```
Latent patch'leri (her biri 8×8 pixel bölgesini temsil eder):

     Col 0    Col 1    Col 2    Col 3
    ┌────────┬────────┬────────┬────────┐
Row 0│ Arka   │ Arka   │ Arka   │ Arka   │  ← Gökyüzü
    ├────────┼────────┼────────┼────────┤
Row 1│ Saç    │ Alın   │ Alın   │ Saç    │  ← Saç + Alın
    ├────────┼────────┼────────┼────────┤
Row 2│ Yanık  │ SOL GÖZ│ SAĞ GÖZ│ Yanık  │  ← GÖZ BÖLGESİ ⭐
    ├────────┼────────┼────────┼────────┤
Row 3│ Boyun  │ Ağız   │ Ağız   │ Boyun  │  ← Alt yüz
    └────────┴────────┴────────┴────────┘
```

---

## 📊 Adım 2: Latent → Attention Tokens

Transformer, latent'i **token sequence** olarak işler:

```python
# Latent: 4×4×4 → Flatten → 16 tokens × 4 channels
# (Gerçekte Flux 3072 dim kullanır, biz basitleştiriyoruz)

tokens = latent.flatten(0, 1)  # Shape: (16, 4)

# Her token'ın anlamı:
token_map = {
    0: "arka_sol_üst",     1: "arka_orta_üst",    2: "arka_orta_üst",    3: "arka_sağ_üst",
    4: "saç_sol",          5: "alın_sol",         6: "alın_sağ",         7: "saç_sağ",
    8: "yanak_sol",        9: "SOL_GÖZ ⭐",       10: "SAĞ_GÖZ ⭐",      11: "yanak_sağ",
    12: "boyun_sol",       13: "ağız_sol",        14: "ağız_sağ",        15: "boyun_sağ"
}
```

---

## 📐 Adım 3: Üç Attention Geçişi

FluxSpace üç paralel attention hesaplar:

### A) Base Attention (Orijinal prompt: "a woman")

```python
# Text embedding: "a woman" → 77 tokens × 4 dim (basitleştirilmiş)
text_base = encode("a woman")  # Shape: (77, 4)

# Attention: image tokens × text tokens
# Q: image (16, 4), K: text (77, 4), V: text (77, 4)
attn_base = attention(Q=tokens, K=text_base, V=text_base)
# Output shape: (16, 4)
```

**Base attention output (16 token × 4 channel):**

```python
attn_base = [
    # Ch0   Ch1   Ch2   Ch3    ← Channels (feature dimensions)
    [0.12, 0.08, 0.15, 0.10],  # Token 0: arka_sol_üst
    [0.11, 0.09, 0.14, 0.11],  # Token 1: arka_orta
    [0.11, 0.09, 0.14, 0.11],  # Token 2: arka_orta
    [0.12, 0.08, 0.15, 0.10],  # Token 3: arka_sağ_üst
    [0.25, 0.45, 0.12, 0.08],  # Token 4: saç_sol (kahverengi)
    [0.72, 0.18, 0.05, 0.03],  # Token 5: alın (ten rengi)
    [0.71, 0.19, 0.06, 0.04],  # Token 6: alın
    [0.24, 0.46, 0.11, 0.09],  # Token 7: saç_sağ
    [0.70, 0.20, 0.08, 0.05],  # Token 8: yanak_sol
    [0.35, 0.15, 0.42, 0.28],  # Token 9: SOL_GÖZ ⭐
    [0.34, 0.16, 0.43, 0.27],  # Token 10: SAĞ_GÖZ ⭐
    [0.69, 0.21, 0.07, 0.06],  # Token 11: yanak_sağ
    [0.68, 0.22, 0.08, 0.07],  # Token 12: boyun_sol
    [0.75, 0.15, 0.06, 0.04],  # Token 13: ağız
    [0.74, 0.16, 0.07, 0.05],  # Token 14: ağız
    [0.67, 0.23, 0.09, 0.08],  # Token 15: boyun_sağ
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
    [0.10, 0.07, 0.12, 0.08],  # Token 0: arka (az etkilendi)
    [0.09, 0.08, 0.11, 0.09],  # Token 1
    [0.09, 0.08, 0.11, 0.09],  # Token 2
    [0.10, 0.07, 0.12, 0.08],  # Token 3
    [0.22, 0.42, 0.10, 0.07],  # Token 4: saç (az etkilendi)
    [0.68, 0.17, 0.08, 0.05],  # Token 5: alın (az etkilendi)
    [0.67, 0.18, 0.09, 0.06],  # Token 6
    [0.21, 0.43, 0.09, 0.08],  # Token 7: saç
    [0.65, 0.19, 0.12, 0.08],  # Token 8: yanak
    [0.15, 0.10, 0.85, 0.72],  # Token 9: SOL_GÖZ ⭐⭐ (GÜÇLÜ!)
    [0.14, 0.11, 0.86, 0.71],  # Token 10: SAĞ_GÖZ ⭐⭐ (GÜÇLÜ!)
    [0.64, 0.20, 0.11, 0.09],  # Token 11: yanak
    [0.63, 0.21, 0.10, 0.08],  # Token 12: boyun
    [0.70, 0.14, 0.08, 0.06],  # Token 13: ağız
    [0.69, 0.15, 0.09, 0.07],  # Token 14
    [0.62, 0.22, 0.11, 0.09],  # Token 15: boyun
]
```

**Dikkat:** Token 9 ve 10 (göz bölgeleri) Ch2 ve Ch3'te çok yüksek değerler aldı!

### C) Null Attention (Boş prompt: "")

```python
text_null = encode("")  # Shape: (77, 4) - sadece padding
attn_null = attention(Q=tokens, K=text_null, V=text_null)
```

**Null attention output (image prior - görüntüyü koruma yönü):**

```python
attn_null = [
    # Ch0   Ch1   Ch2   Ch3
    [0.11, 0.08, 0.14, 0.09],  # Token 0
    [0.10, 0.09, 0.13, 0.10],  # Token 1
    [0.10, 0.09, 0.13, 0.10],  # Token 2
    [0.11, 0.08, 0.14, 0.09],  # Token 3
    [0.24, 0.44, 0.11, 0.08],  # Token 4: saç
    [0.71, 0.18, 0.06, 0.04],  # Token 5: alın
    [0.70, 0.19, 0.07, 0.05],  # Token 6
    [0.23, 0.45, 0.10, 0.09],  # Token 7
    [0.69, 0.20, 0.09, 0.06],  # Token 8: yanak
    [0.34, 0.15, 0.41, 0.27],  # Token 9: SOL_GÖZ
    [0.33, 0.16, 0.42, 0.26],  # Token 10: SAĞ_GÖZ
    [0.68, 0.21, 0.08, 0.07],  # Token 11
    [0.67, 0.22, 0.09, 0.08],  # Token 12
    [0.74, 0.15, 0.07, 0.05],  # Token 13
    [0.73, 0.16, 0.08, 0.06],  # Token 14
    [0.66, 0.23, 0.10, 0.09],  # Token 15
]
```

---

## 📐 Adım 4: Ortogonal Projeksiyon (Token 9 - Sol Göz için detaylı)

### Token 9 (Sol Göz) için hesaplama:

```python
# Vektörler (4 boyutlu)
v_edit = [0.15, 0.10, 0.85, 0.72]  # Edit attention (sunglasses)
v_null = [0.34, 0.15, 0.41, 0.27]  # Null attention (image prior)

# Adım 1: Dot product
dot_product = (0.15 × 0.34) + (0.10 × 0.15) + (0.85 × 0.41) + (0.72 × 0.27)
            = 0.051 + 0.015 + 0.349 + 0.194
            = 0.609

# Adım 2: Null vektörün norm karesi
norm_sq = 0.34² + 0.15² + 0.41² + 0.27²
        = 0.116 + 0.023 + 0.168 + 0.073
        = 0.380

# Adım 3: Projeksiyon katsayısı
coef = dot_product / norm_sq = 0.609 / 0.380 = 1.603

# Adım 4: Projeksiyon vektörü
projection = coef × v_null
           = 1.603 × [0.34, 0.15, 0.41, 0.27]
           = [0.545, 0.240, 0.657, 0.433]

# Adım 5: Ortogonal yön (SAF edit yönü)
v_ortho = v_edit - projection
        = [0.15, 0.10, 0.85, 0.72] - [0.545, 0.240, 0.657, 0.433]
        = [-0.395, -0.140, 0.193, 0.287]
```

### Sonuç Yorumu (Token 9):

```
v_edit  = [0.15,  0.10,  0.85,  0.72 ]  ← "sunglasses" etkisi
v_null  = [0.34,  0.15,  0.41,  0.27 ]  ← Image prior (göz yapısı koruma)
v_ortho = [-0.40, -0.14, 0.19,  0.29 ]  ← SAF gözlük yönü
           ↓       ↓      ↓      ↓
          Ten    Saç   Gözlük Gözlük
          rengi  rengi  çerçeve cam
          (−)    (−)    (+)    (+)
```

**Ortogonal vektör:**
- Ch0, Ch1 (ten/saç) → **Negatif** = Bu özellikleri değiştirME
- Ch2, Ch3 (gözlük) → **Pozitif** = Sadece gözlük ekle

---

## 📐 Adım 5: Tüm Token'lar için Ortogonal Projeksiyon

```python
# Her token için ortogonal yönü hesapla
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

**Sonuç (ortogonal yönler):**

```python
ortho_all = [
    # Ch0    Ch1    Ch2    Ch3     Token    Yorum
    [-0.02, -0.01,  0.00, -0.01],  # 0       Arka plan: ~0 (değişmez)
    [-0.01, -0.01, -0.01,  0.00],  # 1       Arka plan: ~0
    [-0.01, -0.01, -0.01,  0.00],  # 2       Arka plan: ~0
    [-0.02, -0.01,  0.00, -0.01],  # 3       Arka plan: ~0
    [-0.03, -0.02, -0.01, -0.01],  # 4       Saç: ~0 (korunuyor)
    [-0.04, -0.01,  0.02,  0.01],  # 5       Alın: ~0
    [-0.04, -0.01,  0.02,  0.01],  # 6       Alın: ~0
    [-0.03, -0.02, -0.01, -0.01],  # 7       Saç: ~0
    [-0.05, -0.01,  0.03,  0.02],  # 8       Yanak: ~0
    [-0.40, -0.14,  0.19,  0.29],  # 9 ⭐    SOL GÖZ: GÜÇLÜ!
    [-0.41, -0.13,  0.20,  0.30],  # 10 ⭐   SAĞ GÖZ: GÜÇLÜ!
    [-0.05, -0.01,  0.03,  0.02],  # 11      Yanak: ~0
    [-0.05, -0.01,  0.02,  0.01],  # 12      Boyun: ~0
    [-0.05, -0.01,  0.02,  0.01],  # 13      Ağız: ~0
    [-0.05, -0.01,  0.02,  0.01],  # 14      Ağız: ~0
    [-0.05, -0.01,  0.02,  0.01],  # 15      Boyun: ~0
]
```

**Gözlem:** Sadece Token 9 ve 10 (göz bölgeleri) anlamlı ortogonal yöne sahip!

---

## 📐 Adım 6: Spatial Masking

### Cross-Attention ile Maske Hesaplama:

```python
# Her token için "sunglasses" text token'ı ile attention hesapla
# Q: image tokens (16, 4)
# K: "sunglasses" text token (1, 4) - basitleştirilmiş

text_sunglasses = [0.2, 0.1, 0.9, 0.8]  # Gözlük embedding

attention_scores = []
for token in tokens:
    # Dot product attention
    score = sum(t * s for t, s in zip(token, text_sunglasses))
    attention_scores.append(score)
```

**Ham attention skorları:**

```python
scores_raw = [
    0.12,  # Token 0: arka
    0.11,  # Token 1: arka
    0.11,  # Token 2: arka
    0.12,  # Token 3: arka
    0.18,  # Token 4: saç
    0.21,  # Token 5: alın
    0.22,  # Token 6: alın
    0.19,  # Token 7: saç
    0.24,  # Token 8: yanak
    0.78,  # Token 9: SOL GÖZ ⭐⭐
    0.79,  # Token 10: SAĞ GÖZ ⭐⭐
    0.23,  # Token 11: yanak
    0.22,  # Token 12: boyun
    0.19,  # Token 13: ağız
    0.20,  # Token 14: ağız
    0.21,  # Token 15: boyun
]
```

### Maske Oluşturma (3 adım):

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

**Final Maske:**

```python
mask = [
    0.0,  # Token 0: arka → MASKELENDI
    0.0,  # Token 1: arka → MASKELENDI
    0.0,  # Token 2: arka → MASKELENDI
    0.0,  # Token 3: arka → MASKELENDI
    0.0,  # Token 4: saç → MASKELENDI
    0.0,  # Token 5: alın → MASKELENDI
    0.0,  # Token 6: alın → MASKELENDI
    0.0,  # Token 7: saç → MASKELENDI
    0.0,  # Token 8: yanak → MASKELENDI
    1.0,  # Token 9: SOL GÖZ → EDIT UYGULANACAK ✓
    1.0,  # Token 10: SAĞ GÖZ → EDIT UYGULANACAK ✓
    0.0,  # Token 11: yanak → MASKELENDI
    0.0,  # Token 12: boyun → MASKELENDI
    0.0,  # Token 13: ağız → MASKELENDI
    0.0,  # Token 14: ağız → MASKELENDI
    0.0,  # Token 15: boyun → MASKELENDI
]
```

**Maske Görselleştirme (4×4 grid):**

```
     Col 0    Col 1    Col 2    Col 3
    ┌────────┬────────┬────────┬────────┐
Row 0│   0    │   0    │   0    │   0    │
    ├────────┼────────┼────────┼────────┤
Row 1│   0    │   0    │   0    │   0    │
    ├────────┼────────┼────────┼────────┤
Row 2│   0    │   1 ✓  │   1 ✓  │   0    │  ← SADECE GÖZ BÖLGESİ!
    ├────────┼────────┼────────┼────────┤
Row 3│   0    │   0    │   0    │   0    │
    └────────┴────────┴────────┴────────┘
```

---

## 📐 Adım 7: Maskelenmiş Ortogonal Yön

```python
# ortho × mask
masked_ortho = []
for i in range(16):
    masked = [o * mask[i] for o in ortho_all[i]]
    masked_ortho.append(masked)
```

**Sonuç:**

```python
masked_ortho = [
    [0.00, 0.00, 0.00, 0.00],  # Token 0: × 0 = sıfırlandı
    [0.00, 0.00, 0.00, 0.00],  # Token 1
    [0.00, 0.00, 0.00, 0.00],  # Token 2
    [0.00, 0.00, 0.00, 0.00],  # Token 3
    [0.00, 0.00, 0.00, 0.00],  # Token 4: saç KORUNDU
    [0.00, 0.00, 0.00, 0.00],  # Token 5: alın KORUNDU
    [0.00, 0.00, 0.00, 0.00],  # Token 6
    [0.00, 0.00, 0.00, 0.00],  # Token 7
    [0.00, 0.00, 0.00, 0.00],  # Token 8: yanak KORUNDU
    [-0.40, -0.14, 0.19, 0.29],  # Token 9: SOL GÖZ → EDIT UYGULANACAK
    [-0.41, -0.13, 0.20, 0.30],  # Token 10: SAĞ GÖZ → EDIT UYGULANACAK
    [0.00, 0.00, 0.00, 0.00],  # Token 11: yanak KORUNDU
    [0.00, 0.00, 0.00, 0.00],  # Token 12
    [0.00, 0.00, 0.00, 0.00],  # Token 13
    [0.00, 0.00, 0.00, 0.00],  # Token 14
    [0.00, 0.00, 0.00, 0.00],  # Token 15
]
```

---

## 📐 Adım 8: Final Attention Output

```python
# λ = edit strength (örn: 3.0)
lambda_fine = 3.0

# final_attn = base_attn + λ × masked_ortho
final_attn = []
for i in range(16):
    final = [b + lambda_fine * m for b, m in zip(attn_base[i], masked_ortho[i])]
    final_attn.append(final)
```

**Token 9 (Sol Göz) için hesaplama:**

```python
base_token_9  = [0.35, 0.15, 0.42, 0.28]
masked_ortho_9 = [-0.40, -0.14, 0.19, 0.29]

final_token_9 = [
    0.35 + 3.0 × (-0.40),  # Ch0: 0.35 - 1.20 = -0.85 → clamp → 0
    0.15 + 3.0 × (-0.14),  # Ch1: 0.15 - 0.42 = -0.27 → clamp → 0
    0.42 + 3.0 × (0.19),   # Ch2: 0.42 + 0.57 = 0.99  ← GÖZLÜK ÇERÇEVESİ
    0.28 + 3.0 × (0.29),   # Ch3: 0.28 + 0.87 = 1.15  ← GÖZLÜK CAMI
]
```

### Norm Preservation:

```python
# Orijinal norm'u koru (görüntü stabilitesi için)
original_norm = sqrt(0.35² + 0.15² + 0.42² + 0.28²) = 0.59

# Yeni vektörün norm'u
new_norm = sqrt(0² + 0² + 0.99² + 1.15²) = 1.52

# Normalize et
final_token_9_normalized = [v / new_norm * original_norm for v in final_token_9]
                         = [0.00, 0.00, 0.38, 0.45]
```

---

## 📊 Görsel Karşılaştırma

### Token 9 (Sol Göz) - Channel Değerleri:

```
                Ch0(ten)  Ch1(saç)  Ch2(çerçeve)  Ch3(cam)
                
Base:            0.35      0.15       0.42         0.28
                   ↓         ↓          ↓            ↓
                 [ten]    [kahve]    [göz]        [göz]
                 
Normal Edit:     0.15      0.10       0.85         0.72
                   ↓         ↓          ↓            ↓
                [soluk]  [değişti]   [gözlük+]   [gözlük+]
                ❌ Ten kayboldu!
                ❌ Saç bozuldu!
                
FluxSpace:       0.00      0.00       0.38         0.45
                   ↓         ↓          ↓            ↓
                [sıfır]  [sıfır]    [gözlük]    [gözlük]
                
                Ama base'e EKLENİYOR:
Final:           0.35      0.15       0.80         0.73
                   ↓         ↓          ↓            ↓
                [AYNI!]  [AYNI!]    [gözlük+]   [gözlük+]
                ✅ Ten korundu!
                ✅ Saç korundu!
```

---

## 📊 32×32 Pixel Görüntüde Sonuç

```
NORMAL EDIT                           FLUXSPACE EDIT
┌──────────────────────────────────┐  ┌──────────────────────────────────┐
│  Arka plan: ❌ BOZULDU           │  │  Arka plan: ✅ AYNI              │
├──────────────────────────────────┤  ├──────────────────────────────────┤
│  Saç: ❌ RENGİ DEĞİŞTİ          │  │  Saç: ✅ AYNI                    │
├──────────────────────────────────┤  ├──────────────────────────────────┤
│  Alın: ❌ TEN RENGİ DEĞİŞTİ     │  │  Alın: ✅ AYNI                   │
├──────────────────────────────────┤  ├──────────────────────────────────┤
│  Gözler: ✅ GÖZLÜK EKLENDİ      │  │  Gözler: ✅ GÖZLÜK EKLENDİ       │
│  (ama göz şekli bozuldu ❌)      │  │  (göz şekli KORUNDU ✅)          │
├──────────────────────────────────┤  ├──────────────────────────────────┤
│  Ağız: ❌ RENGİ DEĞİŞTİ         │  │  Ağız: ✅ AYNI                   │
├──────────────────────────────────┤  ├──────────────────────────────────┤
│  Boyun: ❌ BOZULDU               │  │  Boyun: ✅ AYNI                  │
└──────────────────────────────────┘  └──────────────────────────────────┘

Değişen pixel sayısı:                 Değişen pixel sayısı:
~800 pixel (tüm görüntü)              ~64 pixel (sadece göz bölgesi 8×8)
```

---

## 🧮 Özet: Sayılarla FluxSpace

| Metrik | Normal Edit | FluxSpace |
|--------|-------------|-----------|
| Değişen token sayısı | 16/16 (tümü) | 2/16 (sadece gözler) |
| Değişen pixel sayısı | ~1024 (32×32) | ~128 (8×16) |
| Arka plan değişimi | %45 | %0 |
| Saç değişimi | %38 | %0 |
| Ten rengi değişimi | %52 | %0 |
| Göz bölgesi değişimi | %100 (hedef) | %100 (hedef) |

**Sonuç:** FluxSpace, gözlük eklemek için sadece göz bölgesini değiştirirken, normal edit tüm 32×32 görüntüyü bozuyor.
