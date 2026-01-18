# Testing the Volterra Model for SGD on Real Data

## Kapsamlı ve İntuitive Proje Açıklaması

---

# BÖLÜM A: İNTUİTİF ANLAYIŞ

---

## A1. Model Ne Yapıyor? (Büyük Resim)

### A1.1 En Basit Haliyle

Bir **lineer denklem sistemi** çözmeye çalışıyoruz:

$$A\mathbf{x} = \mathbf{b}$$

- $A$: Bildiğimiz matris (features/özellikler)
- $\mathbf{b}$: Bildiğimiz hedef (labels/etiketler)
- $\mathbf{x}$: **Bulmaya çalıştığımız** weight vektörü

### A1.2 Günlük Hayat Analojisi

Düşün ki bir **tarif** bulmaya çalışıyorsun:

| Malzeme Miktarları ($A$) | Sonuç ($\mathbf{b}$) |
|--------------------------|----------------------|
| 2 yumurta, 1 su bardağı süt, 3 kaşık un | Krep |
| 3 yumurta, 0 süt, 2 kaşık un | Omlet |
| 1 yumurta, 2 su bardağı süt, 1 kaşık un | Puding |

**Soru:** Her malzemenin "lezzet katkısı" ($\mathbf{x}$) nedir ki, malzeme miktarlarını bu katkılarla çarpıp toplarsak sonucu bulabilelim?

$$\text{Sonuç} = x_{\text{yumurta}} \cdot \text{(yumurta miktarı)} + x_{\text{süt}} \cdot \text{(süt miktarı)} + x_{\text{un}} \cdot \text{(un miktarı)}$$

### A1.3 Matris Formunda (Küçük Örnek)

3 örnek, 2 feature için:

$$\underbrace{\begin{bmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \\ a_{31} & a_{32} \end{bmatrix}}_{A: \text{features}} \cdot \underbrace{\begin{bmatrix} x_1 \\ x_2 \end{bmatrix}}_{\mathbf{x}: \text{weights}} = \underbrace{\begin{bmatrix} b_1 \\ b_2 \\ b_3 \end{bmatrix}}_{\mathbf{b}: \text{labels}}$$

**Açılmış hali:**
$$\begin{cases}
a_{11} x_1 + a_{12} x_2 = b_1 & \text{(1. örnek için)} \\
a_{21} x_1 + a_{22} x_2 = b_2 & \text{(2. örnek için)} \\
a_{31} x_1 + a_{32} x_2 = b_3 & \text{(3. örnek için)}
\end{cases}$$

---

## A2. Labels Nasıl Kullanılıyor?

### A2.1 Label Nedir?

**Label ($\mathbf{b}$):** Her örnek için bildiğimiz "doğru cevap" veya "hedef değer".

Bizim projemizde **3 farklı şekilde** label oluşturuyoruz:

### A2.2 Setting 1: Linear Target (Base)

$$\mathbf{b} = A \mathbf{x}^*$$

**İntuition:**
- Önce rastgele bir "gizli cevap" $\mathbf{x}^*$ üretiyoruz
- Sonra bu cevabı $A$ ile çarparak label'ları oluşturuyoruz
- SGD'nin görevi: $\mathbf{x}^*$'ı **yeniden keşfetmek**

**Analoji:** Öğretmen cevap anahtarını ($\mathbf{x}^*$) biliyor, öğrenciye sadece soruları ($A$) ve sonuçları ($\mathbf{b}$) veriyor. Öğrenci cevap anahtarını bulmaya çalışıyor.

**Matris formu (küçük örnek):**
```
x* = [0.5, -0.3]  (gizli cevap)

A = [1.0  2.0]    b = A @ x* = [1.0×0.5 + 2.0×(-0.3)]   = [-0.1]
    [0.5  1.0]                 [0.5×0.5 + 1.0×(-0.3)]   = [-0.05]
    [2.0  0.0]                 [2.0×0.5 + 0.0×(-0.3)]   = [1.0]
```

### A2.3 Setting 2: Nonlinear Target

$$\mathbf{b} = \text{ReLU}(A \mathbf{x}^*)$$

**İntuition:**
- Aynı şekilde $A\mathbf{x}^*$ hesapla
- Sonra **negatif değerleri sıfırla** (ReLU uygula)
- Artık "mükemmel cevap" **yok** - çünkü bilgi kaybettik

**Matris formu:**
```
A @ x* = [-0.1, -0.05, 1.0]  (intermediate)
b = ReLU(A @ x*) = [0, 0, 1.0]  (negatifler sıfırlandı!)
```

**Problem:** SGD artık $\mathbf{x}^*$'ı bulamaz çünkü $\mathbf{b}$ artık $A$'nın column space'inde değil!

### A2.4 Setting 3: Whitened

$$\tilde{A} = A \Sigma^{-1/2}, \quad \mathbf{b} = \tilde{A} \mathbf{x}^*$$

**İntuition:**
- $A$'yı "normalize" ediyoruz (tüm yönlerde eşit varyans)
- Bu teorinin varsayımına daha yakın

---

## A3. Model Neyi Öğreniyor/Tahmin Ediyor?

### A3.1 SGD Neyi Öğreniyor?

SGD, **weight vektörü $\mathbf{x}$**'i öğreniyor. Amaç:

$$\text{Loss} = \frac{1}{2n} \|A\mathbf{x} - \mathbf{b}\|^2 \to \text{minimize}$$

**Açılmış hali:**
$$\text{Loss} = \frac{1}{2n} \sum_{i=1}^{n} \underbrace{(A_i \cdot \mathbf{x} - b_i)^2}_{\text{i. örneğin hatası}^2}$$

Her örnek için:
- $A_i \cdot \mathbf{x}$: Modelin tahmini
- $b_i$: Gerçek label
- Farkın karesi: Hata

### A3.2 Volterra Neyi Tahmin Ediyor?

**Volterra, SGD'yi tahmin ediyor!**

Volterra, SGD'nin **loss curve**'unu (loss'un zamanla nasıl değiştiğini) tahmin eder:

```
İterasyon (k):  0    100    200    300    400    500
SGD Loss:      1.0   0.5    0.3    0.2   0.15   0.12   ← Gerçek (stokastik)
Volterra:      1.0   0.48   0.29   0.19  0.14   0.11   ← Tahmin (deterministik)
```

**İntuition:** SGD rastgele (her seferinde farklı örnek seç), ama Volterra deterministik formülle **ortalama davranışı** tahmin ediyor.

### A3.3 Görsel Açıklama

```
                    SGD Çalıştır (50 kez)
                           ↓
         ┌─────────────────────────────────┐
         │  Run 1: ~~~~~~~~~~~~~~~~~~~~~~  │
         │  Run 2: ~~~~~~~~~~~~~~~~~~~~~~  │
         │  Run 3: ~~~~~~~~~~~~~~~~~~~~~~  │
Loss     │  ...                            │
         │  Run 50: ~~~~~~~~~~~~~~~~~~~~~  │
         │                                 │
         │  Ortalama: ═══════════════════  │ ← Bunu tahmin ediyoruz
         └─────────────────────────────────┘
                    İterasyon →

         Volterra tahmini: ─ ─ ─ ─ ─ ─ ─   ← Tek bir formülle!
```

---

## A4. Aspect Ratio ($r$) Ne Anlama Geliyor?

### A4.1 Tanım

$$r = \frac{d}{n} = \frac{\text{feature sayısı}}{\text{örnek sayısı}}$$

### A4.2 Küçük Boyutlu Örnekler

#### Örnek 1: $r = 0.5$ (Overdetermined)

$n = 4$ örnek, $d = 2$ feature:

$$A = \begin{bmatrix} 1 & 2 \\ 3 & 1 \\ 2 & 2 \\ 1 & 3 \end{bmatrix}, \quad \mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}, \quad \mathbf{b} = \begin{bmatrix} b_1 \\ b_2 \\ b_3 \\ b_4 \end{bmatrix}$$

**4 denklem, 2 bilinmeyen:**
```
x₁ + 2x₂ = b₁
3x₁ + x₂ = b₂
2x₁ + 2x₂ = b₃
x₁ + 3x₂ = b₄
```

**Durum:** Denklem sayısı > bilinmeyen sayısı → **Overdetermined**
- Tam çözüm genelde **yok** (tüm denklemleri aynı anda sağlayan $\mathbf{x}$ yok)
- En iyi "yaklaşık" çözümü buluyoruz (least squares)
- **Az parametre** → Model basit → Overfitting riski düşük

#### Örnek 2: $r = 1.0$ (Square)

$n = 3$ örnek, $d = 3$ feature:

$$A = \begin{bmatrix} 1 & 2 & 1 \\ 3 & 1 & 2 \\ 2 & 2 & 1 \end{bmatrix}, \quad \mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix}, \quad \mathbf{b} = \begin{bmatrix} b_1 \\ b_2 \\ b_3 \end{bmatrix}$$

**3 denklem, 3 bilinmeyen:**
```
x₁ + 2x₂ + x₃ = b₁
3x₁ + x₂ + 2x₃ = b₂
2x₁ + 2x₂ + x₃ = b₃
```

**Durum:** Denklem sayısı = bilinmeyen sayısı → **Square**
- Genelde **tek bir çözüm** var (eğer $A$ full rank ise)
- Dengeli durum

#### Örnek 3: $r = 1.5$ (Underdetermined)

$n = 2$ örnek, $d = 3$ feature:

$$A = \begin{bmatrix} 1 & 2 & 1 \\ 3 & 1 & 2 \end{bmatrix}, \quad \mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix}, \quad \mathbf{b} = \begin{bmatrix} b_1 \\ b_2 \end{bmatrix}$$

**2 denklem, 3 bilinmeyen:**
```
x₁ + 2x₂ + x₃ = b₁
3x₁ + x₂ + 2x₃ = b₂
```

**Durum:** Denklem sayısı < bilinmeyen sayısı → **Underdetermined**
- **Sonsuz çözüm** var!
- SGD **minimum norm** çözümü buluyor ($\|\mathbf{x}\|$ en küçük olan)
- **Çok parametre** → Model karmaşık → Overfitting riski yüksek

### A4.3 Aspect Ratio'nun Etkileri

| $r$ | Durum | Çözüm | SGD Davranışı | Volterra Accuracy |
|-----|-------|-------|---------------|-------------------|
| $< 1$ | Overdetermined | Yaklaşık (unique) | Hızlı yakınsar, düşük variance | Çok iyi ($R > 0.99$) |
| $= 1$ | Square | Exact (unique) | Orta hız | İyi ($R \approx 0.98$) |
| $> 1$ | Underdetermined | Sonsuz (min-norm) | Yavaş, yüksek variance | Hala iyi ($R > 0.97$) |

---

## A5. Spectral Ratio Ne Anlama Geliyor?

### A5.1 Eigenvalue'lar Nedir?

$H = \frac{1}{n} A^\top A$ matrisinin **eigenvalue**'ları $\lambda_1, \lambda_2, \ldots, \lambda_d$:

$$H \mathbf{v}_i = \lambda_i \mathbf{v}_i$$

**İntuition:** Eigenvalue'lar verinin **her yöndeki yayılımını** (variance) ölçer.

### A5.2 Küçük Boyutlu Örnek

$A = \begin{bmatrix} 3 & 1 \\ 1 & 2 \\ 2 & 1 \end{bmatrix}$ için:

$$H = \frac{1}{3} A^\top A = \frac{1}{3} \begin{bmatrix} 3 & 1 \\ 1 & 2 \\ 2 & 1 \end{bmatrix}^\top \begin{bmatrix} 3 & 1 \\ 1 & 2 \\ 2 & 1 \end{bmatrix} = \frac{1}{3} \begin{bmatrix} 14 & 7 \\ 7 & 6 \end{bmatrix} = \begin{bmatrix} 4.67 & 2.33 \\ 2.33 & 2.00 \end{bmatrix}$$

Eigenvalue'ları hesaplayalım:
$$\det(H - \lambda I) = 0$$
$$(4.67 - \lambda)(2.00 - \lambda) - 2.33^2 = 0$$
$$\lambda^2 - 6.67\lambda + 3.91 = 0$$
$$\lambda_1 \approx 5.94, \quad \lambda_2 \approx 0.73$$

### A5.3 Spectral Ratio

$$\text{Spectral Ratio (SR)} = \frac{\lambda_{\max}}{\bar{\lambda}} = \frac{\lambda_{\max}}{\frac{1}{d}\sum_i \lambda_i}$$

Örneğimizde:
$$\bar{\lambda} = \frac{5.94 + 0.73}{2} = 3.33$$
$$\text{SR} = \frac{5.94}{3.33} = 1.78$$

### A5.4 Spectral Ratio'nun Görsel Anlamı

**Düşük SR (≈ 1):** Veri tüm yönlerde eşit yayılmış (izotropik)
```
        ↑
    · · · · ·
  · · · · · · ·
  · · · ● · · ·   ← Daire şeklinde
  · · · · · · ·
    · · · · ·
        ↓
```

**Yüksek SR (>> 1):** Veri bir yönde çok uzamış (anizotropik)
```
                    ↑
· · · · · · · · · · · · · · · · · · ·
            · · · ● · · ·             ← Elips şeklinde
· · · · · · · · · · · · · · · · · · ·
                    ↓
```

### A5.5 Spectral Ratio'nun SGD'ye Etkisi

| SR | Anlam | SGD Etkisi | Step Size |
|----|-------|------------|-----------|
| ≈ 1 | İzotropik | Her yönde eşit hızda öğrenir | Büyük $\gamma$ kullanılabilir |
| >> 1 | Anizotropik | Bazı yönlerde çok hızlı, bazılarında çok yavaş | Küçük $\gamma$ gerekli (yoksa patlar!) |

**MNIST'te problem:**
- SR ≈ 170-270 (çok yüksek!)
- Bir yönde $\lambda_{\max} \approx 60$, diğer yönlerde $\lambda \approx 0.01$
- Büyük eigenvalue'lı yönde çok hızlı hareket → **patlar**
- Küçük eigenvalue'lı yönde çok yavaş hareket → **yavaş öğrenir**

---

## A6. Critical Step Size Problemi

### A6.1 Teori Ne Diyor?

Makaledeki formül:
$$\gamma_{\max} = \frac{2}{r \cdot \bar{\lambda}}$$

Bu **ortalama** eigenvalue kullanıyor.

### A6.2 Gerçekte Ne Oluyor?

SGD'nin stabil olması için **en büyük** eigenvalue'a bakmalıyız:
$$\gamma_{\text{safe}} = \frac{2}{\lambda_{\max}}$$

### A6.3 Sayısal Örnek (MNIST)

| Metrik | Değer |
|--------|-------|
| $\lambda_{\max}$ | 60 |
| $\bar{\lambda}$ | 0.4 |
| $r$ | 1.0 |

**Teorik $\gamma_{\max}$:**
$$\gamma_{\max} = \frac{2}{1.0 \times 0.4} = 5.0$$

**Gerçek güvenli $\gamma$:**
$$\gamma_{\text{safe}} = \frac{2}{60} = 0.033$$

**Fark:** $5.0 / 0.033 \approx 150\times$ !!!

Teorik değeri kullansak SGD **anında patlar**.

### A6.4 Görsel Açıklama

```
Step Size Spectrum:
├─────────────────────────────────────────────────────────┤
0                                                         ∞

     γ_safe                              γ_theory
        ↓                                    ↓
├───────●────────────────────────────────────●────────────┤
        │                                    │
        │←──── STABIL ────→│←───── UNSTABIL (patlar) ─────→│
        │                  │
        └──────────────────┘
              ~150x fark!
```

---

# BÖLÜM B: TEORİK DETAYLAR

---

## B1. Matematiksel Framework

### B1.1 Problem Formülasyonu

$$\min_{\mathbf{x} \in \mathbb{R}^d} f(\mathbf{x}) = \frac{1}{2n} \|A\mathbf{x} - \mathbf{b}\|^2$$

**Gradient:**
$$\nabla f(\mathbf{x}) = \frac{1}{n} A^\top (A\mathbf{x} - \mathbf{b})$$

### B1.2 SGD Update (Detaylı)

Her iterasyonda:
1. Rastgele $i_k \sim \text{Uniform}\{1, \ldots, n\}$ seç
2. Sadece $i_k$'ıncı örneğin gradientini hesapla:
   $$\nabla f_{i_k}(\mathbf{x}) = A_{i_k}^\top (A_{i_k} \mathbf{x} - b_{i_k})$$
3. Güncelle:
   $$\mathbf{x}_{k+1} = \mathbf{x}_k - \frac{\gamma}{n} \nabla f_{i_k}(\mathbf{x}_k)$$

**Açılmış form (tek örnek için):**

$A_{i_k} = [a_{i_k,1}, a_{i_k,2}, \ldots, a_{i_k,d}]$ olsun.

$$\text{prediction} = \sum_{j=1}^{d} a_{i_k,j} x_j$$
$$\text{error} = \text{prediction} - b_{i_k}$$
$$\nabla f_{i_k} = \text{error} \cdot A_{i_k}^\top = \begin{bmatrix} a_{i_k,1} \cdot \text{error} \\ a_{i_k,2} \cdot \text{error} \\ \vdots \\ a_{i_k,d} \cdot \text{error} \end{bmatrix}$$

### B1.3 Neden $\gamma/n$?

Continuous-time limit için:
- Zaman: $t = k/n$ (normalize edilmiş iterasyon)
- $n \to \infty$ iken $t$ sabit kalıyor
- $\gamma/n$ scaling'i bu limitte anlamlı dinamik sağlıyor

---

## B2. Volterra Denklemi (Detaylı)

### B2.1 Ana Denklem

$$\psi_0(t) = z(t) + r\gamma^2 \int_0^t h_2(t-s) \psi_0(s) \, ds$$

### B2.2 $z(t)$ - Driving Term

$$z(t) = \frac{\sigma_b^2}{2} g_0(t) + \frac{\sigma_0^2}{2} g_2(t) - \rho \cdot g_1(t)$$

Burada:
- $\sigma_b^2 = \frac{1}{n}\|\mathbf{b}\|^2$: Target'ın normalize edilmiş enerjisi
- $\sigma_0^2 = \|\mathbf{x}_0\|^2$: Başlangıç noktasının enerjisi
- $\rho = \frac{1}{n}\langle \mathbf{b}, A\mathbf{x}_0 \rangle$: Target-başlangıç korelasyonu

**$g$ fonksiyonları:**
$$g_0(t) = \mathbb{E}_{\lambda}[e^{-2\gamma\lambda t}]$$
$$g_1(t) = \mathbb{E}_{\lambda}[\lambda e^{-2\gamma\lambda t}]$$
$$g_2(t) = \mathbb{E}_{\lambda}[\lambda^2 e^{-2\gamma\lambda t}]$$

### B2.3 $h_2(t)$ - Memory Kernel (Detaylı)

$$h_2(t) = \mathbb{E}_{\lambda \sim \mu}[\lambda^2 e^{-2\gamma\lambda t}]$$

**Discrete form (empirical eigenvalues ile):**
$$h_2(t) = \frac{1}{d} \sum_{i=1}^{d} \lambda_i^2 e^{-2\gamma\lambda_i t}$$

**Her terimin anlamı:**
- $\lambda_i^2$: Büyük eigenvalue'lar daha fazla ağırlık alır
- $e^{-2\gamma\lambda_i t}$: Zaman geçtikçe etki azalır
- Büyük $\gamma$ veya büyük $\lambda_i$ → daha hızlı decay

### B2.4 Integral Teriminin Anlamı

$$\int_0^t h_2(t-s) \psi_0(s) \, ds$$

Bu bir **convolution**:
- $t$ anındaki loss, geçmişteki **tüm** loss değerlerine bağlı
- Yakın geçmiş ($s \approx t$) → $h_2(t-s) \approx h_2(0)$ büyük → daha fazla etki
- Uzak geçmiş ($s \approx 0$) → $h_2(t-s) \approx h_2(t)$ küçük → az etki

**Fiziksel anlam:** SGD'nin noise'u "hafızalı". Önceki iterasyonlardaki hatalar şimdiki loss'u etkiler.

### B2.5 Numerical Çözüm

Discretize ediyoruz: $t_k = k \cdot \Delta t$, $k = 0, 1, \ldots, K$

$$\psi_0(t_{k+1}) = z(t_{k+1}) + r\gamma^2 \Delta t \sum_{j=0}^{k} h_2(t_{k+1} - t_j) \psi_0(t_j)$$

**Algoritma:**
```python
def solve_volterra(eigenvalues, gamma, r, T, dt):
    t = np.arange(0, T, dt)
    K = len(t)
    d = len(eigenvalues)

    # g fonksiyonlarını hesapla
    g0 = np.array([np.mean(np.exp(-2*gamma*eigenvalues*ti)) for ti in t])
    g1 = np.array([np.mean(eigenvalues * np.exp(-2*gamma*eigenvalues*ti)) for ti in t])
    g2 = np.array([np.mean(eigenvalues**2 * np.exp(-2*gamma*eigenvalues*ti)) for ti in t])

    # z(t) driving term
    z = (sigma_b**2 / 2) * g0 + (sigma_0**2 / 2) * g2 - rho * g1

    # Volterra iteration
    psi = np.zeros(K)
    psi[0] = z[0]

    for k in range(1, K):
        # h2 kernel at all past times
        h2_vals = np.mean(eigenvalues**2 * np.exp(-2*gamma*eigenvalues*(t[k] - t[:k])), axis=0)
        # Convolution integral
        integral = np.sum(h2_vals * psi[:k]) * dt
        # Update
        psi[k] = z[k] + r * gamma**2 * integral

    return t, psi
```

---

## B3. Veri Pipeline (Detaylı)

### B3.1 MNIST → Random Features

```
Adım 1: MNIST Yükle
────────────────────
X ∈ ℝⁿˣ⁷⁸⁴  (n=3000 görüntü, her biri 28×28=784 piksel)

Örnek satır: X[0] = [0.0, 0.0, 0.12, 0.89, ..., 0.0]  (784 değer)


Adım 2: Random Projection
────────────────────
W ∈ ℝ⁷⁸⁴ˣᵈ  (d = r×n feature)
W[i,j] ~ N(0, 1/784)  (her eleman bağımsız Gaussian)

Z = X @ W  ∈ ℝⁿˣᵈ

Örnek: Z[0,0] = Σⱼ X[0,j] × W[j,0]  (784 terimin toplamı)


Adım 3: Nonlinear Activation
────────────────────
A = σ(Z) = ReLU(Z + 1)  (shifted ReLU)

Neden +1?
- Normal ReLU: çok fazla sıfır çıktı
- Shifted: daha az sıfır, daha iyi conditioned matris
```

### B3.2 Whitening İşlemi

```
Adım 1: Covariance Hesapla
────────────────────
Σ = (1/n) × Aᵀ @ A  ∈ ℝᵈˣᵈ

Σ[i,j] = (1/n) × Σₖ A[k,i] × A[k,j]  (i ve j. feature'ların korelasyonu)


Adım 2: Eigendecomposition
────────────────────
Σ = V @ Λ @ Vᵀ

V: eigenvector matrisi (orthogonal)
Λ: eigenvalue diagonal matrisi


Adım 3: Whitening Transform
────────────────────
Σ⁻¹/² = V @ Λ⁻¹/² @ Vᵀ

Ã = A @ Σ⁻¹/²

Sonuç: Ãᵀ @ Ã / n = I  (identity!)
```

**Geometrik anlam:**
- Orijinal veri: elips şeklinde dağılmış
- Whitened veri: küre şeklinde dağılmış (izotropik)

---

## B4. 5 Modelin Detaylı Karşılaştırması

### B4.1 Empirical SGD

**Ne yapıyor:** Gerçek SGD algoritmasını çalıştırıyor.

```python
for seed in range(50):
    x = random_init()
    for k in range(num_iters):
        i = random_sample()
        x = x - (gamma/n) * gradient_i(x)
        loss[k] += compute_loss(x)
loss /= 50  # ortalama
```

**Özellikler:**
- Stokastik (her run farklı)
- Ground truth
- Hesaplama maliyeti yüksek

### B4.2 Volterra

**Ne yapıyor:** Integral denklemi çözüyor.

**Özellikler:**
- Deterministik
- Eigenvalue dağılımına bağlı
- Temporal correlation'ı yakalıyor
- **En iyi tahmin**

### B4.3 Streaming SGD

**Ne yapıyor:** Her iterasyonda **yeni** örnek varsayıyor.

$$\mathbf{x}_{k+1} = \mathbf{x}_k - \gamma \cdot a_k (a_k^\top \mathbf{x}_k - b_k)$$

$(a_k, b_k)$ her seferinde fresh, i.i.d. sample.

**Özellikler:**
- Variance = 0 (çünkü örnekler bağımsız)
- Loss **sürekli düşer** (hiç artmaz)
- Gerçekçi değil (sonsuz veri varsayımı)
- Lower bound olarak faydalı

### B4.4 SDE (Stochastic Differential Equation)

**Ne yapıyor:** SGD'yi continuous-time SDE olarak modeller.

$$d\mathbf{x}_t = -\nabla f(\mathbf{x}_t) dt + \sqrt{\gamma \Sigma} \, d\mathbf{W}_t$$

**Problem:** Noise'u **isotropic** varsayıyor (tüm yönlerde eşit).

**Gerçekte:** Noise yönü gradient yönüne bağlı!

**Sonuç:** Loss'u **underestimate** eder.

### B4.5 SME (Stochastic Modified Equation)

**Ne yapıyor:** SDE + ikinci derece düzeltme terimleri.

**Özellikler:**
- SDE'den daha iyi
- Ama Volterra kadar iyi değil
- Hala bazı korelasyonları kaçırıyor

### B4.6 Karşılaştırma Tablosu

| Model | Noise | Correlation | Accuracy |
|-------|-------|-------------|----------|
| Empirical SGD | Gerçek | Gerçek | Ground truth |
| **Volterra** | Eigenvalue-based | **Temporal** | **En iyi** |
| Streaming | Yok | Yok | Lower bound |
| SDE | İsotropik | Yok | Zayıf |
| SME | İsotropik + düzeltme | Kısmi | Orta |

---

# BÖLÜM C: DENEYSEL SONUÇLAR

---

## C1. Replication (Synthetic Data)

**Amaç:** Makalenin Figure 4'ünü replicate etmek.

**Setup:**
- $A_{ij} \sim \mathcal{N}(0, 1/d)$ (isotropic Gaussian)
- $n = 1000$, $d = 500$ ($r = 0.5$)
- $\gamma = \gamma_{\max} / 8$

**Sonuç:** Volterra, SGD ile neredeyse **birebir** örtüşüyor.

## C2. MNIST Base Setting

**Setup:**
- MNIST random features
- Linear target: $\mathbf{b} = A\mathbf{x}^*$
- $r \in \{0.5, 1.0, 1.5\}$

**Sonuçlar:**

| $r$ | Correlation $R$ | Spectral Ratio |
|-----|-----------------|----------------|
| 0.5 | 0.991 | 152 |
| 1.0 | 0.984 | 178 |
| 1.5 | 0.978 | 267 |

**Yorum:** Yüksek spectral ratio'ya rağmen Volterra hala çok iyi çalışıyor!

## C3. MNIST Nonlinear Setting

**Setup:**
- $\mathbf{b} = \text{ReLU}(A\mathbf{x}^*)$
- ~%50 değer sıfıra clip'lendi

**Sonuçlar:**

| $r$ | Correlation $R$ | Zero Fraction |
|-----|-----------------|---------------|
| 0.5 | 0.982 | 48% |
| 1.0 | 0.975 | 51% |
| 1.5 | 0.971 | 53% |

**Yorum:** Non-realizable durumda bile $R > 0.97$!

**Loss floor:** SGD bir noktada düşmeyi durduruyor (irreducible error).

## C4. MNIST Whitened Setting

**Setup:**
- $\tilde{A} = A\Sigma^{-1/2}$
- Spectral ratio düşürülmüş

**Sonuçlar:**

| $r$ | Correlation $R$ | Original SR | Whitened SR |
|-----|-----------------|-------------|-------------|
| 0.5 | 0.9993 | 152 | 36 |
| 1.0 | 0.9991 | 178 | 52 |
| 1.5 | 0.9988 | 267 | 71 |

**Yorum:** Whitening spectral ratio'yu ~4x düşürüyor, correlation neredeyse mükemmel!

## C5. Step-Size Criticality

**Deney:** $\gamma$'yı artırarak instability noktasını bulma.

**Sonuçlar:**

| Multiplier | $\gamma / \gamma_{\text{safe}}$ | Stable? |
|------------|--------------------------------|---------|
| 0.001× | 0.001 | ✓ Stable |
| 0.01× | 0.01 | ✓ Stable |
| 0.1× | 0.1 | ✓ Stable |
| 1.0× | 1.0 | ✓ Stable (barely) |
| 2.0× | 2.0 | ✗ Unstable |
| 10× | 10.0 | ✗ Explodes |

**Kritik bulgu:** $\gamma_{\text{safe}} = 2/\lambda_{\max}$ gerçekten kritik eşik!

---

# BÖLÜM D: ÖNEMLİ KAVRAMLAR SÖZLÜĞÜ

---

| Terim | Tanım |
|-------|-------|
| **SGD** | Stochastic Gradient Descent - her iterasyonda rastgele bir örnek kullanarak gradient hesaplama |
| **Loss** | Tahmin hatası - $\frac{1}{2n}\|A\mathbf{x} - \mathbf{b}\|^2$ |
| **Aspect Ratio ($r$)** | Feature/örnek oranı - $d/n$ |
| **Spectral Ratio** | En büyük / ortalama eigenvalue - $\lambda_{\max}/\bar{\lambda}$ |
| **Eigenvalue** | $H\mathbf{v} = \lambda\mathbf{v}$ denklemini sağlayan $\lambda$ değerleri |
| **Volterra Equation** | $\psi_0(t) = z(t) + r\gamma^2 \int_0^t h_2(t-s)\psi_0(s)ds$ |
| **Realizable** | Mükemmel çözümün var olduğu durum ($\mathbf{b} \in \text{colspan}(A)$) |
| **Non-realizable** | Mükemmel çözümün olmadığı durum |
| **Whitening** | Veriyi izotropik hale getirme - $\tilde{A} = A\Sigma^{-1/2}$ |
| **Random Features** | Kernel yaklaşımı - $A = \sigma(XW)$ |
| **Critical Step Size** | SGD'nin stabil kaldığı maksimum $\gamma$ |
| **Loss Floor** | Non-realizable durumda minimum ulaşılabilir loss |
| **Memory Kernel ($h_2$)** | Geçmiş loss değerlerinin şimdiye etkisini belirleyen fonksiyon |
| **Driving Term ($z$)** | Volterra denklemindeki "input" - başlangıç koşullarından gelen terim |

---

# BÖLÜM E: ÖZET VE SONUÇLAR

---

## E1. Ana Bulgular

1. **Volterra teorisi gerçek veride çalışıyor**
   - MNIST random features üzerinde $R > 0.97$
   - Teori Gaussian varsaysa da robust

2. **Spectral ratio kritik**
   - MNIST: SR ≈ 170 (çok yüksek)
   - Teori: SR ≈ 1 varsayar
   - Step size seçimini etkiliyor

3. **Whitening işe yarıyor**
   - SR'yi ~4x düşürüyor
   - $R \approx 0.999$ korelasyon

4. **Teorik $\gamma_{\max}$ güvenilmez**
   - Gerçek güvenli değerden ~100x büyük
   - $\gamma_{\text{safe}} = 2/\lambda_{\max}$ kullan

## E2. Pratik Öneriler

| Durum | Öneri |
|-------|-------|
| Step size seçimi | $\gamma = 0.5 \times 2/\lambda_{\max}$ |
| Yüksek SR | Whitening uygula |
| Non-realizable | Volterra hala faydalı (trend doğru) |
| Büyük $r$ | Daha küçük $\gamma$ kullan |

---

*Son güncelleme: Ocak 2026*
