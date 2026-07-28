# DijitalKoku — Detaylı Teknik Analiz Dökümanı
**Versiyon:** 3.0 | **Mimari:** Semantik-Olasılıksal Çok Katmanlı Sistem

---

## 1. PROJE GENEL BAKIŞ

DijitalKoku, gerçek zamanlı video/ekran görüntüsü analizinden 7 farklı koku sinyali üretip Arduino tabanlı donanıma PWM komutu gönderen bir sistemdir. 7 koku kanalı: **Çam, Deniz, Vanilya, İs/Duman, Nane, Toprak, Limon**.

### Sistem Akışı (Üst Seviye)

```
Ekran Yakalama (mss)
        ↓
  CLAHE Normalizasyonu
        ↓
┌───────────────────────────────────────────┐
│              KATMAN A — Sinyal Kaynakları  │
│  YOLOv8 Nano │ K-Means │ Laplacian │ CLIP │
└───────────────────┬───────────────────────┘
                    ↓
┌───────────────────────────────────────────┐
│         KATMAN B — VLM Semantik Kilit      │
│    moondream2 / Ollama → Semantik Maske    │
└───────────────────┬───────────────────────┘
                    ↓
┌───────────────────────────────────────────┐
│      KATMAN C — BayesFuser (Log-Uzayı)     │
│    P(s|data) ∝ Prior × ∏ P(data_i|s)^w_i  │
└───────────────────┬───────────────────────┘
                    ↓
┌───────────────────────────────────────────┐
│      KATMAN D — KalmanFilterBank           │
│       7 bağımsız 1D Kalman filtresi        │
└───────────────────┬───────────────────────┘
                    ↓
       Post-İşlem (Histerezis, Oylama,
       Baskınlık Hiyerarşisi, Doygunluk)
                    ↓
          PWM Seri Komut → Donanım
```

---

## 2. KATMAN A — SİNYAL KAYNAKLARI

### 2.1 YOLOv8 Nano — Nesne Tespiti

**Dosya:** `analyzer.py` → `run_yolo()` + `pdm_engine.py` → `_yolo_module_scores()`

**Yükleme:** CUDA float16 (GPU varsa), CPU fallback. Geç başlatma (`_get_yolo()`).

**Çalışma Adımları:**

1. Frame 640 px'e ölçeklenir: `scale = 640 / max(h, w)`
2. YOLO çalıştırılır, dahili eşik `min(conf_threshold, 0.25)`
3. Koordinatlar orijinal boyuta geri dönüştürülür: `xyxy_orig = box.xyxy[0] / scale`

**Etiket Bazlı Güven Eşikleri:**
```
donut/cake    → conf >= 0.65   (sahte tespit riski yüksek)
pizza/hot dog → conf >= 0.60
fork/knife    → conf >= 0.60
cow/horse/... → conf >= max(0.25, global - 0.12)  (büyük hayvanlar daha esnek)
```

**Alan Kısıtı:** `donut/cake` için `area_ratio > 0.55/0.60` → filtrele (dev nesne = sahte).

**YOLO → Koku Skoru Formülü:**
```
area_w  = min(1.0, area_ratio × 8.0)
penalty = 0.20 if (conf < 0.55 AND area < 0.05) else 1.0
scores[scent] += conf × w_scent × area_w × penalty
```

Tüm kanallar L1 normalize edilir: `scores[s] = scores[s] / sum(scores.values())`

**Ağaç/Bitki Bonusu (Çam):**
```
boost = min(0.25, tree_area × 1.5)
yolo_s["Çam"] += boost
yolo_s["Toprak"]   -= boost / 2
yolo_s["İs/Duman"] -= boost / 2
```

**Temporal Memory (FPS-Bağımsız Hafıza):**

Nesne kaybolduğunda tamamen sıfırlanmaz; logaritmik yarılanma ömrü uygulanır:
```
weight(t) = conf × 2^(−age / half_life)
```
`half_life = 2.0 s`, `max_objects = 30`. Hafıza katkısı: `yolo_s[scent] += mem_w × w × 0.45`

---

### 2.2 K-Means Renk Analizi

**Dosya:** `analyzer.py` → `run_kmeans()`, `pdm_engine.py` → `_kmeans_module_scores()`

**Çerçeve Kırpma:** Frame'in merkezi %60'ı alınır (kenar UI elementleri sahte sinyal üretmesin):
```
center = frame[h×0.20 : h×0.80, w×0.20 : w×0.80]
small  = resize(center, (160, 90))
```

**GPU K-Means (Lloyd Algoritması):**

CUDA mevcutsa PyTorch tabanlı GPU K-Means, yoksa sklearn fallback.

```python
# Merkez başlatma: rastgele k piksel
centroids = X[randperm(N)[:k]]
for _ in range(n_iter=50):
    dists  = cdist(X, centroids)       # (N, k)
    labels = argmin(dists, dim=1)      # (N,)
    new_c  = mean(X[labels == i])      # her küme için
    if allclose(centroids, new_c, atol=1.0): break
```

**Warm Start (cv2.kmeans):** Önceki kare merkezleri başlangıç noktası olarak kullanılır, tek geçiş (1 iterasyon) yeterli olur → ~3-5x hızlanma.

**Renk → Koku Eşleme (HSV Kuralları):**

`_COLOR_RULES` tablosu: `(h_min, h_max, s_min, v_min)` → `{scent: weight}`

```
Nötr filtre: S < 38 veya V < 38 → atla (siyah/beyaz/gri koku tetiklemesin)
```

Sıralama ağırlığı: `rank_w = 1 / (rank + 1)` (en baskın renk en çok katkı yapar)

Kritik kurallar:
| H Aralığı | S_min | Koku |
|---|---|---|
| 0-25 | 150 | İs/Duman 0.85, Vanilya 0.15 |
| 55-95 | 45 | Çam 0.65, Toprak 0.20, Nane 0.15 |
| 170-230 | 102 | Deniz 0.80, Nane 0.20 |
| 270-330 | 100 | Vanilya 0.70, Limon 0.30 |

**K-Means Tamer (Dinamik Ağırlıklandırma):**

Matematiksel temel (log-uzayı eşdeğeri):
```
w_dyn × log(val) = w_base × log(val^(w_dyn/w_base))
⇒ kmeans_tamed[s] = max(pad, val)^(w_dyn/w_base)
```

Dört mod:
- **Normal:** `w_dyn = w_base = 0.15`
- **Aksiyon** (CLIP İs/Duman > 0.30): İs/Duman hariç tüm `w_dyn = 0.01` (sarı/turuncu piksel yanlış eşleme engeli)
- **Kentsel** (VLM urban kapıları aktif): `Vanilya: 0.02, Nane: 0.04`
- **Kurtarıcı** (VLM kör + CLIP doğa görüyor): `Çam/Toprak/Deniz: 0.45`

`w_dyn < 0.015` → `tamed[s] = 1e-9` (log-uzayında gerçek sıfır maskesi, `log(1e-9) ≈ -20`)

---

### 2.3 Laplacian Doku Analizi

**Dosya:** `pdm_engine.py` → `_texture_module_scores()`

```
lap_var  = Var(Laplacian(gray))
richness = min(1.0, lap_var / 800.0)    # 0 = düz alan, 1 = zengin doku
```

HSV istatistiklerinden koku çıkarımı:

| Koşul | Koku | Formül |
|---|---|---|
| H∈[80,160], richness > 0.35 | Çam | `richness × 0.75` |
| H∈[170,260], S > 60, richness < 0.4 | Deniz | `(1-richness) × 0.70` |
| V < 60, S < 55 | İs/Duman | `(1-richness) × 0.60` |
| H∈[15,45], richness > 0.45 | Toprak | `richness × 0.65` |
| H∈[35,62], S > 130, richness < 0.35 | Limon | `0.55` |
| V > 180, S < 60, richness < 0.30 | Vanilya | `0.50` |

**Doku Tabanlı Sucul Filtresi:** `lap_var > 600` (orman/dağ/kaya) + aquatik YOLO nesnesi yok → `Deniz × 0.10`

---

### 2.4 CLIP Zero-Shot (OVL Modülü)

**Dosya:** `pdm_engine.py` → `_init_ovl()`, `_analyze_ovl()`

**Model:** `openai/clip-vit-base-patch32`, CUDA float32 + `autocast(float16)`

**Absorber Mimarisi:**

36 toplam prompt: 7 koku × 3 prompt + 3 absorber × 3 prompt + 6 doğrulama.

```
[0-2]   Çam       [3-5]   Deniz     [6-8]   Vanilya
[9-11]  İs/Duman  [12-14] Nane      [15-17] Toprak   [18-20] Limon
[21-23] ABSORBER: Gökyüzü/Bulut
[24-26] ABSORBER: Kaya/Taş
[27-29] ABSORBER: Bina/Şehir
[30-31] Su doğrulama  [32-33] Mekan  [34-35] Karasal/Sucul
```

Absorber fikri: Gökyüzü sahnesi oluştuğunda softmax kütlesini absorber yutar → Deniz/İs/Duman doğal olarak sıfıra yaklaşır (renormalizasyon yapılmaz, aksi halde absorber cezası geri verilmiş olur).

**Tek Forward Pass — 10-Sınıf Softmax:**

```
all_groups  = [logits[i×3 : i×3+3] for i in range(10)]   # 10 = 7 koku + 3 absorber
all_means   = stack([g.mean() for g in all_groups])
all_stds    = stack([g.std()  for g in all_groups])
coherence   = 1 / (1 + 2×stds)                            # yüksek std → düşük güven
all_weighted = all_means × coherence
all_probs   = softmax(all_weighted / T)                    # T = temperature = 0.7
```

**Doğrulama Prob'ları:**
```
water_probs   = softmax(logits[30:32] / T)    → {"water": p0, "dry": p1}
outdoor_probs = softmax(logits[32:34] / T)    → {"outdoor": p0, "indoor": p1}
terr_aq_probs = softmax(logits[34:36] / T)    → {"terrestrial": p0, "aquatic": p1}
```

**Asenkron Çalışma:** Her `analysis_interval=15` karede bir arka plan thread'inde çalışır. `_ovl_run_lock` (non-blocking acquire) ile iki thread aynı anda başlamaz. VRAM temizliği `_ovl_lock` altında yapılır.

**Warmup Lock:** CLIP ilk geçerli inference üretene kadar tüm PWM çıkışı sıfırlanır. İlk sonuç gelince `_clip_ever_run = True` set edilir.

---

### 2.5 Optik Akış

**Dosya:** `pdm_engine.py` → `_optical_flow_scores()`

`cv2.calcOpticalFlowFarneback(prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)` ile 160×90 küçültülmüş gri kareler üzerinde hesaplanır.

| Bölge | Koşul | Sinyal |
|---|---|---|
| Mavi bölge (H∈[170,260], S>60) | mag >= 0.5 AND fx > fy (yatay) | `Deniz += min(0.25, b_mag×0.12)` |
| Mavi bölge | mag < 0.5 (statik) | `Deniz -= 0.15` |
| Gri bölge (S<55) | fy < -0.30 (yukarı hareket) | `İs/Duman += min(0.20, |fy|×0.15)` |

---

## 3. KATMAN B — VLM SEMANTİK KİLİT

**Dosya:** `context_analyzer.py`

### 3.1 ContextAnalyzerThread

Daemon thread olarak çalışır. Her `interval_frames=20` karede bir kuyruğa kare ekler. Kuyruk `maxsize=1` — eski kare atılır, her zaman en güncel kare analiz edilir.

**Frame Encode:** `BGR → resize(512,512) → RGB → JPEG(q=80) → base64`

**Ollama API Çağrısı:**
```
POST http://localhost:11434/api/generate
{
  "model":   "moondream",
  "prompt":  CONTEXT_PROMPT,
  "images":  [base64_frame],
  "stream":  false,
  "options": {"temperature": 0.1, "num_predict": 256}
}
```

**Thread Güvenliği:** `RLock` + GIL-atomik snapshot `_snapshot: tuple = (mask, is_void, strict_ok)`. Ana döngü `blocking=False` acquire ile VLM thread'ini asla beklemez.

**Tazelik Bypass:** `_VLM_FRESHNESS_TIMEOUT = 3.0 s`. VLM son analizden 3 saniye geçtiyse maske `1.0` (tam açık) olarak döner — eski VLM kararı Katman A'yı bloke etmez.

**Stale Timeout:** `fallback_timeout=10 s` geçince `DEFAULT_MASK` (tümü 1.0) + `vlm_status="stale"`.

---

### 3.2 compute_semantic_mask()

VLM JSON çıktısını 7 kanalın çarpanına dönüştürür.

**Sabitler:**
```
_HARD_VETO        = 1e-7     → log(1e-7) ≈ -16.1  (Bayes'te kanal pratikte imkansız)
_SOFT_SUPPRESSION = 0.02     → log(0.02) ≈ -3.9   (sinyal zayıflıyor ama geçebilir)
```

**Karar Ağacı:**

```
is_abstract_or_void = True
    → TÜM KANALLAR = _HARD_VETO (soyut/grafik/rüya sahne)

strict_scent_allowed = False  (VLM koku kaynağı bulamadı)
    → TÜM KANALLAR = _SOFT_SUPPRESSION
    → Katman A %80+ sinyal üretirse aşılabilir (Bayesçi esneklik)

strict_scent_allowed = True  →  kanal bazlı kurallar:
    has_veg = False  →  Çam = HARD_VETO,  Nane × 0.15,  Toprak × 0.20
    env = urban_historic/urban_modern:
        has_water = False  →  Deniz = HARD_VETO
        has_veg   = False  →  Çam = HARD_VETO, Toprak × 0.10, Nane × 0.10
        has_veg   = True   →  Çam × 0.10, Toprak × 0.15, Nane × 0.40, Deniz × 0.10
        → Vanilya = min(Vanilya, 0.01),  Nane = min(Nane, 0.05)
    has_fire = True   →  İs/Duman = min(İs/Duman × 2.0, 1.0)
    env = nature_water →  İs/Duman = HARD_VETO
    has_water = True  →  Deniz = min(Deniz × 1.5, 1.0)
    env = indoor_*    →  Çam × 0.45, Deniz × 0.40, Toprak × 0.50
```

---

## 4. KATMAN C — BAYES LOG-UZAYI FÜZYON

**Dosya:** `bayes_fuser.py` → `BayesFuser`

### 4.1 Matematiksel Model

```
P(s | data) ∝ P(s) × ∏_i P(data_i | s)^w_i × mask[s]
```

Log-uzayında (sayısal kararlılık için):
```
log_post(s) = log(prior(s)) + Σ_i w_i × log(P_i(s)) + log(mask[s])
```

**Log-Sum-Exp Normalizasyonu** (taşma ve underflow önleme):
```
max_lp      = max(log_posteriors.values())
exp_shifted = {s: exp(log_post(s) - max_lp) for s in SCENTS}
total       = sum(exp_shifted.values())
posterior   = {s: exp_shifted[s] / total for s in SCENTS}
```

**Sıfır Koruması:** `val = max(float(pdf.get(scent, 1e-7)), 1e-7)` — sıfır PDF logunun -∞ olmasını engeller.

### 4.2 Modül Ağırlıkları

| Modül | Ağırlık |
|---|---|
| CLIP | 0.40 |
| YOLO | 0.30 |
| K-Means | 0.15 |
| Texture | 0.10 |
| vlm_bias | 0.05 |

### 4.3 Prior Güncelleme

Her karenin posterior'u bir sonraki karenin prior'u olur:
```python
self._prior = dict(normalized)
```
Başlangıç değeri: `prior(s) = 1/7 ≈ 0.1429` (düz / bilgisiz prior)

Sahne kesiminde: `reset_prior()` → tüm kanallar `1/7`'ye döner.

### 4.4 VLM Master Prior

VLM yüksek güvenle (`vlm_conf >= 0.70`) sahne modunu belirlediğinde, `fuse()` çağrısından önce yasaklanan kanallar prior'da log-uzayında baskılanır:
```python
_rp = {s: (1e-9 if s in suppress_set else 1.0) for s in SCENTS}
_vlm_master_prior = {s: v / sum(_rp.values()) for s, v in _rp.items()}
```

---

## 5. KATMAN D — KALMAN FİLTRE BANKASI

**Dosya:** `kalman.py` → `KalmanFilter1D`, `KalmanFilterBank`

### 5.1 1D Kalman Filtresi

Durum: koku yoğunluğu `x ∈ [0, 1]`

**Tahmin Adımı:**
```
x_pred = x
P_pred = P + Q
```

**Kalman Kazancı:**
```
K = P_pred / (P_pred + R)
```

**Güncelleme Adımı:**
```
x = x_pred + K × (measurement - x_pred)
P = (1 - K) × P_pred
```

**Parametre Mantığı:**

| Kanal | Q | R | Yorum |
|---|---|---|---|
| İs/Duman | 0.025 | 0.06 | Hızlı tepki (keskin koku) |
| Deniz | 0.015 | 0.08 | Orta hız |
| Çam | 0.008 | 0.12 | Yavaş (orman sessiz değişir) |
| Nane | 0.008 | 0.15 | En pürüzsüz çıkış |
| Vanilya | 0.010 | 0.10 | Standart |
| Toprak | 0.010 | 0.12 | Standart |
| Limon | 0.012 | 0.10 | Standart |

**VLM-Driven Adaptive R-Scaling:** Aksiyon/Kurtarıcı+Emniyet modunda `R = R_base × 10.0` → Kalman anlık K-Means gürültüsüne yavaş tepki verir.

**EMA'nın Yerini Alması:** Eski sistem `_smooth_ema()` ile sabit alpha kullanıyordu. Kalman'da Q/R ayrımı fiziksel anlam taşır; config'den per-kanal ayarlanabilir; sahne kesiminde `x=0, P=1.0` ile sıfırlanır.

---

## 6. POST-İŞLEM ZİNCİRİ

### 6.1 CLAHE — Işık Normalizasyonu

```
LAB = BGR→LAB
L_eq = CLAHE(L, clipLimit=2.0, tileGridSize=(8,8))
output = LAB(L_eq, A, B) → BGR
```
Düşük ışık/yüksek kontrast koşullarında HSV analizini stabilize eder.

### 6.2 Aydınlatma Profili

```
GOLDEN_HOUR: warm_mask = (H∈[5°,35°]) AND (S > 80),  warm_ratio > 0.25
NIGHT_DARK:  mean(V) < 60
DAYLIGHT:    diğer
```

Sıralama: GOLDEN_HOUR > NIGHT_DARK (sıcak renk varsa karanlık bile GOLDEN_HOUR).

### 6.3 Makro Bağlam Sınıflandırma

YOLO etiketlerinden `INDOOR / NATURE / WATER / KITCHEN / CITY / UNKNOWN`. Eşitlikte öncelik: `INDOOR > KITCHEN > NATURE > WATER > CITY`.

Bağlam çarpanları örnek:
```
INDOOR:  Çam × 0.45,  Deniz × 0.40,  Vanilya × 1.25
NATURE:  Çam × 1.35,  Toprak × 1.25, Vanilya × 0.55
WATER:   Deniz × 1.45, İs/Duman × 0.45
CITY:    İs/Duman × 1.25, Çam × 0.65
```

### 6.4 Vektörel Bağlam Doğrulaması (VCV)

**Koku Ekseni:** Her koku için `(p_nature, p_urban)` çifti:
```
Çam:      (0.95, 0.05)   Deniz:    (0.88, 0.12)
Nane:     (0.60, 0.40)   Toprak:   (0.88, 0.12)
Limon:    (0.25, 0.75)   Vanilya:  (0.10, 0.90)
İs/Duman: (0.22, 0.78)
```

**Bağlam Ekseni Hesaplama:**
```
urban_raw  = (1-w) × yolo_urban  + w × (0.05×yolo_urban  + 0.95×clip_urban)
nature_raw = (1-w) × yolo_nature + w × (0.05×yolo_nature + 0.95×clip_nature)
w = min(1.0, ovl_conf × 1.5)    (CLIP güveni ağırlık belirler)
```

**Üstel Sönümleme Formülü:**
```
alignment   = p_nature × nature_signal + p_urban × urban_signal
conflict    = 1.0 − alignment
decay       = exp(−λ × conflict)          λ = 2.5 (varsayılan)
score_final = score × decay
```

Örnekler (λ=2.5):
- Kentsel sahne (u=0.8) + Çam (p_nat=0.95): `alignment=0.19`, `conflict=0.81`, `decay≈0.13` → güçlü ceza
- Kentsel sahne (u=0.8) + Vanilya (p_urb=0.90): `alignment=0.74`, `conflict=0.26`, `decay≈0.52` → hafif ceza
- Doğa sahnesi (n=0.9) + Vanilya (p_urb=0.90): `alignment=0.09+0.09=0.18`, `conflict=0.82`, `decay≈0.13` → güçlü ceza

**Kentsel Sertleştirme:**
```
urban_signal > 0.60 AND p_nat > 0.50:
    effective_λ = λ × (1 + (urban_signal - 0.60) × 5.0)
```

**Limon Kuzey/Karasal Sertleştirme:**
```
karasal = Çam + Toprak + İs/Duman
karasal > 0.45: effective_λ_limon = λ × (1 + (karasal - 0.45) × 2.0)
```

**VCV Karasal Güvence:** K-Means baskın rengi yeşilse CLIP hatalı aquatic ürettiyse düzelt:
```
terrestrial = max(terrestrial, 0.65)
aquatic     = min(aquatic, 0.20)
```

### 6.5 Histerezis Kapısı

```
raw >= 0.20  →  state = True   (aç)
raw <  0.08  →  state = False  (kapa)
0.08 ≤ raw < 0.20  →  önceki durum korunur (ölü bant)

Pasif kanal:  score × passive_decay_factor (= 0.45)
Aktif kanal:  score korunur
```

STATE kararları Bayes posterior snapshot (`blended_4way`) üzerinden alınır; çıktı değerleri post-VCV blended üzerinden uygulanır → latch riski önlenir.

### 6.6 N-Kare Oylama (Temporal Consensus)

`voting_window = 5` kare tamponu:
```python
vote_ratio = sum(buf) / len(buf)   # eşik üstünde kaç kare?
if vote_ratio >= 0.30:
    result[s] = score               # güçlü çoğunluk → tam sinyal
else:
    result[s] = score × vote_ratio  # zayıf çoğunluk → orantılı baskı
```

### 6.7 Cognitive Soft-Gate

Final Tamer Guard'dan sonra VETO kanallarının VLM güvenine göre yumuşak kapısı:
```
Final = vlm_conf × VLM_Masked + (1 - vlm_conf) × Bayes_Output
```
- `vlm_conf = 1.0` → tam VETO (VLM hâkim)
- `vlm_conf = 0.0` → saf Bayes çıkışı (istatistiksel füzyon hâkim)

### 6.8 Baskınlık Hiyerarşisi

```
dominant_power = SCENT_DOMINANCE[dominant_scent]
power_diff     = dominant_power - SCENT_DOMINANCE[scent]
suppression    = min(0.45, power_diff × dominant_score × 0.50)
result[scent] *= (1 - suppression)
```

Güç sıralaması: `İs/Duman(1.0) > Nane(0.80) > Çam(0.60) > Deniz(0.50) > Toprak(0.40) > Limon(0.35) > Vanilya(0.25)`

**Doğa Koalisyonu:** Baskın koku doğa kokusuysa (`Çam/Toprak/Nane`) diğer doğa kokuları `protected_min = original × 0.60`'ın altına inemez.

### 6.9 Doygunluk Filtresi

```
window = 60 sn, high_threshold = 0.70, fade_rate = 0.008
sat_ratio = (score > 0.70 frame sayısı) / toplam frame

sat_ratio > 0.80: attenuation = max(0.40, att - 0.008)  (yavaş azal)
sat_ratio ≤ 0.80: attenuation = min(1.0,  att + 0.004)  (yarım hızda topar)
```
UI skorları tam gösterilir; `attenuation` yalnızca donanım PWM çıkışına çarpılır.

### 6.10 Purge Controller

Sahne kesimi veya baskın skor atlaması (`|dom_t - dom_{t-1}| >= 0.50`):
```
purge_speed = 1.0  →  1.5 saniye boyunca ch7 = pwm_max
```
8. seri kanal olarak iletilir.

---

## 7. SAHNE KESME ALGILAMA

**Dosya:** `analyzer.py` → `Analyzer.analyze()`

160×90 gri histogramlar karşılaştırılır (`cv2.HISTCMP_CORREL`):

```
is_scene_cut = (
    similarity < 0.25                           (hard cut)
    OR (similarity < 0.35 AND motion > 0.35)    (mixed cut)
    OR (motion > 0.35    AND similarity < 0.18) (motion cut)
)
```

Sahne kesiminde:
1. `BayesFuser.reset_prior()` → düz 1/7 dağılım
2. `KalmanFilterBank.reset()` → `x=0, P=1.0`
3. Histerezis + oylama tamponu sıfırlanır
4. `ScentController.scene_cut_purge()` → 1.5 sn ch7 maksimum

**Hareket Skoru:**
```
diff   = absdiff(prev_gray, curr_gray)
thresh = threshold(diff, motion_diff_threshold=25, 255)
motion_score = sum(thresh) / (thresh.size × 255.0)
```

---

## 8. DONANIM KONTROL PROTOKOLܜ

**Dosya:** `scent_controller.py` → `ScentController`

**Seri Protokol:**
```
Format: !ch0,ch1,ch2,ch3,ch4,ch5,ch6,ch7|checksum#
ch0-ch6 = 7 koku kanalı (PWM 0-4095)
ch7     = tahliye fanı (purge)
checksum = sum(ch0..ch7) % 256
```

**PWM Değer Hesabı:**
```
pwm_val = int(score × attenuation × pwm_max × intensity)
pwm_max = 4095  (PCA9685 12-bit)
```

**Bayat Komut Kontrolü:** `(now - cmd.timestamp) × 1000 > 300 ms` → komut atlanır.

**Minimum Değişim Eşiği:** `|score_new - score_old| < 0.05` → seri yazılmaz.

**Back-flush:** Tüm kanallar 0, 200 ms bekleme (pompa geri emiş).

**Purge:** Tüm kanallar 0 + ch7 = 4095, 500 ms bekleme → ch7 = 0.

---

## 9. NESNE → KOKU EŞLEMELERİ

**Dosya:** `engine.py` → `OBJECT_SCENT_MAP`

80 COCO sınıfı kapsanmıştır. Örnekler:

| Nesne | Koku Dağılımı |
|---|---|
| car/truck/bus | İs/Duman 0.80-0.85, Limon 0.15-0.20 |
| boat | Deniz 0.90, Nane 0.10 |
| donut/cake | Vanilya 1.00 |
| horse/sheep/cow | Toprak 0.70-0.80, Çam 0.20-0.30 |
| bear | Çam 0.70, Toprak 0.30 |
| toothbrush | Nane 1.00 |
| orange | Limon 0.95, Vanilya 0.05 |

---

## 10. KARŞILAŞILAN PROBLEMLER VE ÇÖZÜMLER

### Problem 1: EMA Sinyali — Sabit Alpha Yetersizliği
**Sorun:** `_smooth_ema()` sabit `alpha` ile hızlı değişen sahnelerde gecikme, yavaş sahnelerde salınım yapıyordu. `dynamic_alpha`, `max_rise/fall_per_frame` gibi karmaşık ad-hoc kurallar birikmişti.

**Çözüm:** Tamamen kaldırıldı. Yerini `KalmanFilter1D` aldı. Q/R parametreleri fiziksel anlam taşır (işlem gürültüsü / ölçüm gürültüsü); per-kanal ayarlanabilir; config hot-reload destekler.

---

### Problem 2: Gökyüzü Mavi → Sahte Deniz
**Sorun:** Açık mavi gökyüzü K-Means ve Doku modüllerini Deniz'e yönlendiriyordu.

**Çözümler:**
- `_sky_false_positive_gate()`: Üst %40 bölge `blue_ratio > 0.45` + `lap_var < 150` → `Deniz × max(0.05, 1 - strength×0.90)`
- `_grid_sky_penalty()`: Gökyüzü bölgesine koku kimliği atanmaz, sadece `sky_freshness` float'ı üretilir; VCV'ye aktarılır.
- CLIP Absorber: Gökyüzü/Bulut absorber sınıfı softmax kütlesini yutar.

---

### Problem 3: Normalizasyon Şişmesi (Anti-Normalization Inflation)
**Sorun:** Birçok kanal VETO'landığında kalan tek zayıf kanal normalize ile %100'e fırlar. Örnek: İs/Duman=0.05, diğerleri veto → standart normalize → İs/Duman=1.00.

**Çözüm:** `_anti_inflate_normalize()`:
```
denom = sum(kalan kanallar) + ambient_floor (= 0.20)
result[s] = val / denom     (s ∈ kalan kanallar)
```
Örnek: `0.05 / (0.05 + 0.20) = 0.20` — makul değer.

---

### Problem 4: Kalman Artık Koku (Residual Smell)
**Sorun:** Post-Fusion Hard VETO bir kanalı sıfırlasa da 7-VCV / 7b / 7c / 7d adımları kanalı yeniden canlandırabiliyordu. Kalman'ın `x` hafızası eski değeri sızdırıyordu.

**Çözüm:** `Final Tamer Guard` (Phase-3): Kalman'a ulaşmadan önce VETO kanallarının Kalman hafızası mutlak sıfırlanır:
```python
blended[s]  = 0.0
kalman.filters[s].reset()      # x=0, P=1
hysteresis_state[s] = False
```

---

### Problem 5: Kensel Bej Bina → Sahte Vanilya
**Sorun:** K-Means bej/krem bina cephelerini Vanilya'ya eşliyordu. VLM urban dese de Bayes normalizasyonu sonrası Vanilya sızıyordu.

**Çözümler:**
- `compute_semantic_mask()` kentsel kapılar: `Vanilya = min(Vanilya, 0.01)`
- K-Means Tamer Kentsel modu: `w_Vanilya = 0.02`
- Post-Fusion Veto Kentsel mod: `blended["Vanilya"] = 0.0` + anti-inflate normalize

---

### Problem 6: SOFT_SUPPRESSION Bayes'te İptal Olması
**Sorun:** `strict_scent_allowed=False` durumunda VLM maskesi tüm kanallara `0.02` uyguluyordu. Ancak `log(0.02)` tüm kanallara eşit eklendi → normalizasyonda birbirini iptal etti → sıfır etki.

**Çözüm:** Post-Bayes Clamping: Bayes+Kalman hesaplandıktan SONRA çıkış kapısında:
```python
if not strict_ok:
    ui_scores = {s: v × _SOFT_SUPPRESSION for s, v in ui_scores.items()}
```

---

### Problem 7: VLM Eski Karar → Katman A Blokajı
**Sorun:** VLM 3 saniye önce "urban" demiş, sahne orman olmuş; maske hâlâ kentsel kapıları aktif tutuyordu.

**Çözüm:** `_VLM_FRESHNESS_TIMEOUT = 3.0 s`. Son VLM yanıtı 3 saniyeden eskiyse mask = `{s: 1.0}` (tam açık, bypass) döner.

---

### Problem 8: Thread Race Condition — mask ≠ is_void Tutarsızlığı
**Sorun:** `get_current_mask()`, `get_void_state()`, `consume_void_trigger()` üç ayrı çağrı arasına VLM thread'i yazabiliyordu → Bayes uniform dağılım bug'ı.

**Çözüm:** `get_mask_and_void_state()` tek lock altında (mask, is_void, void_triggered, strict_ok) dörtlüsünü atomik okur.

---

### Problem 9: GPU K-Means vs sklearn
**Sorun:** sklearn K-Means her karede `n_init=3` × `max_iter=50` = çok yavaş.

**Çözümler:**
- GPU: PyTorch `torch.cdist` ile tek pass Lloyd, `atol=1.0` erken çıkış.
- CPU: `cv2.kmeans` (C++ implementasyonu, sklearn'dan 3-5x hızlı) + warm start.

---

### Problem 10: CLIP float16 + LayerNorm Çakışması
**Sorun:** `torch_dtype=float16` + `autocast` kombinasyonunda LayerNorm float32 çıktı üretir → sonraki float16 matmul RuntimeError.

**Çözüm:** Model `float32` yüklenir, `autocast(enabled=cuda)` ile sadece matmul operasyonları float16'ya düşürülür:
```python
self._ovl_model = CLIPModel.from_pretrained(model_id).to(dev_str)   # float32
with torch.cuda.amp.autocast(enabled=(dev == "cuda")):
    out = self._ovl_model(**inputs)
logits = out.logits_per_image.squeeze(0).cpu().float()
```

---

## 11. KONFİGÜRASYON REFERANSI (config.json)

| Alan | Değer | Açıklama |
|---|---|---|
| `module_weights.clip` | 0.40 | BayesFuser CLIP ağırlığı |
| `module_weights.yolo` | 0.30 | BayesFuser YOLO ağırlığı |
| `module_weights.kmeans` | 0.15 | BayesFuser K-Means temel ağırlığı |
| `module_weights.texture` | 0.10 | BayesFuser Laplacian ağırlığı |
| `kalman_filters.İs/Duman` | Q=0.025, R=0.06 | En hızlı tepki |
| `kalman_filters.Nane` | Q=0.008, R=0.15 | En pürüzsüz |
| `hysteresis.activate_threshold` | 0.20 | Kanal açılma eşiği |
| `hysteresis.deactivate_threshold` | 0.08 | Kanal kapanma eşiği |
| `hysteresis.passive_decay_factor` | 0.45 | Pasif kanal gradyan çarpanı |
| `vcv.lambda` | 2.5 | VCV üstel sönümleme katsayısı |
| `vcv.urban_hardening_scale` | 5.0 | Kentsel sertleştirme çarpanı |
| `optical_flow.smoke_upward_threshold` | 0.30 | Duman yukarı hareket eşiği |
| `ovl.forest_dominance_threshold` | 0.60 | Orman baskın geçme engeli |
| `hardware.pwm_max` | 4095 | PCA9685 12-bit maksimum |
| `vlm.interval_frames` | 20 | VLM analiz kare aralığı |
| `vlm.fallback_timeout_seconds` | 10 | VLM stale timeout |

---

## 12. MİMARİ EVRİMİ (V1 → V3)

| Özellik | V1 | V2 | V3 |
|---|---|---|---|
| Sinyal birleştirme | Sabit ağırlık toplama | Ağırlıklı matris | Bayes log-uzayı posterior |
| Zamanlama | EMA (sabit alpha) | EMA (dinamik alpha) | Kalman filtresi |
| Semantik anlayış | Yok | Makro bağlam (YOLO) | VLM (moondream2/Ollama) |
| CLIP | Yok | Var (basit) | Absorber mimarisi + VCV |
| Prior güncelleme | Yok | Yok | Temporal Bayesian zincir |
| Sahne geçiş | Basit sıfırla | Purge controller | Kalman+Bayes+Histerezis sıfırla |

---

*Bu döküman `analyzer.py`, `pdm_engine.py`, `bayes_fuser.py`, `kalman.py`, `context_analyzer.py`, `engine.py`, `scent_controller.py` ve `config.json` dosyalarının tamamı incelenerek üretilmiştir.*
