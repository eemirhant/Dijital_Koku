# PRD: DijitalKoku V3 — Semantik-Olasılıksal Mimari

> **Versiyon:** 3.0 | **Tarih:** Mayıs 2026  
> **Hazırlayan:** Windsurf Cascade için teknik uygulama dokümanı  
> **Danışman:** Dr. Öğr. Üyesi Huzeyfe Muhammet Kocabaş  
> **Mevcut Durum:** V2.1 Faz 1 (pdm_engine.py aktif)  
> **Hedef Durum:** V3.0 — 4 haftalık refaktör ve yeni katman entegrasyonu

---

## 0. Neden V3? — Kök Sorun Analizi

Mevcut sistemin temel hastalığı **mimari tutarsızlık** değil, **bilgi hiyerarşisinin yokluğudur.**

V2.1'deki tüm modüller (YOLO, K-Means, Laplacian, CLIP) eşit oy hakkına sahiptir ve birbirinden habersiz çalışır. Galata Kulesi örneğinde Laplacian'ın kiremit dokusundan ürettiği yüksek frekans sinyali, K-Means'ın yeşilsizliği ve CLIP'in "urban" bağlamını sayısal olarak geçersiz kılabilir — çünkü sistem bu bilgiler arasında **semantik öncelik** kuramaz.

### V2.1'in Yapısal Sorunları

| Sorun | Nasıl Tezahür Ediyor |
|---|---|
| **Sihirli sayı patikası** | `if score > 0.38`, `if motion > 0.1` — her yeni sahne için yeni kural gerektirir |
| **Simetrik modül ağırlığı** | Laplacian yüzey frekansı ile CLIP semantiği aynı ağırlık havuzuna girer |
| **Düz sinyal işleme** | EMA, Kalman'ın aksine işlem gürültüsü (Q) ve ölçüm gürültüsü (R) ayrımı yapamaz |
| **Bağlam körlüğü** | Sistem karenin "ne gösterdiğini" değil, "ne içerdiğini" bilir — sahne tipi bilgisi yoktur |
| **Anlık modül vetosunun yokluğu** | Botanikal koku filtresi için renk sinyalini beklemek gerekir; makro bağlam bilgisi sisteme girmez |

### V3'ün Temel Felsefesi

> *"Düşük seviyeli sinyaller (piksel, doku, renk) ham veri üretir. Semantik katman bu veriyi anlamlı hale getirir. Bayes motoru anlamlı veriyi birleştirir. Kalman filtresi birleşik veriyi donanıma iletir."*

Bu hiyerarşi **yukarıdan aşağıya (top-down)** çalışır: VLM'in "kentsel alan, ağaç yok" kararı, aşağıdaki hiçbir düşük seviyeli sinyalin botanik koku üretmesine izin vermez.

---

## 1. Sistem Mimarisi — V3

### 1.1 Katman Şeması

```
┌─────────────────────────────────────────────────────┐
│                  VİDEO / EKRAN AKIŞI                │
└────────────────────────┬────────────────────────────┘
                         │
          ┌──────────────▼──────────────┐
          │   KATMAN A: Ham Öznitelikler │  (senkron, her kare)
          │  YOLOv8 │ K-Means │ Laplacian│
          └──────────────┬──────────────┘
                         │  normalize → [0,1] PDF vektörü
          ┌──────────────▼──────────────┐
          │ KATMAN B: VLM Semantik Kilit │  (asenkron thread, 15-30 kare/bir)
          │   Ollama → moondream2        │
          │   Çıktı: JSON bağlam maskesi │
          └──────────────┬──────────────┘
                         │  veto / ölçek maskesi
          ┌──────────────▼──────────────┐
          │  KATMAN C: Bayes Füzyon      │  (senkron, her kare)
          │  Prior × Likelihood → Posterior│
          │  CLIP absorber + VCV dahil   │
          └──────────────┬──────────────┘
                         │  posterior vektörü [0,1]
          ┌──────────────▼──────────────┐
          │  KATMAN D: Kalman Filtresi   │  (senkron, her kare)
          │  7 bağımsız 1D KalmanFilter  │
          │  Q (işlem gürültüsü) + R     │
          │  (ölçüm gürültüsü) ayrımı    │
          └──────────────┬──────────────┘
                         │  kararlı PWM skoru [0,1]
          ┌──────────────▼──────────────┐
          │   DONANIM: Arduino+PCA9685   │
          │   7 Difüzör + Tahliye Fanı   │
          └─────────────────────────────┘
```

### 1.2 Katman Sorumlulukları ve Sınırları

| Katman | Sorumluluk | Kesinlikle Yapmaması Gereken |
|---|---|---|
| A (Ham) | Piksel → normalize PDF | Eşik uygulamak, karar vermek |
| B (VLM) | Makro bağlam → JSON maske | Piksel düzeyinde analiz yapmak |
| C (Bayes) | PDF'leri birleştirip posterior üretmek | Donanıma doğrudan yazmak |
| D (Kalman) | Sinyali stabilize etmek | Skor yorumlamak |

---

## 2. Katman A — Ham Öznitelik Modülleri (Refaktör)

### 2.1 Genel İlke

Her modül artık deterministik bir karar değil, **7 elemanlı olasılık yoğunluk vektörü** döndürür. Tüm vektörler `[0, 1]` aralığına normalize edilir, toplamları 1 olmak zorunda değildir (her eleman bağımsız posterior hesabına girer).

```python
# Her modülün imzası:
def analyze(frame: np.ndarray) -> dict[str, float]:
    # Döndürür: {"Çam": 0.12, "Deniz": 0.67, "Vanilya": 0.05, ...}
```

### 2.2 YOLO Modülü

**Değişiklik:** `OBJECT_SCENT_MAP` tablosu korunur ancak çıktı normalize edilir.

```python
# ESKI — deterministik puan
raw_score += weight * spatial_weight * size_weight

# YENİ — normalize PDF
pdf = softmax(raw_scores_dict)  # veya min-max normalize
```

Kaldırılacak: `DOMINANCE_THRESHOLD = 0.38`, `DOMINANCE_SUPPRESS = 0.15` sabitleri.  
Korunacak: `TemporalMemory` (half_life=2.0sn), mekansal ve boyut ağırlıkları, çift katmanlı güven filtresi.

### 2.3 K-Means Modülü

**Değişiklik yok** — renk kuralları `COLOR_SCENT_RULES` normalize edilmiş çıktı üretecek şekilde sarmalanır. HSV ayrıştırma (sarı → alev/limon/krem) korunur.

### 2.4 Laplacian Doku Modülü

**Değişiklik:** Mutlak varyans → normalize frekans vektörü.

```python
lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
# Zengin doku → Çam/Toprak lehine; düz yüzey → Deniz/Gökyüzü lehine
texture_pdf = compute_texture_pdf(lap_var, dominant_hue)
```

**ÖNEMLİ:** Bu modülün Çam sinyali, Katman B VLM maskesiyle doğrudan çarpılır. Botanik veto gelirse bu modülün Çam çıktısı `→ 0` olur.

### 2.5 CLIP Modülü

**Değişiklik:** Absorber mimarisi korunur. `warmup_lock` korunur. EMA smoothing **kaldırılır** — bu artık Katman D'nin işidir.

---

## 3. Katman B — VLM Semantik Kilit (YENİ)

### 3.1 Teknoloji Seçimi

| Model | Boyut | Gecikme (RTX 3050) | Öneri |
|---|---|---|---|
| `moondream2` | ~1.8GB | ~180ms/kare | **Birincil tercih** |
| `minicpm-v` | ~4GB | ~280ms/kare | Yedek |
| `llava:7b` | ~4.5GB | ~350ms/kare | Hayır — bütçe aşımı |

**Seçim:** `moondream2` — Ollama API üzerinden, tamamen yerel, internet bağımlılığı yok.

### 3.2 ContextAnalyzerThread

`video_worker.py` veya `dashboard.py` ana döngüsünden bağımsız çalışan bir thread:

```python
class ContextAnalyzerThread(threading.Thread):
    """
    Asenkron VLM bağlam analizcisi.
    Ana video döngüsünü bloklamamak için daemon thread olarak çalışır.
    """
    def __init__(self, ollama_url: str, model: str, interval_frames: int):
        self.interval_frames = interval_frames  # config.json'dan: varsayılan 20
        self.context_mask: dict = DEFAULT_MASK  # thread-safe paylaşılan maske
        self._lock = threading.RLock()
        self._frame_queue = Queue(maxsize=1)    # sadece en güncel kare
    
    def submit_frame(self, frame: np.ndarray, frame_count: int):
        """Ana döngüden çağrılır. Sadece interval dolduğunda veya sahne kesiminde."""
        if frame_count % self.interval_frames == 0 or scene_cut_detected:
            # Queue doluysa eski kareyi at
            try: self._frame_queue.get_nowait()
            except: pass
            self._frame_queue.put_nowait(frame.copy())
    
    def get_current_mask(self) -> dict:
        """Bayes motoru tarafından her karede çağrılır. Non-blocking."""
        with self._lock:
            return self.context_mask.copy()
```

### 3.3 VLM Prompt Tasarımı

```python
CONTEXT_PROMPT = """Analyze this image and respond ONLY with a valid JSON object.
No explanations, no markdown, no extra text.

Required JSON structure:
{
  "environment": "<one of: urban_historic, urban_modern, indoor_domestic, indoor_commercial, nature_forest, nature_water, nature_open, vehicle_interior, unknown>",
  "organic_vegetation": <true/false>,
  "water_body": <true/false>,
  "fire_smoke": <true/false>,
  "dominant_materials": ["<material1>", "<material2>"],
  "confidence": <0.0-1.0>
}"""
```

**Prompt'u küçük tutmanın önemi:** moondream2 kısa, yapısal promptlarda daha tutarlı JSON üretir. Uzun açıklama eklenmez.

### 3.4 Semantik Maske Hesaplama

VLM'den gelen JSON, her koku kanalı için `[0.0, 1.0]` arası bir **ölçek katsayısı** üretir:

```python
def compute_semantic_mask(context: dict) -> dict[str, float]:
    """
    Returns a multiplier for each scent channel.
    1.0 = tam izin, 0.0 = tam veto, 0.5 = %50 azaltma
    """
    mask = {scent: 1.0 for scent in SCENTS}  # varsayılan: hepsine izin
    
    env = context.get("environment", "unknown")
    has_vegetation = context.get("organic_vegetation", True)  # şüphede izin ver
    has_water = context.get("water_body", False)
    has_fire = context.get("fire_smoke", False)
    
    # Botanik filtreler
    if not has_vegetation:
        mask["Çam"] *= 0.05    # Neredeyse veto
        mask["Nane"] *= 0.15   # Güçlü baskı
        mask["Toprak"] *= 0.30 # Orta baskı
    
    # Kentsel çevre düzeltmeleri
    if env in ("urban_historic", "urban_modern"):
        mask["Deniz"] *= 0.25 if not has_water else 1.0
        mask["Çam"] *= 0.10 if not has_vegetation else 1.0
    
    # Yangın doğrulaması
    if has_fire:
        mask["İs/Duman"] = min(mask["İs/Duman"] * 2.0, 1.0)  # güçlendir
    elif env == "nature_water":
        mask["İs/Duman"] *= 0.10  # su ortamında is mantıksız
    
    # Su ortamı güçlendirmesi
    if has_water or env == "nature_water":
        mask["Deniz"] = min(mask["Deniz"] * 1.5, 1.0)
    
    return mask
```

### 3.5 Fallback Davranışı

VLM yanıt vermezse veya JSON parse edilemezse:
- Önceki geçerli maske korunur (`last_valid_mask`)
- Maksimum 10 saniyelik stale tolerance — sonrasında `DEFAULT_MASK` (tüm değerler 1.0) devreye girer
- Bu durum telemetri loguna yazılır: `"vlm_status": "stale"` 

---

## 4. Katman C — Bayes Füzyon Motoru (Refaktör)

### 4.1 Matematiksel Model

Her koku kanalı `s` için posterior hesabı:

```
P(s | data) ∝ P(s) × ∏ P(data_i | s)
```

- `P(s)` = Prior: önceki karedeki posterior (zaman serisi sürekliliği)
- `P(data_i | s)` = Likelihood: her modülün o kanal için ürettiği normalize değer
- Maske uygulaması: `P(data | s) *= semantic_mask[s]`

```python
def fuse(
    module_pdfs: dict[str, dict[str, float]],  # {"yolo": {...}, "kmeans": {...}, ...}
    semantic_mask: dict[str, float],
    prior: dict[str, float],
    module_weights: dict[str, float]            # config.json'dan
) -> dict[str, float]:
    
    posterior = {}
    for scent in SCENTS:
        # Log-uzayında Bayes çarpımı (sayısal kararlılık için)
        log_likelihood = 0.0
        for module_name, pdf in module_pdfs.items():
            w = module_weights.get(module_name, 0.25)
            val = max(pdf.get(scent, 1e-6), 1e-6)  # sıfırdan kaçın
            log_likelihood += w * math.log(val)
        
        # Semantik maske → log-uzayında ölçekleme
        mask_val = max(semantic_mask.get(scent, 1.0), 1e-6)
        log_likelihood += math.log(mask_val)
        
        # Prior ile birleştir
        prior_val = max(prior.get(scent, 0.1), 1e-6)
        posterior[scent] = prior_val * math.exp(log_likelihood)
    
    # Normalize (softmax benzeri)
    total = sum(posterior.values()) + 1e-9
    return {s: v / total for s, v in posterior.items()}
```

### 4.2 Prior Yönetimi

- İlk kare: `prior = {scent: 1/7 for scent in SCENTS}` (düz prior)
- Sahne kesimi: prior tamamen sıfırlanır — `engine.reset()` çağrısı gibi
- Normal akış: önceki karenin posterior'u yeni prior olur

### 4.3 Mevcut Güvenlik Katmanları — Muhafaza

Aşağıdaki mekanizmalar Bayes motoruna entegre edilir (kaldırılmaz):

| Mekanizma | Konum V3'te | Notu |
|---|---|---|
| VCV (Vektörel Bağlam Doğrulaması) | Katman C — likelihood ölçeği | `_SCENT_AXIS` tablosu korunur |
| Histerezis | Katman C — posterior eşiği | Açılma ≥ 0.20, kapanma < 0.08 |
| N-Kare Oylama | Katman C — posterior kararlılığı | Son N posterior ortalaması |
| Gökyüzü Kapısı | Katman B maske (VLM doğrular) | Ayrıca Katman A CLIP absorber |
| Aydınlatma Profili | Katman A → YOLO/CLIP ön işleme | Korunur |
| Koku Körlüğü | Katman D çıkış sınırlayıcısı | Saturasyon mantığı Kalman'a taşınır |
| Tahliye Kontrolörü | Katman D — sahne kesimi tetikleyici | Korunur |

### 4.4 Kaldırılan Sabit Eşikler

Aşağıdaki satırlar `pdm_engine.py` ve `engine.py` dosyalarından **tamamen silinir:**

```python
# KALDIRILACAK — bu tür her ifade
if score > 0.38: ...
if motion > 0.1: ...
DOMINANCE_THRESHOLD = 0.38
if blend_score > threshold: suppress(others, 0.15)
# vb.
```

Bu eşiklerin yerini Bayes posterior'unun doğal olasılık dağılımı alır.

---

## 5. Katman D — 1D Kalman Filtresi (YENİ)

### 5.1 Teori

EMA formülü: `s_t = α × x_t + (1-α) × s_{t-1}`

EMA'nın sorunu: `α` sabit veya kaba hareket heuristiğiyle ayarlanır. Kalman filtresi ise **iki belirsizliği ayrı modeller:**
- `Q` (process noise): sistemin ne kadar hızlı gerçekten değiştiği
- `R` (measurement noise): modüllerden gelen verinin ne kadar güvenilir olduğu

### 5.2 Implementasyon

```python
class KalmanFilter1D:
    """
    Her koku kanalı için bağımsız 1D Kalman filtresi.
    Durum: tek boyutlu koku yoğunluğu [0, 1]
    """
    def __init__(self, Q: float = 0.01, R: float = 0.1):
        self.Q = Q   # İşlem gürültüsü — düşük = daha yavaş adaptasyon
        self.R = R   # Ölçüm gürültüsü — yüksek = ölçüme daha az güven
        self.x = 0.0  # Durum tahmini
        self.P = 1.0  # Hata kovaryansı
    
    def update(self, measurement: float) -> float:
        # 1. Tahmin adımı
        x_pred = self.x
        P_pred = self.P + self.Q
        
        # 2. Kalman kazancı
        K = P_pred / (P_pred + self.R)
        
        # 3. Güncelleme adımı
        self.x = x_pred + K * (measurement - x_pred)
        self.P = (1 - K) * P_pred
        
        return max(0.0, min(1.0, self.x))  # [0,1] sınırla
    
    def reset(self):
        self.x = 0.0
        self.P = 1.0
```

### 5.3 Kanal Başına Filtre Parametreleri

`config.json`'dan yönetilir, varsayılanlar:

```json
{
  "kalman_filters": {
    "Çam":      {"Q": 0.008, "R": 0.12},
    "Deniz":    {"Q": 0.015, "R": 0.08},
    "Vanilya":  {"Q": 0.010, "R": 0.10},
    "İs/Duman": {"Q": 0.025, "R": 0.06},
    "Nane":     {"Q": 0.008, "R": 0.15},
    "Toprak":   {"Q": 0.010, "R": 0.12},
    "Limon":    {"Q": 0.012, "R": 0.10}
  }
}
```

**Parametre mantığı:**
- Yüksek `Q` → hızlı değişen sahne kokulara hızlı tepki (İs/Duman, Deniz — ani yangın/su sahneleri)
- Yüksek `R` → ölçüme az güven → daha pürüzsüz çıkış (Nane, Çam — nadiren ani değişir)

### 5.4 Sahne Kesimi Yönetimi

```python
def on_scene_cut(self):
    """Tüm Kalman filtrelerini sıfırla + Bayes prior'u resetle."""
    for kf in self.kalman_filters.values():
        kf.reset()
    self.bayes_prior = {s: 1/7 for s in SCENTS}
    self.purge_controller.trigger()  # 1.5sn tahliye fanı
```

---

## 6. Fonksiyonel Gereksinimler

| ID | Gereksinim | Öncelik | Kabul Kriteri |
|---|---|---|---|
| **FR-01** | Tüm sihirli sayı eşikleri kaldırılır | **Epik / Must** | `pdm_engine.py` ve `engine.py`'da `if score >` veya `if X > THRESHOLD` kalıbı bulunmaz |
| **FR-02** | Modüllerin çıktısı normalize PDF vektörü | **Epik / Must** | Her modülden dönen dict değerleri `[0,1]` aralığında, `softmax` veya `min-max` ile normalize |
| **FR-03** | Ollama VLM entegrasyonu (moondream2) | **Epik / Must** | Kare başına `< 300ms` (asenkron), JSON parse başarı oranı `> %90` |
| **FR-04** | Semantik maske Bayes motoruna bağlanır | **Epik / Must** | Galata Kulesi karesi: `organic_vegetation=false` geldiğinde Çam PWM çıkışı `< 0.05` |
| **FR-05** | Bayes Log-Uzayı Füzyonu | **Yüksek / Should** | Tüm modüllerin ağırlıklı Bayes çarpımı — doğrudan toplamsal blend yok |
| **FR-06** | 7 bağımsız 1D Kalman Filtresi | **Yüksek / Should** | PWM çıkışlarındaki kare-kare dalgalanma (jitter) `< %3` (sabit sahne, 5 ardışık kare) |
| **FR-07** | `config.json` üzerinden Q/R yönetimi | **Orta / Could** | Uygulama yeniden başlatmadan Kalman parametresi değiştirilir (hot-reload) |
| **FR-08** | VLM fallback koruması | **Yüksek / Should** | VLM 10sn yanıt vermezse `DEFAULT_MASK` devreye girer, sistem çalışmaya devam eder |
| **FR-09** | Telemetri paneli genişletme | **Orta / Could** | Dashboard'da VLM bağlam JSON ve Kalman state grafikleri görünür |
| **FR-10** | `engine.py` legacy motoru koru | **Düşük / Won't** | V3 refaktörü sadece `pdm_engine.py` üzerinde yapılır; `engine.py` dokunulmaz (yedek) |

---

## 7. Teknik Gereksinimler & Performans Kriterleri

### 7.1 Gecikme Bütçesi (RTX 3050 referans)

| Katman | Hedef Gecikme | Çalışma Modu |
|---|---|---|
| YOLO + K-Means + Laplacian (Katman A) | `< 30ms` / kare | Senkron, ana thread |
| CLIP (Katman A) | `< 80ms` / analiz aralığı | Asenkron (`_ovl_worker`) |
| VLM / moondream2 (Katman B) | `< 300ms` / 20 kare | Asenkron (`ContextAnalyzerThread`) |
| Bayes Füzyon (Katman C) | `< 5ms` / kare | Senkron, ana thread |
| Kalman Filtresi (Katman D) | `< 1ms` / kare | Senkron, ana thread |
| **Toplam (ana thread)** | `< 36ms` / kare | ~27 FPS |

### 7.2 Stabilite Kriterleri

- **Jitter:** Ardışık 5 kare sabit sahnede PWM delta `< %3`
- **VLM bağlam geçişi:** Yeni maske geldiğinde PWM değişimi ani değil, Kalman rampalı
- **Sahne kesimi:** Reset sonrası ilk geçerli posterior `< 3 kare` içinde stabilize olur

### 7.3 Yerel Çalışma (Privacy-First)

- Ollama API: tamamen yerel (`http://localhost:11434`)
- CLIP: yerel GPU
- İnternet bağlantısı gerektiren bileşen: **sıfır**

---

## 8. Dosya Değişiklik Haritası

### 8.1 Değişecek Dosyalar

| Dosya | Değişiklik Özeti |
|---|---|
| `pdm_engine.py` | **Büyük refaktör** — Katman A normalize çıktı, sihirli sayılar kaldırılır, Bayes log füzyon eklenir, Kalman entegrasyonu, VLM maske bağlantısı |
| `config.json` | Kalman `Q/R` parametreleri, VLM interval, Ollama URL, Bayes modül ağırlıkları eklenir |
| `video_worker.py` | `ContextAnalyzerThread` entegrasyonu — frame submission noktaları eklenir |
| `dashboard.py` | VLM durum göstergesi, Kalman state grafikleri, `context_mask` görselleştirme eklenir |

### 8.2 Yeni Dosyalar

| Dosya | İçerik |
|---|---|
| `context_analyzer.py` | `ContextAnalyzerThread` sınıfı + Ollama API istemcisi + maske hesaplama |
| `kalman.py` | `KalmanFilter1D` sınıfı + `KalmanFilterBank` (7 kanal yöneticisi) |
| `bayes_fuser.py` | `BayesFuser` sınıfı + log-uzayı çarpım motoru + prior yönetimi |

### 8.3 Dokunulmayacak Dosyalar

| Dosya | Gerekçe |
|---|---|
| `engine.py` | Legacy yedek motor — V3 tamamlanana kadar çalışır halde kalır |
| `analyzer.py` | Görüntü yakalama + YOLOv8 + K-Means — çıktı formatı güncellenir ama mantık korunur |
| `requirements.txt` | Eklenti: `requests` (Ollama HTTP), `filterpy` veya sıfırdan `kalman.py` |

---

## 9. Yol Haritası (Roadmap)

### Faz 1 — Temizlik ve Normalize Altyapı (Hafta 1)

**Hedef:** Sistem sihirli sayılardan arındırılır, modül çıktıları standartlaşır.

**Görevler:**
1. `pdm_engine.py`: tüm `if score > X` ve `DOMINANCE_THRESHOLD` ifadelerini kaldır
2. YOLO, K-Means, Laplacian modüllerinin çıktısını `softmax` ile normalize et
3. `kalman.py` — `KalmanFilter1D` ve `KalmanFilterBank` sınıflarını yaz
4. `bayes_fuser.py` — log-uzayı Bayes çarpım prototipini yaz (VLM maskesi olmadan, `mask=1.0` sabit)
5. Galata Kulesi ve İstanbul Boğazı test videoları ile temel regresyon testleri yap
6. **Kabul kriteri:** Sistem EMA'sız, Kalman ile çalışıyor ve kaba testlerde mantıklı çıktı üretiyor

### Faz 2 — VLM Semantik Kilit (Hafta 2)

**Hedef:** Ollama/moondream2 entegrasyonu tamamlanır, Katman B çalışır.

**Görevler:**
1. Ollama kurulumu doğrula, `moondream2` modelini indir: `ollama pull moondream2`
2. `context_analyzer.py` — `ContextAnalyzerThread` yaz
3. Prompt'u test et: çeşitli sahne tipleri (şehir, orman, deniz, iç mekan) için JSON çıktısını doğrula
4. `compute_semantic_mask()` fonksiyonunu yaz ve `BayesFuser`'a bağla
5. **Kritik test:** Galata Kulesi karesi → `organic_vegetation=false` → Çam PWM `< 0.05`
6. **Kabul kriteri:** VLM asenkron çalışıyor, ana döngü bloklama yaşamıyor, fallback çalışıyor

### Faz 3 — Kalman Entegrasyonu ve Validasyon (Hafta 3)

**Hedef:** Donanım çıkışı stabilize edilir, akademik telemetri üretilir.

**Görevler:**
1. `KalmanFilterBank`'ı `pdm_engine.py` çıkış katmanına bağla (EMA'yı tamamen kaldır)
2. `config.json`'a Kalman Q/R parametrelerini ekle, hot-reload'ı test et
3. İstanbul Boğazı panning testi: Deniz kokusu Kalman ile yumuşak sönümleniyor mu?
4. `dashboard.py`'a Kalman state grafiği ekle (Matplotlib/Streamlit line chart)
5. Sahne kesimi → filtre reset → rampasız geçiş testini doğrula
6. **Kabul kriteri:** Jitter `< %3`, panning testinde pürüzsüz sönümlenme grafiği

### Faz 4 — Entegrasyon, Optimizasyon, Akademik Rapor (Hafta 4)

**Hedef:** Sistem bütünleşik çalışır, mezuniyet raporu için çıktılar üretilir.

**Görevler:**
1. VLM prompt'u küçült — `moondream2` ile en kısa güvenilir JSON prompt'u bul
2. Gecikme profillemesi: her katmanın kare başına süresini logla
3. Telemetri CSV'ye `vlm_context`, `kalman_state`, `bayes_posterior` kolonları ekle
4. Sistem mimarisi şeması (bu PRD'deki Bölüm 1.1 diyagramı) Matplotlib veya Draw.io ile görselleştir
5. Dr. Kocabaş için: doğruluk grafikleri, telemetri raporları, sistem dondurma
6. **Kabul kriteri:** Tüm FR-01..FR-08 kriterleri karşılanmış; iki test videosu için tam telemetri mevcut

---

## 10. Başarı Kriterleri (Kabul Testleri)

### Test T-01: Galata Kulesi Testi (Botanik Veto)

```
Girdi: Galata Kulesi videosu (kentsel, kiremit çatılar, ağaç yok)
Beklenen VLM çıktısı: {"environment": "urban_historic", "organic_vegetation": false, ...}
Beklenen Semantik Maske: {"Çam": 0.05, "Nane": 0.15, ...}
Beklenen PWM (Çam): < 0.05 (4095 × 0.05 = ~205)
Başarı: Çam ve Nane kanalları kararlı şekilde 0'a yakın kalır
```

### Test T-02: İstanbul Boğazı Panning Testi (Kalman Stabilizasyonu)

```
Girdi: Hızlı kamera dönüşlü boğaz videosu
Beklenen: Deniz koku skoru panning sırasında anlık spike yapmaz
Beklenen: Kalman çıkış grafiği pürüzsüz, derivative < 0.05/kare
Başarı: PWM dalgalanması ardışık 5 karede <%3
```

### Test T-03: Yangın Sahnesi (VLM Güçlendirme)

```
Girdi: GTA V yanan araba karesi
Beklenen VLM: {"fire_smoke": true, ...}
Beklenen Maske: {"İs/Duman": min(x*2.0, 1.0), "Deniz": *0.10, ...}
Beklenen PWM: İs/Duman dominant, Deniz/Vanilya < 0.10
Başarı: VLM maske yokken de doğru ama maskeyle daha hızlı/güçlü yanıt
```

### Test T-04: VLM Fallback Testi

```
Senaryo: Ollama servisi kapatılır, sistem çalışmaya devam eder
Beklenen: 10sn sonra DEFAULT_MASK (tüm 1.0) devreye girer
Beklenen: Telemetri'de "vlm_status: stale" görünür
Başarı: Sistem crash olmaz, koku çıkışı (VLM'siz) üretilmeye devam eder
```

---

## 11. config.json — V3 Şeması

```json
{
  "version": "3.0",
  
  "module_weights": {
    "clip": 0.40,
    "yolo": 0.30,
    "kmeans": 0.15,
    "texture": 0.10,
    "vlm_bias": 0.05
  },
  
  "bayes": {
    "prior_reset_on_scene_cut": true,
    "flat_prior_value": 0.1429,
    "log_space_fusion": true
  },
  
  "kalman_filters": {
    "Çam":      {"Q": 0.008, "R": 0.12},
    "Deniz":    {"Q": 0.015, "R": 0.08},
    "Vanilya":  {"Q": 0.010, "R": 0.10},
    "İs/Duman": {"Q": 0.025, "R": 0.06},
    "Nane":     {"Q": 0.008, "R": 0.15},
    "Toprak":   {"Q": 0.010, "R": 0.12},
    "Limon":    {"Q": 0.012, "R": 0.10}
  },
  
  "vlm": {
    "enabled": true,
    "ollama_url": "http://localhost:11434",
    "model": "moondream2",
    "interval_frames": 20,
    "scene_cut_trigger": true,
    "fallback_timeout_seconds": 10,
    "max_retries": 2
  },
  
  "hysteresis": {
    "activate_threshold": 0.20,
    "deactivate_threshold": 0.08
  },
  
  "temporal_memory": {
    "half_life_seconds": 2.0
  },
  
  "purge_controller": {
    "trigger_duration_ms": 1500
  }
}
```

---

## 12. Bağımlılık Güncellemeleri

### Eklenecek

```
requests>=2.31.0          # Ollama HTTP API istemcisi
# filterpy>=1.4.5         # OPSİYONEL — sıfırdan KalmanFilter1D yazılırsa gerek yok
```

### Kaldırılabilir (opsiyonel)

```
# EMA smoothing logic — Kalman tamamen yerini aldığında
```

### Değişmeyenler

```
ultralytics, transformers, torch, opencv-python, scikit-learn, streamlit, ...
```

---

## 13. Akademik Çıktılar (Dr. Kocabaş için)

Faz 4 sonunda aşağıdaki materyaller dondurulacak:

1. **Sistem Mimarisi Şeması** — 4 katmanlı pipeline diyagramı (bu PRD Bölüm 1.1)
2. **Telemetri Grafikleri** — Test T-01 ve T-02 için kare-kare koku skor grafikleri
3. **Bayes Analizi** — Prior/Likelihood/Posterior geçişlerini gösteren örnek hesap tablosu
4. **Kalman Validasyonu** — EMA vs Kalman jitter karşılaştırma grafiği
5. **Gecikme Profili** — Her katmanın ortalama ve maksimum gecikme tablosu
6. **VLM Doğruluk Özeti** — 10 test karesi üzerinde JSON parse başarı oranı ve maske tutarlılığı

---

*PRD Hazırlık Tarihi: Mayıs 2026 — DijitalKoku V3.0 Mimarisi*  
*Bu doküman Windsurf Cascade/Agent moduna doğrudan verilebilir.*
