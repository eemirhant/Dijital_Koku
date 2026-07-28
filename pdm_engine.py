"""
pdm_engine.py — DijitalKoku Olasılıksal Karar Matrisi (PDM)

Statik kural tabanlı yapıdan modüler, konfigüre edilebilir mimariye geçiş:
  - Her analiz modülü (YOLO, K-Means, Texture) bağımsız 0-1 güven skoru üretir
  - Ağırlıklar config.json'dan okunur; hot-reload desteklenir
  - Histerezis kapısı + N-kare çoğunluk oylaması kararsız sinyal salınımını önler
  - CLAHE ön işleme: ışık koşullarından bağımsız stabil normalizasyon
"""

import json
import logging
import math
import time
import threading
import cv2
import numpy as np
from collections import deque
from pathlib import Path
from engine import SCENTS, OBJECT_SCENT_MAP

# V3 yeni katmanlar
from kalman import KalmanFilterBank
from bayes_fuser import BayesFuser
from context_analyzer import ContextAnalyzerThread, DEFAULT_MASK, _SOFT_SUPPRESSION

_log = logging.getLogger(__name__)

# ── GPU Durum Kontrolü (başlangıçta bir kez yazdırılır) ──────────────────────
try:
    import torch as _t
    _GPU_DEVICE = "cuda" if _t.cuda.is_available() else "cpu"
    if _GPU_DEVICE == "cuda":
        _gpu_name = _t.cuda.get_device_name(0)
        _vram_mb  = _t.cuda.get_device_properties(0).total_memory // (1024 ** 2)
        _log.info("[GPU CHECK] Aktif Cihaz : cuda  |  %s  |  %d MB VRAM", _gpu_name, _vram_mb)
    else:
        _log.warning("[GPU CHECK] UYARI: CUDA bulunamadı — tüm modeller CPU üzerinde çalışıyor!")
        _log.warning("[GPU CHECK] CUDA destekli PyTorch için: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
    del _t
except ImportError:
    _GPU_DEVICE = "cpu"
    _log.warning("[GPU CHECK] PyTorch bulunamadı — GPU desteği devre dışı")

CONFIG_PATH = Path(__file__).parent / "config.json"

# Module-level torch + PIL — OVL metodları için; ImportError yoksa None
try:
    import torch
    from PIL import Image as PILImage
except ImportError:
    torch = None      # type: ignore[assignment]
    PILImage = None   # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Satır içi renk→koku kuralları  (H: 0-360, S/V: 0-255)
# Formatı: {"range": (h_min, h_max, s_min, v_min), "scents": {koku: ağırlık}}
# ---------------------------------------------------------------------------
_COLOR_RULES: list[dict] = [
    # — Alev / Kor (Yüksek doygunluk, sıcak tonlar) ————————————————————————
    {"range": (0,   25,  150,  80), "scents": {"İs/Duman": 0.85, "Vanilya": 0.15}},
    {"range": (25,  45,  150, 100), "scents": {"İs/Duman": 0.70, "Toprak":  0.30}},
    # — Sarı alev (yüksek S zorunlu; limon/krem buna düşmez) ————————————————
    {"range": (45,  65,  150, 150), "scents": {"Limon":    0.80, "Vanilya": 0.20}},
    # — Sarı-yeşil / Açık orman (s_min=45 ile desature yeşil yakalanır) ———————
    {"range": (55,  95,   45,  40), "scents": {"Çam": 0.65, "Toprak": 0.20, "Nane": 0.15}},
    # — Orman yeşili / koyu yaprak (s_min=45; orta V yeterli) ——————————————
    {"range": (95,  150,  45,  30), "scents": {"Çam": 0.55, "Deniz": 0.25, "Nane": 0.20}},
    # — Desatüre / Soluk Yeşil (sonbahar yaprak, çayır, sis içi orman; s=20-44)
    # Bu kural s<45 yeşil tonlarını Toprak fallback'e düşmekten korur.
    {"range": (55,  150,  20,  30), "scents": {"Çam": 0.65, "Toprak": 0.25, "Nane": 0.10}},
    # — Turkuaz / Teal (s_min=102≈40%; düşük doygunluk gri-yeşil tonu Deniz'e düşmesin) ———
    {"range": (150, 170, 102,  60), "scents": {"Nane": 0.55, "Deniz": 0.45}},
    # — Su / Gökyüzü / Nehir (s_min=102≈40%; puslu gökyüzü/gri-mavi gölgeler elenir) ————
    {"range": (170, 230, 102,  40), "scents": {"Deniz": 0.80, "Nane": 0.20}},
    # — Yosunlu/Mat Nehir / Akarsu Yüzeyi (orman yansıması; S düşük ama koyu; s_min=76≈30%, v_min=100) —
    {"range": (175, 220,  76, 100), "scents": {"Deniz": 0.60, "Nane": 0.25, "Çam": 0.15}},
    # — Mavi-mor ————————————————————————————————————————————————————————————
    {"range": (230, 270,  80,  60), "scents": {"Deniz": 0.60, "Vanilya": 0.40}},
    # — Mor / Pembe ————————————————————————————————————————————————————————
    {"range": (270, 330, 100,  80), "scents": {"Vanilya": 0.70, "Limon":  0.30}},
    # — Kırmızı tonu (ters taraf) ——————————————————————————————————————————
    {"range": (330, 360, 150,  80), "scents": {"İs/Duman": 0.80, "Vanilya": 0.20}},
    # — Beyaz / Açık gri (şelale köpüğü, sis, bulut; düşük S, yüksek V) ————
    # NOT: bu kural s_min=0 olduğundan yüksek V'li tüm renklere uyar;
    # yukarıdaki yeşil/mavi kuralları s>=45 gerektirdiğinden önce kesilir.
    {"range": (0,   360,   0, 175), "scents": {"Deniz": 0.40, "Nane": 0.35, "Çam": 0.25}},
    # — Genel fallback (koyu/nötr — doğa bağlamına uygun dağılım) ————————
    {"range": (0,   360,   0,   0), "scents": {"Toprak": 0.40, "Çam": 0.35, "Vanilya": 0.25}},
]

# ---------------------------------------------------------------------------
# Formsuz Ortam Kokuları — YOLO'nun sınıf olarak tanımlayamadığı sahne türleri
# (açık deniz, orman, çayır, toprak alanı gibi)
# Bu kokular K-Means'ten ezici çoğunluk geldiğinde YOLO vetosundan muaf tutulur.
# ---------------------------------------------------------------------------
FORMLESS_SCENTS: frozenset[str] = frozenset({"Deniz", "Çam", "Toprak"})

# ---------------------------------------------------------------------------
# Koku Baskınlık Hiyerarşisi — gerçek dünya keskinlik/maskeleme güçleri
# ---------------------------------------------------------------------------
SCENT_DOMINANCE: dict[str, float] = {
    "İs/Duman": 1.00,  # En keskin; tüm ortamı kaplar
    "Nane":     0.80,  # Serin ve belirgin
    "Çam":      0.60,  # Orta güç, doğal
    "Deniz":    0.50,  # Orta-hafif
    "Toprak":   0.40,  # Hafif zemin tonu
    "Limon":    0.35,  # Hafif-tatlı
    "Vanilya":  0.25,  # En zayıf; kolayca maskelenir
}

# ---------------------------------------------------------------------------
# Makro Bağlam Etiketleri — YOLO label seti → sahne kategorisi
# ---------------------------------------------------------------------------
_MACRO_CTX_LABELS: dict[str, frozenset] = {
    "INDOOR": frozenset({
        "sofa", "chair", "bed", "tv", "laptop", "keyboard", "mouse",
        "remote", "book", "clock", "vase", "couch", "toilet", "sink",
        "refrigerator", "oven", "toaster", "microwave", "dining table",
        "monitor", "cell phone", "scissors", "teddy bear",
    }),
    "NATURE": frozenset({
        "bird", "bear", "zebra", "giraffe", "elephant", "cow", "horse",
        "sheep", "cat", "dog", "potted plant",
    }),
    "WATER": frozenset({
        "boat", "surfboard",
    }),
    "KITCHEN": frozenset({
        "banana", "apple", "orange", "pizza", "cake", "carrot", "broccoli",
        "hot dog", "sandwich", "donut", "cup", "fork", "knife", "spoon",
        "bowl", "wine glass",
    }),
    "CITY": frozenset({
        "car", "truck", "bus", "motorcycle", "bicycle", "person",
        "traffic light", "stop sign", "parking meter",
    }),
}

# ---------------------------------------------------------------------------
# Bağlama göre koku çarpanları  (1.0 = değişiklik yok, <1 baskıla, >1 güçlendir)
# ---------------------------------------------------------------------------
_CTX_SCENT_ADJUSTMENTS: dict[str, dict[str, float]] = {
    "INDOOR": {
        "Çam":      0.45,   # Kapalı mekanda orman kokusu anlamsız
        "Deniz":    0.40,   # Kapalı mekanda deniz kokusu anlamsız
        "Toprak":   0.50,
        "Nane":     1.15,   # Aromaterapi / iç mekan
        "Vanilya":  1.25,   # Ev kokusu; makul
        "İs/Duman": 0.85,
        "Limon":    1.10,
    },
    "NATURE": {
        "Çam":      1.35,
        "Toprak":   1.25,
        "Nane":     1.10,
        "Deniz":    1.00,
        "Vanilya":  0.55,
        "Limon":    0.70,
        "İs/Duman": 0.75,
    },
    "WATER": {
        "Deniz":    1.45,
        "Nane":     1.20,
        "Çam":      0.80,
        "Toprak":   0.55,
        "Vanilya":  0.55,
        "Limon":    0.80,
        "İs/Duman": 0.45,
    },
    "KITCHEN": {
        "Vanilya":  1.35,
        "Limon":    1.30,
        "Nane":     1.10,
        "Toprak":   0.45,
        "İs/Duman": 0.55,
        "Çam":      0.55,
        "Deniz":    0.60,
    },
    "CITY": {
        "İs/Duman": 1.25,
        "Toprak":   1.15,
        "Deniz":    0.75,
        "Çam":      0.65,
        "Vanilya":  0.65,
        "Limon":    0.80,
        "Nane":     0.90,
    },
}

# ---------------------------------------------------------------------------
# Kesin Veto Tablosu — bağlamla mantıksal olarak bağdaşmayan kokular
# {macro_context: sıfırlanacak kokular kümesi}
# ---------------------------------------------------------------------------
_HARD_VETO_TABLE: dict[str, frozenset] = {
    "NATURE": frozenset({"Vanilya"}),
    "WATER":  frozenset({"Vanilya"}),
}

# ---------------------------------------------------------------------------
# Çapa Dış Mekan Nesneleri
# Bu nesnelerden herhangi biri tespit edilirse, K-Means ne kadar krem/sarı
# rengi bulursa bulsun, iç mekan kokuları (ANCHOR_VETO_SCENTS) kesin veto alır.
# ---------------------------------------------------------------------------
_ANCHOR_OUTDOOR_OBJECTS: frozenset[str] = frozenset({
    "bird", "airplane", "boat", "horse", "cow", "sheep", "elephant",
    "bear", "zebra", "giraffe", "kite", "frisbee", "snowboard", "skis",
    "surfboard",
})

_ANCHOR_VETO_SCENTS: frozenset[str] = frozenset({"Vanilya"})

# Gün batımı (GOLDEN_HOUR) senaryosunda mantıksız kokular — kesin sıfır
_GOLDEN_HOUR_VETO_SCENTS: frozenset[str] = frozenset({"Vanilya", "İs/Duman", "Nane", "Çam"})

# İç mekan çapa nesneleri — bunlardan en az biri varsa sahne asla dış mekan sayılamaz
_ANCHOR_INDOOR_OBJECTS: frozenset[str] = frozenset({
    "book", "shelf", "chair", "table", "tv", "laptop",
    "mouse", "keyboard", "cup", "bottle", "vase",
})

# ---------------------------------------------------------------------------
# Vektörel Bağlam Doğrulaması (VCV) — Anlamsal Eksen Yapıları
# ---------------------------------------------------------------------------
# Her koku için (doğa_ağırlığı, kentsel_ağırlık) toplamı 1.0.
# Kokuların hangi bağlama "ait olduğunu" kodlar; sabit kural yerine
# matematiksel sönümleme (weight decay) hesaplamasında kullanılır.
_SCENT_AXIS: dict[str, tuple[float, float]] = {
    "Çam":      (0.95, 0.05),   # neredeyse tamamen doğa/orman → urban çatışması çok sert
    "Deniz":    (0.88, 0.12),   # su/açık doğa bağlamı
    "Nane":     (0.60, 0.40),   # doğal ama kentsel bahçelerde de var
    "Toprak":   (0.88, 0.12),   # doğa ağırlığı artırıldı → urban sönümü güçlendi (0.78→0.88)
    "Limon":    (0.25, 0.75),   # tropikal/Akdeniz → karasal/kuzey doğa sahnelerinde güçlü sönümleme
    "Vanilya":  (0.10, 0.90),   # neredeyse tamamen iç mekan / kentsel
    "İs/Duman": (0.22, 0.78),   # egzoz / fabrika → kentsel ağırlıklı
}

# VCV: YOLO kentsel sinyal kaynakları
_VCV_URBAN_LABELS: frozenset[str] = frozenset({
    "car", "truck", "bus", "motorcycle", "train", "bicycle",
    "traffic light", "stop sign", "parking meter", "fire hydrant",
})

# VCV: Büyük ekran sahte pozitif koruması — geniş alan tespitinde CLIP bağlam güveni kırpılır
_VCV_SCREEN_LABELS: frozenset[str] = frozenset({
    "tv", "laptop", "monitor",
})

# VCV: YOLO doğa sinyal kaynakları
_VCV_NATURE_LABELS: frozenset[str] = frozenset({
    "tree", "plant", "potted plant", "flower", "grass", "mountain",
    "rock", "beach", "bird", "horse", "cow", "sheep", "elephant",
    "bear", "zebra", "giraffe", "dog", "cat", "kite",
    "surfboard", "boat", "skis", "snowboard",
})

# ---------------------------------------------------------------------------
# Yardımcı Sınıf 1 — Zaman Damgalı Nesne Hafızası
# ---------------------------------------------------------------------------

class TemporalMemory:
    """
    FPS'den bağımsız nesne hafızası.
    Her tespitte last_seen_timestamp güncellenir.
    Nesne kaybolduktan sonra get_weight() logaritmik yarılanma ömrü uygular:
        weight = conf * 2^(-age / half_life)
    """

    def __init__(self, half_life: float = 2.0, max_objects: int = 30):
        self._half_life  = half_life
        self._max        = max_objects
        self._seen:  dict[str, float] = {}   # label → son görülme zamanı
        self._conf:  dict[str, float] = {}   # label → son güven skoru

    def update(self, label: str, confidence: float):
        self._seen[label] = time.perf_counter()
        self._conf[label] = confidence
        if len(self._seen) > self._max:
            oldest = min(self._seen, key=self._seen.get)
            self._seen.pop(oldest, None)
            self._conf.pop(oldest, None)

    def get_weight(self, label: str) -> float:
        if label not in self._seen:
            return 0.0
        age = time.perf_counter() - self._seen[label]
        return float(self._conf[label]) * (2.0 ** (-age / max(0.001, self._half_life)))

    def cleanup(self, cutoff_multiplier: float = 5.0):
        """Yarılanma ömrünün N katı kadar eski kayıtları sil."""
        limit = time.perf_counter() - self._half_life * cutoff_multiplier
        stale = [k for k, t in self._seen.items() if t < limit]
        for k in stale:
            self._seen.pop(k, None)
            self._conf.pop(k, None)

    def reset(self):
        self._seen.clear()
        self._conf.clear()


# ---------------------------------------------------------------------------
# Yardımcı Sınıf 2 — Adaptif Tahliye Kontrolörü
# ---------------------------------------------------------------------------

class PurgeController:
    """
    Sahne kesimi veya baskın skor atlamasında tahliye fanını 1.5 sn tetikler.
    Çıktı: purge_speed (0.0 - 1.0) — 8. seri kanal olarak iletilir.
    """

    def __init__(self, duration_s: float = 1.5):
        self._duration     = duration_s
        self._active_until = 0.0
        self._prev_dom     = 0.0

    def check(
        self,
        scores: dict[str, float],
        is_scene_cut: bool,
        jump_threshold: float = 0.50,
    ) -> float:
        now = time.perf_counter()
        dom = max(scores.values()) if scores else 0.0

        if is_scene_cut or abs(dom - self._prev_dom) >= jump_threshold:
            self._active_until = now + self._duration

        self._prev_dom = dom
        return 1.0 if now < self._active_until else 0.0

    def reset(self):
        self._active_until = 0.0
        self._prev_dom     = 0.0


# ---------------------------------------------------------------------------
# Yardımcı Sınıf 3 — Koku Körlüğü (Saturation) Filtresi
# ---------------------------------------------------------------------------

class SaturationTracker:
    """
    Uzun süreli yüksek maruziyeti tespit eder ve donanım yoğunluğunu düşürür.
    UI skorları tam gösterilir; hardware'e attenuation çarpanları ayrıca iletilir.

    Algoritma:
        - Her kanal için son `window_seconds` içindeki skor geçmişi tutulur.
        - Yüksek eşik (high_threshold) üzerinde geçen kare oranı %80'i aşarsa
          attenuation kademeli olarak min_output_ratio'ya kadar düşer.
        - Eşik altında kalındığında attenuation yarım hızda toparlanır.
    """

    def __init__(
        self,
        window_s:   float = 60.0,
        high_thr:   float = 0.70,
        min_ratio:  float = 0.40,
        fade_rate:  float = 0.008,
    ):
        self._window    = window_s
        self._high_thr  = high_thr
        self._min_ratio = min_ratio
        self._fade_rate = fade_rate
        self._history:     dict[str, list] = {}
        self._attenuation: dict[str, float] = {}

    def update_and_apply(
        self, scores: dict[str, float]
    ) -> tuple[dict[str, float], dict[str, float]]:
        """
        Döndürür:
            ui_scores   — UI'da tam gösterilecek skorlar (değişmez)
            attenuation — donanıma uygulanacak çarpanlar {scent: 0.0-1.0}
        """
        now = time.perf_counter()
        cutoff = now - self._window

        for scent, score in scores.items():
            buf = self._history.setdefault(scent, [])
            buf.append((now, score))
            self._history[scent] = [(t, s) for t, s in buf if t > cutoff]

            high_n    = sum(1 for _, s in self._history[scent] if s >= self._high_thr)
            sat_ratio = high_n / max(1, len(self._history[scent]))

            att = self._attenuation.get(scent, 1.0)
            if sat_ratio > 0.80:
                att = max(self._min_ratio, att - self._fade_rate)
            else:
                att = min(1.0, att + self._fade_rate * 0.5)
            self._attenuation[scent] = att

        return dict(scores), dict(self._attenuation)

    def reset(self):
        self._history.clear()
        self._attenuation.clear()


# ---------------------------------------------------------------------------

class ProbabilisticDecisionMatrix:
    """
    Üç bağımsız güven modülünü ağırlıklı matris ile birleştirir:
        1. YOLO Modülü      — nesne tespiti güven skoru
        2. K-Means Modülü   — baskın renk güven skoru
        3. Doku Modülü      — OpenCV Laplacian + HSV doku analizi

    Ardından:
        4. Context Processor — bağlamsal mantık süzgeci
        5. Histerezis        — aç/kapa kararsızlığını önler
        6. N-Kare Oylama     — geçici gürültü filtreleme
        7. Kalman Filtresi    — donanıma stabil çıktı
    """

    def __init__(self, config_path: str | Path = CONFIG_PATH):
        self.config_path = Path(config_path)
        self._cfg: dict = {}
        self._cfg_lock   = threading.RLock()   # KRİTİK-2: hot-reload race condition
        self._load_config()

        # Histerezis: her kanal için aktif/pasif durumu
        self._hysteresis_state: dict[str, bool] = {s: False for s in SCENTS}
        # VLM mask geçiş takibi — bypass/strict-recovery anında Kalman sıfırlamak için
        # Kural: SOFT/HARD mask (ütfm kanallar <0.5) → 1.0 geçişinde P=1.0 reset
        self._prev_mask_was_restricted: bool = False

        # N-kare oylama tamponu: kanal başına bool deque
        vw = self._cfg["temporal"]["voting_window"]
        self._vote_buffer: dict[str, deque] = {s: deque(maxlen=vw) for s in SCENTS}

        # Optik akış için önceki küçük gri kare
        self._prev_flow_gray: np.ndarray | None = None

        # Yardımcı modüller — config parametreleriyle başlatılır
        tm_cfg  = self._cfg.get("temporal_memory",   {})
        sat_cfg = self._cfg.get("saturation_tracker", {})
        pur_cfg = self._cfg.get("purge_controller",   {})

        self._memory = TemporalMemory(
            half_life   = float(tm_cfg.get("half_life_seconds", 2.0)),
            max_objects = int(tm_cfg.get("max_objects", 30)),
        )
        self._saturation = SaturationTracker(
            window_s  = float(sat_cfg.get("window_seconds",   60.0)),
            high_thr  = float(sat_cfg.get("high_threshold",   0.70)),
            min_ratio = float(sat_cfg.get("min_output_ratio", 0.40)),
            fade_rate = float(sat_cfg.get("fade_rate",        0.008)),
        )
        self._purge_ctrl = PurgeController(
            duration_s = float(pur_cfg.get("scene_cut_duration_s", 1.5)),
        )

        # OVL modülü — CLIP zero-shot, CUDA float16, async arka plan
        self._ovl_pipe            = None   # True = yuklu, None = devre disi
        self._ovl_model           = None
        self._ovl_processor       = None
        self._ovl_text_features   = None
        self._ovl_text_inputs     = {}     # GPU'da pre-tokenize metin girdileri
        self._ovl_device          = "cpu"
        self._ovl_validation      = {"water": 0.5, "outdoor": 0.5, "terrestrial": 0.5, "aquatic": 0.5}  # CLIP capraz dogrulama
        self._sf_active           = False   # güvenlik tabanı durum takibi (loglama)
        self._ovl_busy            = False
        self._ovl_run_lock        = threading.Lock()  # KRİTİK-1: atomik thread baskısı
        self._ovl_lock            = threading.Lock()
        self._kmeans_prev_centers: np.ndarray | None = None  # ORTA-1: warm start
        self._ovl_cache: dict[str, float] = {s: 0.0 for s in SCENTS}
        self._ovl_confidence      = 0.0
        self._ovl_frame_counter   = 0
        self._ovl_debug: dict      = {}   # per-prompt ham skor detayı (debug panel)
        self._vcv_prev_urban: float = 0.5  # VCV: önceki kare kentsel sinyal (log throttle)
        # Warmup Lock: CLIP ilk geçerli inference üretene kadar PWM çıkışı sıfır
        self._clip_ever_run: bool  = False
        self._init_ovl()

        # ── V3 Katman D: Kalman Filtre Bankası (EMA'nın yerini alır) ─────────
        _kal_params = self._cfg.get("kalman_filters", {})
        self._kalman = KalmanFilterBank(
            params={k: v for k, v in _kal_params.items() if not k.startswith("_")}
        )
        _log.info("[V3] KalmanFilterBank başlatıldı (%d kanal)", len(SCENTS))

        # ── V3 Katman C: Bayes Log-Uzayı Füzyon Motoru ───────────────────────
        _mod_w = {
            k: v for k, v in self._cfg.get("module_weights", {}).items()
            if not k.startswith("_")
        }
        self._bayes = BayesFuser(module_weights=_mod_w if _mod_w else None)
        _log.info("[V3] BayesFuser başlatıldı — ağırlıklar: %s", self._bayes.module_weights)

        # ── V3 Katman B: VLM Semantik Kilit (ContextAnalyzerThread) ──────────
        _vlm_cfg = self._cfg.get("vlm", {})
        self._vlm: ContextAnalyzerThread | None = None
        if _vlm_cfg.get("enabled", True):
            self._vlm = ContextAnalyzerThread(
                ollama_url       = str(_vlm_cfg.get("ollama_url", "http://localhost:11434")),
                model            = str(_vlm_cfg.get("model", "moondream2")),
                interval_frames  = int(_vlm_cfg.get("interval_frames", 20)),
                fallback_timeout = float(_vlm_cfg.get("fallback_timeout_seconds", 10.0)),
                max_retries      = int(_vlm_cfg.get("max_retries", 2)),
                enabled          = True,
            )
            self._vlm.start()
            _log.info(
                "[V3] ContextAnalyzerThread başladı — model=%s interval=%d kare",
                self._vlm.model, self._vlm.interval_frames,
            )
        else:
            _log.info("[V3] VLM semantik kilit devre dışı (config)")

        # VLM son semantik maske (metadata için)
        self._last_semantic_mask: dict[str, float] = dict(DEFAULT_MASK)

    # ------------------------------------------------------------------
    # Config yönetimi
    # ------------------------------------------------------------------
    def _load_config(self):
        with self._cfg_lock:
            with open(self.config_path, "r", encoding="utf-8") as f:
                self._cfg = json.load(f)

    def reload_config(self):
        """Çalışma sırasında config.json'ı yeniden yükler — hot-reload."""
        self._load_config()  # _load_config zaten _cfg_lock alıyor (RLock → reentrant)
        with self._cfg_lock:
            cfg = dict(self._cfg)
        # V3 hot-reload: Kalman Q/R parametreleri
        kal_params = {k: v for k, v in cfg.get("kalman_filters", {}).items()
                      if not k.startswith("_")}
        if kal_params:
            self._kalman.reload_params(kal_params)
        # V3 hot-reload: BayesFuser modül ağırlıkları
        mod_w = {k: v for k, v in cfg.get("module_weights", {}).items()
                 if not k.startswith("_")}
        if mod_w:
            self._bayes.reload_weights(mod_w)

    # ------------------------------------------------------------------
    # Modül 1 — YOLO Güven Skoru
    # ------------------------------------------------------------------
    def _yolo_module_scores(
        self, detections: list[dict]
    ) -> tuple[dict[str, float], dict[str, float]]:
        scores: dict[str, float]      = {s: 0.0 for s in SCENTS}
        penalty_log: dict[str, float] = {}
        for det in detections:
            label = det["label"].lower()
            conf  = float(det.get("confidence", 0.0))
            area  = float(det.get("area_ratio", 0.05))
            area_w = min(1.0, area * 8.0)
            mapping = OBJECT_SCENT_MAP.get(label)
            if not mapping:
                continue
            # Güven+alan sıkılaştırması: conf<0.55 VE ekran alanı<%5 → %80 sönümleme
            if conf < 0.55 and area < 0.05:
                penalty = 0.20
                penalty_log[label] = round(penalty, 2)
            else:
                penalty = 1.0
            for scent, w in mapping.items():
                scores[scent] += conf * w * area_w * penalty

        total = sum(scores.values())
        if total > 0:
            scores = {k: min(1.0, v / total) for k, v in scores.items()}
        return scores, penalty_log

    # ------------------------------------------------------------------
    # Modül 2 — K-Means Renk Güven Skoru
    # ------------------------------------------------------------------
    def _kmeans_module_scores(
        self, colors_bgr: list[tuple]
    ) -> tuple[dict[str, float], int]:
        scores: dict[str, float] = {s: 0.0 for s in SCENTS}
        neutral_skipped = 0
        if not colors_bgr:
            return scores, neutral_skipped

        for rank, bgr in enumerate(colors_bgr):
            rank_w = 1.0 / (rank + 1)
            pixel  = np.uint8([[list(bgr)]])
            hsv    = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0][0]
            h = float(hsv[0]) * 2.0   # 0-360
            s = float(hsv[1])
            v = float(hsv[2])

            # Nötr renk ön-filtresi: S<%15 (s<38) veya V<%15 (v<38)
            # Siyah / beyaz / gri tonları koku kuralı tetiklemesin
            if s < 38.0 or v < 38.0:
                neutral_skipped += 1
                continue

            for rule in _COLOR_RULES:
                h_min, h_max, s_min, v_min = rule["range"]
                if h_min <= h <= h_max and s >= s_min and v >= v_min:
                    for scent, w in rule["scents"].items():
                        scores[scent] += w * rank_w
                    break  # ilk eşleşen kural kazanır

        total = sum(scores.values())
        if total > 0:
            scores = {k: min(1.0, v / total) for k, v in scores.items()}
        return scores, neutral_skipped

    # ------------------------------------------------------------------
    # Modül 2b — K-Means Tamer (Dinamik Ağırlıklandırma)
    # ------------------------------------------------------------------
    # K-Means'in yüksek FPS avantajını korur; CLIP/VLM ile çeliştiği
    # durumlarda kanal bazında ağırlığını dinamik olarak ölçekler.
    #
    # Matematiksel temel (log-uzayı eşdeğeri):
    #   w_dyn * log(val) = w_base * log(val^(w_dyn/w_base))
    # ⇒ kmeans_tamed[s] = val^(w_dyn/w_base)
    # BayesFuser log-sum-exp normalizasyonu farkı emer; mutlak ölçek önemsiz.
    # ------------------------------------------------------------------
    def _kmeans_tamer(
        self,
        kmeans_s:      dict[str, float],
        semantic_mask: dict[str, float],
        ovl_scores:    dict[str, float],
        clip_conf:     float,
        vlm_conf:      float = 0.0,
    ) -> tuple[dict[str, float], str, dict[str, float]]:
        """
        Returns:
            (kmeans_tamed, mode_name, dynamic_weights)
        clip_conf: CLIP max kanalı güveni (self._ovl_confidence)
        vlm_conf:  VLM Ollama JSON 'confidence' alanı (ContextAnalyzerThread.vlm_confidence)
        """
        base_w = float(self._bayes.module_weights.get("kmeans", 0.15))
        dyn_w  = {s: base_w for s in SCENTS}
        mode   = "Normal"

        # ── Anahtar eşleşme doğrulaması (ilk çağrıda tek seferlik) ──────────────
        if not getattr(self, "_tamer_keys_ok", False):
            _keys = set(ovl_scores.keys())
            _need = {"\u0130s/Duman", "\u00c7am", "Toprak"}
            _miss = _need - _keys
            if _miss:
                _log.error("[K-Means Tamer] ANAHTAR ESLESME HATASI — ovl_scores eksik: %s "
                           "| Mevcut anahtarlar: %s", _miss, sorted(_keys))
            else:
                _log.info("[K-Means Tamer] Anahtar dogrulama OK — "
                          "Is/Duman:%.3f Cam:%.3f Toprak:%.3f clip_conf:%.3f vlm_conf:%.3f",
                          ovl_scores.get("\u0130s/Duman", 0.0),
                          ovl_scores.get("\u00c7am", 0.0),
                          ovl_scores.get("Toprak", 0.0),
                          clip_conf, vlm_conf)
            self._tamer_keys_ok = True

        # Urban tespiti: compute_semantic_mask() kentsel kapıları koydu mu?
        _is_urban  = (
            semantic_mask.get("Vanilya", 1.0) <= 0.015 and
            semantic_mask.get("Nane",    1.0) <= 0.055
        )

        # Aksiyon/Sinema: CLIP Is/Duman > 0.30 → sarı piksel yanlış eşleme tehlikesi
        _is_action = ovl_scores.get("\u0130s/Duman", 0.0) > 0.30

        # Kurtarıcı Modu: VLM kör veya kentsel dedi ama CLIP doğal sahne yakaladı
        _clip_strong_nature = (
            ovl_scores.get("\u00c7am",    0.0) > 0.25 or
            ovl_scores.get("Toprak", 0.0) > 0.25
        )
        _is_savior = (
            (vlm_conf == 0.0  and _clip_strong_nature) or  # VLM güvensiz (Ollama conf=0)
            (_is_urban        and _clip_strong_nature)      # VLM kentsel ama CLIP orman görüyor
        )

        # Öncelik: Aksiyon > Kurtarıcı > Kentsel > Normal
        # Gerekçe: sahne yoğun İs/Duman taşıyorsa VLM güveni 0 bile olsa aksiyon sahnesidir.
        if _is_action:
            mode = "Aksiyon"
            for s in SCENTS:
                if s != "\u0130s/Duman":
                    dyn_w[s] = 0.01  # Sarı/turuncu pikseller yanlış koku üretmesin
        elif _is_savior:
            mode = "Kurtarıcı"
            for s in ("\u00c7am", "Toprak", "Deniz"):
                dyn_w[s] = 0.45   # Renk katmanı sistemi sırtlasın
            # ── Emniyet Kilidi: Kurtarıcı modda bile yüksek İs/Duman varsa gurme koku sızmasın
            # Priority inversion guard — _is_action=False iken bu kola ulaşılır;
            # threshold değişse bile savunma katı korunur.
            if ovl_scores.get("\u0130s/Duman", 0.0) > 0.30:
                dyn_w["Vanilya"] = 0.01
                dyn_w["Nane"]    = 0.01
                mode = "Kurtarıcı+Emniyet"
        elif _is_urban:
            mode = "Kentsel"
            dyn_w["Vanilya"] = 0.02  # Bej bina Vanilya → sustur
            dyn_w["Nane"]    = 0.04  # Kaldırım ağacı Nane → sustur

        # Epsilon Padding: sıfır kanallar power-scaling'de 0^x=0 → Bayes trap oluşturur.
        # Tüm sıfırlara 0.01 taban ekle → CLIP sinyali bu kanalları absorbe edebilsin.
        _PAD = 0.01
        _eps = 1e-9
        padded = {s: max(_PAD, float(kmeans_s.get(s, 0.0))) for s in SCENTS}
        # Padding sonrası yeniden normalize et (PDF toplamı ≈ 1 korusun)
        _pad_total = sum(padded.values())
        if _pad_total > _eps:
            padded = {s: v / _pad_total for s, v in padded.items()}

        # ── Multiplicative Zero-Masking (Fix-1) ─────────────────────────────────
        # w=0.01 → ratio=0.0667 → padded^0.0667 ≈ 0.74: log-uzayında kanalı
        # düzleştirir, SIFIRLAMAZ (etkisiz elaman 1.0'a yaklaşır).
        # _VETO_W_THRESH'in altındaki kanallar → _VETO_EPS=1e-9 (log≈-20):
        # BayesFuser log-sumda bu kanalı gerçekten baskılar → sızıntı yok.
        _VETO_W_THRESH = 0.015  # dyn_w < bu değer → VETO (0.01 tüm VETO ağırlıkları < 0.015)
        _VETO_EPS      = 1e-9   # log(1e-9) ≈ -20 → log-uzayı gerçek sıfır maskesi

        tamed: dict[str, float] = {}
        for s in SCENTS:
            if dyn_w[s] < _VETO_W_THRESH:
                tamed[s] = _VETO_EPS   # Multiplicative zero mask: Bayes'e sıfır girdi
            else:
                ratio    = dyn_w[s] / base_w if base_w > _eps else 1.0
                tamed[s] = max(_eps, padded[s]) ** ratio

        return tamed, mode, dyn_w

    # ------------------------------------------------------------------
    # Modül 3 — Doku Analizi Güven Skoru
    # Laplacian varyansı (zenginlik) + HSV kanal istatistikleri
    # ------------------------------------------------------------------
    def _texture_module_scores(self, frame: np.ndarray) -> tuple[dict[str, float], float]:
        scores: dict[str, float] = {s: 0.0 for s in SCENTS}

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Doku zenginliği: yüksek Laplacian varyansı = ayrıntılı yüzey
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        richness = min(1.0, lap_var / 800.0)   # 0→düz alan, 1→zengin doku

        hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h_mu  = float(np.mean(hsv[:, :, 0])) * 2.0   # 0-360
        s_mu  = float(np.mean(hsv[:, :, 1]))
        v_mu  = float(np.mean(hsv[:, :, 2]))

        # Yeşil + zengin doku → orman/çam yaprağı
        if 80 <= h_mu <= 160 and richness > 0.35:
            scores["Çam"]   += richness * 0.75
            scores["Toprak"] += richness * 0.25

        # Mavi + düzgün yüzey → su/deniz
        if 170 <= h_mu <= 260 and s_mu > 60 and richness < 0.4:
            scores["Deniz"] += (1.0 - richness) * 0.70

        # Karanlık + düşük doygunluk → is/duman/gece
        if v_mu < 60 and s_mu < 55:
            scores["İs/Duman"] += (1.0 - richness) * 0.60

        # Turuncu/kahve + zengin doku → toprak, kaya
        if 15 <= h_mu <= 45 and richness > 0.45:
            scores["Toprak"]   += richness * 0.65
            scores["İs/Duman"] += richness * 0.20

        # Sarı + düzgün + yüksek doygunluk → limon/narenciye
        # H≤ 62°: sarı-yeşil tonlar (62-70°) hariç; S>130 (≥51%): çimen doygunluğu elenir
        if 35 <= h_mu <= 62 and s_mu > 130 and richness < 0.35:
            scores["Limon"] += 0.55

        # Beyaz/aydınlık + düzgün → vanilya/iç mekan
        if v_mu > 180 and s_mu < 60 and richness < 0.30:
            scores["Vanilya"] += 0.50

        total = sum(scores.values())
        if total > 0:
            scores = {k: v / total for k, v in scores.items()}
        return scores, lap_var

    # ------------------------------------------------------------------
    # Bağlamsal İşlemci (Context Processor)
    # Ham skorlara uygulanır — son mantık süzgeci
    # ------------------------------------------------------------------
    def _context_processor(
        self,
        scores: dict[str, float],
        frame: np.ndarray,
        detections: list[dict],
    ) -> dict[str, float]:
        result   = dict(scores)
        det_lbls = {d["label"].lower() for d in detections}
        rules    = self._cfg.get("context_rules", {})

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h_mu = float(np.mean(hsv[:, :, 0])) * 2.0
        s_mu = float(np.mean(hsv[:, :, 1]))

        for rule in rules.values():
            if not (det_lbls & set(rule.get("trigger_labels", []))):
                continue

            cond = rule.get("color_condition", "")
            if cond == "orange_grey":
                color_ok = (20 <= h_mu <= 45 and s_mu > 80) or s_mu < 50
            elif cond == "blue_dominant":
                color_ok = 170 <= h_mu <= 260
            elif cond == "green_dominant":
                color_ok = 80 <= h_mu <= 160 and s_mu > 60
            else:
                color_ok = True

            if not color_ok:
                continue

            strength  = float(rule.get("override_strength", 0.5))
            tgt_scent = rule.get("override_scent")
            if tgt_scent and tgt_scent in result:
                result[tgt_scent] = min(1.0, result[tgt_scent] + strength)
            for sup in rule.get("suppress_scents", []):
                if sup in result:
                    result[sup] *= max(0.0, 1.0 - strength * 0.85)

        total = sum(result.values())
        if total > 0:
            result = {k: v / total for k, v in result.items()}
        return result

    # ------------------------------------------------------------------
    # Histerezis Kapısı
    # Aç  → skor >= activate_threshold
    # Kapa → skor <  deactivate_threshold
    # Ara  → önceki durum korunur (salınım engellenir)
    # ------------------------------------------------------------------
    def _apply_hysteresis(
        self,
        scores:     dict[str, float],
        raw_scores: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """raw_scores verilirse STATE kararları orada; scores ise çıktı değerleri için kullanılır.
        raw_scores = blended_4way (Bayes posterior snapshot, post-işlem öncesi) → latch riskini önler."""
        act   = self._cfg["hysteresis"]["activate_threshold"]
        deact = self._cfg["hysteresis"]["deactivate_threshold"]
        ref   = raw_scores if raw_scores is not None else scores
        result: dict[str, float] = {}
        for scent, score in scores.items():
            raw = ref.get(scent, score)
            if raw >= act:
                self._hysteresis_state[scent] = True
            elif raw < deact:
                self._hysteresis_state[scent] = False
            # Pasif kanalı tamamen sıfırlama: gradyan bırak
            result[scent] = score if self._hysteresis_state[scent] else score * float(self._cfg["hysteresis"].get("passive_decay_factor", 0.45))
        return result

    # ------------------------------------------------------------------
    # N-Kare Oylama  (Temporal Consensus)
    # ------------------------------------------------------------------
    def _update_voting_buffer(self, scores: dict[str, float]):
        act = self._cfg["hysteresis"]["activate_threshold"]
        for scent, score in scores.items():
            self._vote_buffer[scent].append(score >= act)

    def _get_voted_scores(self, scores: dict[str, float]) -> dict[str, float]:
        vt = self._cfg["temporal"]["voting_threshold"]
        result: dict[str, float] = {}
        for scent, score in scores.items():
            buf = self._vote_buffer[scent]
            vote_ratio = sum(buf) / len(buf) if buf else 0.0
            if vote_ratio >= vt:
                result[scent] = score               # güçlü çoğunluk → tam sinyal
            else:
                result[scent] = score * vote_ratio  # zayıf çoğunluk → orantılı baskı
        return result

    # ------------------------------------------------------------------
    # Yeşil Piksel Oran Analizi — ROI frame HSV maskesi
    # ------------------------------------------------------------------
    def _compute_green_ratio_frame(self, frame: np.ndarray) -> float:
        """Karenin yeşil piksel oranı (HSV h=55-150, s>=20, v>=30)."""
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h    = hsv[:, :, 0].astype(np.float32) * 2.0  # 0-360
        s    = hsv[:, :, 1].astype(np.float32)
        v    = hsv[:, :, 2].astype(np.float32)
        mask = ((h >= 55) & (h <= 150) & (s >= 20) & (v >= 30))
        return float(mask.sum()) / max(1, mask.size)

    # ------------------------------------------------------------------
    # CLAHE Ön İşleme — ışık normalizasyonu
    # ------------------------------------------------------------------
    def apply_clahe(self, frame: np.ndarray) -> np.ndarray:
        c = self._cfg.get("clahe", {"clip_limit": 2.0, "tile_grid_size": [8, 8]})
        lab  = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=float(c["clip_limit"]),
            tileGridSize=tuple(c["tile_grid_size"]),
        )
        lab_eq = cv2.merge([clahe.apply(l), a, b])
        return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    # ------------------------------------------------------------------
    # Yardımcı: ROI K-Means (sklearn, bounding-box veya merkez kırpma)
    # ------------------------------------------------------------------
    def _run_kmeans_on_roi(
        self, roi_frame: np.ndarray, k: int = 5
    ) -> list[tuple]:
        """
        cv2.kmeans (C++ implementasyonu) — sklearn'dan ~3-5x hızlı.
        Warm start: önceki merkez varsa piksel→en yakın atama → tek geçiş yeterli.
        """
        pixels = roi_frame.reshape(-1, 3).astype(np.float32)
        if len(pixels) < k:
            return []
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
        prev = self._kmeans_prev_centers
        if prev is not None and prev.shape == (k, 3):
            # Warm start: her piksel → en yakın önceki merkeze atanır
            diff        = pixels[:, None, :] - prev[None, :, :]   # (N, k, 3)
            # .copy() zorunlu: cv2.kmeans writeable C-contiguous int32 array bekler
            init_labels = (diff * diff).sum(axis=2).argmin(axis=1).astype(np.int32).reshape(-1, 1).copy()
            _, labels, centers = cv2.kmeans(
                pixels, k, init_labels, criteria, 1, cv2.KMEANS_USE_INITIAL_LABELS
            )
        else:
            _, labels, centers = cv2.kmeans(
                pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS
            )
        self._kmeans_prev_centers = centers.copy()
        counts = np.bincount(labels.flatten(), minlength=k)
        order  = np.argsort(counts)[::-1]
        return [tuple(centers[i].astype(int)) for i in order]

    # ------------------------------------------------------------------
    # Dinamik ROI: YOLOv8 tespitine odaklan, yoksa merkezi kırp
    # ------------------------------------------------------------------
    def _dynamic_roi_crop(
        self, frame: np.ndarray, detections: list[dict]
    ) -> np.ndarray:
        roi_cfg     = self._cfg.get("dynamic_roi", {})
        focus_lbls  = set(roi_cfg.get("focus_labels", []))
        h, w        = frame.shape[:2]

        for det in detections:
            if det["label"].lower() in focus_lbls:
                x1, y1, x2, y2 = det["box"]
                crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                if crop.size > 0:
                    return crop

        ratio  = float(roi_cfg.get("fallback_center_ratio", 0.60))
        pad_h  = int(h * (1 - ratio) / 2)
        pad_w  = int(w * (1 - ratio) / 2)
        return frame[pad_h: h - pad_h, pad_w: w - pad_w]

    # ------------------------------------------------------------------
    # Optik Akış Güven Skoru
    # Mavi bölge yatay hareket → Deniz bonus
    # Gri bölge yukarı hareket → İs/Duman bonus
    # Statik renk bölgesi      → baskılama
    # ------------------------------------------------------------------
    def _optical_flow_scores(self, frame: np.ndarray) -> dict[str, float]:
        scores: dict[str, float] = {s: 0.0 for s in SCENTS}
        cfg_of = self._cfg.get("optical_flow", {})
        min_mag = float(cfg_of.get("min_magnitude", 0.5))

        gray_small = cv2.resize(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (160, 90)
        )

        if self._prev_flow_gray is None:
            self._prev_flow_gray = gray_small
            return scores

        flow = cv2.calcOpticalFlowFarneback(
            self._prev_flow_gray, gray_small,
            None, 0.5, 3, 15, 3, 5, 1.2, 0,
        )
        self._prev_flow_gray = gray_small

        fx, fy = flow[..., 0], flow[..., 1]
        mag    = np.sqrt(fx ** 2 + fy ** 2)

        hsv_s = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), (160, 90))
        H = hsv_s[:, :, 0].astype(np.float32) * 2.0
        S = hsv_s[:, :, 1].astype(np.float32)

        # --- Mavi bölge: Deniz vs Gökyüzü ---
        blue_mask = (H >= 170) & (H <= 260) & (S > 60)
        if blue_mask.any():
            b_mag = float(np.mean(mag[blue_mask]))
            b_fx  = float(np.mean(np.abs(fx[blue_mask])))
            b_fy  = float(np.mean(np.abs(fy[blue_mask])))
            if b_mag >= min_mag and b_fx > b_fy:
                scores["Deniz"] += min(
                    float(cfg_of.get("water_horizontal_bonus", 0.25)),
                    b_mag * 0.12,
                )
            elif b_mag < min_mag:
                scores["Deniz"] -= float(cfg_of.get("static_suppression", 0.15))

        # --- Gri/koyu bölge: Duman hareket yönü ---
        grey_mask = (S < 55) & (mag >= min_mag)
        if grey_mask.any():
            g_fy = float(np.mean(fy[grey_mask]))
            if g_fy < -float(cfg_of.get("smoke_upward_threshold", 0.30)):   # negatif fy = yukarı hareket
                scores["İs/Duman"] += min(
                    float(cfg_of.get("smoke_upward_bonus", 0.20)),
                    abs(g_fy) * 0.15,
                )

        return scores

    # ------------------------------------------------------------------
    # OVL Modülü — kordelfrance/ovlc-base  (Duyusal Hakem)
    # CUDA float16 üzerinde RTX 3050 optimizasyonu; async 30-kare döngüsü
    # ------------------------------------------------------------------
    def _init_ovl(self) -> None:
        """CLIP zero-shot modelini CUDA float16 ile başlat; hata durumunda sessizce devam et."""
        ovl_cfg  = self._cfg.get("ovl", {})
        if not ovl_cfg.get("enabled", True):
            _log.info("[OVL] Devre dışı (config)")
            return
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor

            dev_str = "cuda" if torch.cuda.is_available() else "cpu"
            hw_tag  = f"CUDA ({torch.cuda.get_device_name(0)})" if dev_str == "cuda" else "CPU"
            self._ovl_device = dev_str

            model_id = ovl_cfg.get("model_id", "openai/clip-vit-base-patch32")
            # float32 yükle — torch.cuda.amp.autocast() içerde float16'ya çevirir.
            # torch_dtype=float16 + autocast kombinasyonu LayerNorm çıktısını float32
            # yapar, sonraki float16 matmul patlatır.
            self._ovl_model     = CLIPModel.from_pretrained(model_id).to(dev_str)
            self._ovl_model.eval()
            self._ovl_processor = CLIPProcessor.from_pretrained(model_id)

            # Absorber Mimarisi — Paratoner Sınıflar
            # 3 prompt × (7 koku + 3 absorber) = 30 prompt + 6 doğrulama = 36 toplam
            # [0-2]   Çam   [3-5]  Deniz   [6-8]  Vanilya  [9-11]  İs/Duman
            # [12-14] Nane  [15-17] Toprak  [18-20] Limon
            # [21-23] ABSORBER: Gökyüzü/Bulut  → mavi gökyüzü Deniz'e gitmiyor
            # [24-26] ABSORBER: Kaya/Taş       → gri karalık İs/Duman'a gitmiyor
            # [27-29] ABSORBER: Bina/Şehir     → kentsel yapı diğer kokulara gitmiyor
            # [30-31] Su doğrulama  [32-33] Mekan  [34-35] Karasal/Sucul
            _prompts = [
                # Çam / Orman [0-2]
                "a forest path with pine trees and ferns",
                "dense green woodland with tall conifer trees",
                "aerial view of dark evergreen pine forest",
                # Deniz / Su [3-5]
                "a photo of the open sea, blue ocean waves, or a beach coastline",
                "a flowing river, rushing stream, creek, or a powerful splashing waterfall",
                "natural moving water body, turquoise rapid water currents with white foam",
                # Vanilya [6-8]
                "vanilla sweet dessert, warm bakery aroma",
                "freshly baked pastry or cake inside a cozy kitchen",
                "indoor dining with warm spices and candles",
                # İs / Duman [9-11]
                "urban cityscape with dense traffic, exhaust fumes and smog",
                "historical city buildings with coal smoke and air pollution",
                "industrial district with factory chimneys and smoky atmosphere",
                # Nane [12-14]
                "a close-up photo of fresh mint leaves or peppermint herb",
                "green spearmint plant with small oval serrated leaves",
                "a photo of mint plants growing in a herb garden or pot",
                # Toprak [15-17]
                "muddy ground or wet soil after rain",
                "rocky cliffs and dry mountain terrain",
                "earthy dirt road with stones and dust",
                # Limon [18-20]
                "citrus lemon grove with ripe yellow fruit",
                "orange or lime orchard under bright sunlight",
                "fresh tropical fruit market stall",
                # ABSORBER: Gökyüzü / Bulut [21-23]
                "a clear blue sky with white clouds and no ground or water",
                "bright sunny cloudscape, aerial view of clouds from above",
                "blue sky horizon with scattered clouds, no sea or lake below",
                # ABSORBER: Kaya / Taş [24-26]
                "gray rocky cliff face or rough stone wall texture up close",
                "bare granite rock formation, boulder in dry barren landscape",
                "rocky mountain terrain, gravel scree, gray stone ground",
                # ABSORBER: Bina / Şehir [27-29]
                "urban architecture, tall glass and concrete building facades",
                "city skyline with modern skyscrapers against blue sky",
                "concrete walls, paved street, building exterior in a city",
                # Su doğrulama [30-31]
                "water body, ocean, sea, lake, river, aquatic scene",
                "dry land, forest, mountain, urban, indoor, no water",
                # Mekan tespiti [32-33]
                "outdoor nature, trees, sky, mountain, grass, open field",
                "indoor room, furniture, ceiling, wall, building interior",
                # Karasal / Sucul doğa ayrımı [34-35]
                "rocky mountain forest inland landscape with trees and no water",
                "ocean sea beach river stream flowing water with current or waves",
            ]
            txt_in = self._ovl_processor(text=_prompts, return_tensors="pt", padding=True)
            self._ovl_text_inputs = {k: v.to(dev_str) for k, v in txt_in.items()}

            self._ovl_pipe = True  # sentinel: model yüklendi
            _log.info("[OVL] Model Yuklendi → %s | %s (CLIP zero-shot)", hw_tag, model_id)
        except Exception as exc:
            # KRİTİK-3: GPU tensörlerini açıkça serbest bırak
            if hasattr(self, '_ovl_text_inputs'):
                for _tv in self._ovl_text_inputs.values():
                    del _tv
                self.__dict__.pop('_ovl_text_inputs', None)
            _mdl_init = getattr(self, '_ovl_model', None)
            if _mdl_init is not None:
                del _mdl_init
            try:
                if torch is not None and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            self._ovl_pipe = None
            self._ovl_model = self._ovl_processor = self._ovl_text_features = None
            _log.warning("[OVL] Yukleme basarisiz (%s) — YOLO+K-Means moduna donuldu", exc.__class__.__name__)

    def _analyze_ovl(
        self, frame: np.ndarray
    ) -> tuple[dict[str, float], float, dict[str, float], dict]:
        """
        Çok-Katmanlı CLIP zero-shot analizi (224×224, float16).
        Döndürür: (scent_scores, max_confidence, validation, debug)
          27 prompt, tek forward pass; 3-prompt ortalama → 7-yollu softmax.
          [0-20] 3×7 koku  [21-22] su  [23-24] mekan  [25-26] karasal/sucul
        """
        try:
            small   = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)
            rgb     = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            pil_img = PILImage.fromarray(rgb)

            dev = self._ovl_device

            img_in = self._ovl_processor(images=pil_img, return_tensors="pt")
            pix    = img_in["pixel_values"].to(dev)   # float32 — autocast halleder

            inputs = dict(self._ovl_text_inputs)
            inputs["pixel_values"] = pix

            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=(dev == "cuda")):
                    out = self._ovl_model(**inputs)
                logits = out.logits_per_image.squeeze(0).cpu().float()

            # Absorber Mimarisi: 10-sınıf softmax (7 koku + 3 absorber)
            # Gökyüzü/Kaya/Bina sahnesinde absorber softmax kütlesini emer;
            # ilgisiz kokular (Deniz, İs/Duman) doğal olarak sıfırlanır.
            n_scents     = 7
            n_absorbers  = 3
            n_groups     = n_scents + n_absorbers   # 10
            temperature  = float(self._cfg.get("clip", {}).get("temperature", 0.7))

            all_groups   = [logits[i * 3: i * 3 + 3] for i in range(n_groups)]
            all_means    = torch.stack([g.mean() for g in all_groups])
            all_stds     = torch.stack([g.std()  for g in all_groups])
            coherence    = 1.0 / (1.0 + 2.0 * all_stds)   # yüksek std → düşük güven
            all_weighted = all_means * coherence
            all_probs    = (all_weighted / temperature).softmax(dim=0).numpy()  # 10-class

            # Absorber soyma: absorber olasılık kütlesini aldı, koku skorları düşmüş durumda
            scent_probs    = all_probs[:n_scents]     # [7] — gökyüzü/kaya/bina etkisi kaldırıldı
            absorber_probs = all_probs[n_scents:]     # [3] — Gökyüzü, Kaya, Bina
            coherence_np   = coherence.numpy()

            # Renorm YOK: absorber'ların yuttuğu olasılık kütlesi korunuyor.
            # Gökyüzü/%90 absorber → kalan 7 koku küçük ham skorlarla VCV'ye girer.
            # (Yeniden normalize etsek absorber cezasını geri vermiş olurduk.)

            # Validation probes — yeni indeksler [30-35]
            water_probs   = (logits[30:32] / temperature).softmax(dim=0).numpy()
            outdoor_probs = (logits[32:34] / temperature).softmax(dim=0).numpy()
            terr_aq_probs = (logits[34:36] / temperature).softmax(dim=0).numpy()

            scores   = {s: float(p) for s, p in zip(SCENTS, scent_probs)}
            max_conf = float(scent_probs.max())
            validation = {
                "water":       float(water_probs[0]),
                "outdoor":     float(outdoor_probs[0]),
                "terrestrial": float(terr_aq_probs[0]),
                "aquatic":     float(terr_aq_probs[1]),
            }
            debug = {
                "prompt_detail": {
                    s: [round(float(logits[i * 3 + j]), 3) for j in range(3)]
                    for i, s in enumerate(SCENTS)
                },
                "coherence": {
                    s: round(float(coherence_np[i]), 3)
                    for i, s in enumerate(SCENTS)
                },
                "absorbers": {
                    "sky":      round(float(absorber_probs[0]), 3),
                    "rock":     round(float(absorber_probs[1]), 3),
                    "building": round(float(absorber_probs[2]), 3),
                },
            }
            return scores, max_conf, validation, debug

        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                _log.warning("[OVL] VRAM yetersiz — model devre dışı bırakıldı")
                with self._ovl_lock:          # KRİTİK-1 + KRİTİK-3: lock altında temizlik
                    if hasattr(self, '_ovl_text_inputs'):
                        for _t in self._ovl_text_inputs.values():
                            del _t
                        self.__dict__.pop('_ovl_text_inputs', None)
                    _mdl = self._ovl_model
                    self._ovl_model = None
                    self._ovl_pipe  = None
                    if _mdl is not None:
                        del _mdl
                try:
                    if torch is not None and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
            return {s: 0.0 for s in SCENTS}, 0.0, {"water": 0.5, "outdoor": 0.5, "terrestrial": 0.5, "aquatic": 0.5}, {}
        except Exception:
            return {s: 0.0 for s in SCENTS}, 0.0, {"water": 0.5, "outdoor": 0.5, "terrestrial": 0.5, "aquatic": 0.5}, {}

    def _ovl_worker(self, frame: np.ndarray) -> None:
        """Arka plan thread hedefi — OVL analizi yap, cache'e yaz, VRAM temizle."""
        self._ovl_busy = True
        try:
            scores, confidence, validation, debug = self._analyze_ovl(frame)
            with self._ovl_lock:
                self._ovl_cache      = scores
                self._ovl_confidence = confidence
                self._ovl_validation = validation
                self._ovl_debug      = debug
                # KRİTİK-1: VRAM temizliği _ovl_lock içinde — race condition önler
                try:
                    if torch is not None and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass

            # Sağlık Raporu — terminal'e özet yaz
            try:
                if torch is not None and torch.cuda.is_available():
                    used  = torch.cuda.memory_allocated() / 1024 ** 2
                    total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 2
                    vram  = f"{used:.0f}/{total:.0f} MB"
                else:
                    vram = "CPU"
            except Exception:
                vram = "N/A"

            top_scent = max(scores, key=scores.get) if scores else "-"
            top_conf  = max(scores.values()) if scores else 0.0
            _log.debug(
                "[OVL-HEALTH] kare=%d | top=%s(%.2f) | VRAM=%s | conf>=thr=%s",
                self._ovl_frame_counter, top_scent, top_conf, vram,
                "EVET" if confidence >= self._cfg.get("ovl", {}).get("confidence_threshold", 0.70) else "HAYIR",
            )
        finally:
            self._ovl_busy = False
            # KRİTİK-1: lock'u serbest bırak — bir sonraki kare thread başlatabilir
            try:
                self._ovl_run_lock.release()
            except RuntimeError:
                pass  # zaten serbest ise ignore

    # ------------------------------------------------------------------
    # Modül G — Bölgesel Grid Analizi (Gökyüzü / Zemin Ayrımı)
    # Üst %30 → gökyüzü bölgesi: mavi yoğunsa Deniz'i cezalandır
    # Alt %70 → zemin bölgesi: yeşil/su yoğunsa kokuları doğrudan artır
    # ------------------------------------------------------------------
    def _grid_sky_penalty(
        self, frame: np.ndarray
    ) -> tuple[dict[str, float], float, float]:
        """
        Çevresel Filtre (VCV Mimarisi).
        Üst %30 = GÖKYÜZܑ bölgesi: yalnızca 'sky_freshness' float'ı üretir.
          • Mavi gökyüzü → açık alan sinyali (koku kimliği ATANAMAZ).
          • Eskideki Deniz cezası KALDIRILDI — vektörel sönümleme halleder.
        Alt %70 = ZEMİN bölgesi: fiziksel doku → koku kimliği sinyali.
          • Yeşil doku  → Çam + Toprak bonusu.
          • Su dokusu   → Deniz bonusu (ZEMINDEN geldiği için güvenilir).
        Döndürür: (delta_dict, sky_blue_ratio, sky_freshness)
        """
        h, w = frame.shape[:2]
        sky_zone    = cv2.resize(frame[:max(1, int(h * 0.30)), :], (160, 48))
        ground_zone = cv2.resize(frame[int(h * 0.30):, :],        (160, 112))

        deltas = {s: 0.0 for s in SCENTS}

        # — Gökyüzü: ferahlık sinyali (koku kimliği ataması YOK) —————————————
        sky_hsv         = cv2.cvtColor(sky_zone, cv2.COLOR_BGR2HSV).astype(np.float32)
        sky_h, sky_v    = sky_hsv[:, :, 0], sky_hsv[:, :, 2]
        sky_blue_mask   = (sky_h >= 90) & (sky_h <= 140) & (sky_v > 100)
        sky_blue_ratio  = float(np.mean(sky_blue_mask))
        # sky_freshness: açık alan / hava sinyali → _compute_context_axes'e aktarılır
        sky_freshness   = min(1.0, sky_blue_ratio * 1.5)

        # — Zemin: fiziksel doku → koku kimliği sinyali ———————————————————————
        gnd_hsv         = cv2.cvtColor(ground_zone, cv2.COLOR_BGR2HSV).astype(np.float32)
        gnd_h, gnd_s, gnd_v = gnd_hsv[:, :, 0], gnd_hsv[:, :, 1], gnd_hsv[:, :, 2]

        # Yeşil doku (çimen, yaprak, orman) → Çam + Toprak
        green_mask  = (gnd_h >= 35) & (gnd_h <= 85) & (gnd_s > 50) & (gnd_v > 40)
        green_ratio = float(np.mean(green_mask))
        if green_ratio > 0.20:
            deltas["Çam"]    += min(0.18, green_ratio * 0.50)
            deltas["Toprak"] += min(0.10, green_ratio * 0.20)

        # Su dokusu zemin bölgesinde (mavi-yeşil) → Deniz güvenilir kanıt
        water_mask  = (gnd_h >= 85) & (gnd_h <= 130) & (gnd_s > 40) & (gnd_v > 60)
        water_ratio = float(np.mean(water_mask))
        if water_ratio > 0.15:
            deltas["Deniz"] += min(0.20, water_ratio * 1.20)

        return deltas, sky_blue_ratio, sky_freshness

    # ------------------------------------------------------------------
    # Modül SF — Gökyüzü Yanlış Pozitif Kapısı (Sky False-Positive Gate)
    # Üst %40 mavi + düşük Laplacian → K-Means/Texture Deniz/Nane yanılgısını kapat.
    # Aquatik YOLO nesnesi varsa filtre devredışı kalır (gerçek su koruması).
    # ------------------------------------------------------------------
    _SKY_GATE_AQUATIC: frozenset = frozenset({"boat", "ship", "sailboat", "surfboard", "canoe"})

    def _sky_false_positive_gate(
        self,
        frame:      np.ndarray,
        blended:    dict[str, float],
        detections: list[dict],
    ) -> dict[str, float]:
        if any(d.get("label", "").lower() in self._SKY_GATE_AQUATIC for d in detections):
            return blended

        h, w    = frame.shape[:2]
        sky_raw = frame[:max(1, int(h * 0.40)), :]
        sky_sm  = cv2.resize(sky_raw, (160, 64))

        gray_sky = cv2.cvtColor(sky_sm, cv2.COLOR_BGR2GRAY)
        lap_var  = float(cv2.Laplacian(gray_sky, cv2.CV_64F).var())
        if lap_var >= 150.0:
            return blended

        hsv_sky  = cv2.cvtColor(sky_sm, cv2.COLOR_BGR2HSV).astype(np.float32)
        h_ch, s_ch, v_ch = hsv_sky[:, :, 0], hsv_sky[:, :, 1], hsv_sky[:, :, 2]
        blue_mask  = (h_ch >= 90) & (h_ch <= 140) & (s_ch >= 30) & (v_ch >= 100)
        blue_ratio = float(np.mean(blue_mask))
        if blue_ratio < 0.45:
            return blended

        strength = min(1.0, (blue_ratio - 0.45) / 0.30)   # 0→0.45 mavi, 1→0.75+ mavi
        deniz_mult = max(0.05, 1.0 - strength * 0.90)
        nane_mult  = max(0.10, 1.0 - strength * 0.75)
        result = dict(blended)
        result["Deniz"] = blended.get("Deniz", 0.0) * deniz_mult
        result["Nane"]  = blended.get("Nane",  0.0) * nane_mult
        _log.debug(
            "[SKY-GATE] blue=%.2f lap=%.1f → Deniz×%.2f Nane×%.2f",
            blue_ratio, lap_var, deniz_mult, nane_mult,
        )
        return result

    # ------------------------------------------------------------------
    # Modül F — Aydınlatma Profili Tespiti (HSV Profiling)
    # K-Means analizine girmeden önce karenin genel atmosferini tanımla.
    # NIGHT_DARK  → Value ortalaması çok düşük (karanlık ortam)
    # GOLDEN_HOUR → Hue sıcak tonlarda (turuncu/sarı) yoğunlaşmış
    # DAYLIGHT    → Diğer tüm durumlar
    # ------------------------------------------------------------------
    def _detect_lighting_profile(self, frame: np.ndarray) -> str:
        cfg_lp = self._cfg.get("lighting_profile", {})
        night_v_thresh   = float(cfg_lp.get("night_v_threshold",   60.0))
        golden_h_low     = float(cfg_lp.get("golden_h_low",         5.0))
        golden_h_high    = float(cfg_lp.get("golden_h_high",       35.0))
        golden_s_min     = float(cfg_lp.get("golden_s_min",        80.0))
        golden_ratio_min = float(cfg_lp.get("golden_ratio_min",     0.25))
        golden_mean_s    = float(cfg_lp.get("golden_mean_s_min",   60.0))

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        # Önce GOLDEN_HOUR: sıcak renkler varsa mean_V düşük olsa bile (karanlık silüet)
        # GOLDEN_HOUR döndür — NIGHT_DARK kontrolü sadece sıcak renk yoksa çalışsın
        warm_mask  = ((h >= golden_h_low) & (h <= golden_h_high)) & (s > golden_s_min)
        warm_ratio = float(np.mean(warm_mask))
        # mean_s ek koşulu kaldırıldı: karanlık silüetler mean_s'düşürür;
        # sıcak renk oranı eşiğin üzero, sahne kesinlikle GOLDEN_HOUR'dır.
        if warm_ratio > golden_ratio_min:
            return "GOLDEN_HOUR"

        # Yalınızca sıcak renk yoksa karanlık kontrolü yap
        mean_v = float(np.mean(v))
        if mean_v < night_v_thresh:
            return "NIGHT_DARK"

        return "DAYLIGHT"

    # ------------------------------------------------------------------
    # Yardımcı: Güvenli Normalizasyon (Normalizasyon Tuzağı Koruması)
    # Toplam min_total'ın altındaysa zorla 1'e tamamlama: motorlar yavaş
    # çalışsın veya kare atlanmış gibi düşük çıktı üretsin.
    # ------------------------------------------------------------------
    def _safe_normalize(
        self, scores: dict[str, float], min_total: float | None = None
    ) -> dict[str, float]:
        if min_total is None:
            min_total = float(
                self._cfg.get("color_veto", {}).get("min_total_to_normalize", 0.40)
            )
        total = sum(scores.values())
        # ── Fix-3: Zero-Sum Blackout Guard ───────────────────────────────────
        # Tüm kanallar sıfır/negatif → sıfır şablonu döndür; NaN/Inf/ZeroDivision yok.
        if total <= 1e-12:
            return {k: 0.0 for k in scores}
        if total >= min_total:
            return {k: v / total for k, v in scores.items()}
        return scores  # Düşük toplam → zorla şişirme; ham değerler korunur

    def _anti_inflate_normalize(
        self,
        scores:  dict[str, float],
        vetoed:  set[str],
        ambient: float | None = None,
    ) -> dict[str, float]:
        """VETO sonrası normalizasyon şişmesini (Anti-Normalization Inflation) engelle.

        Standart normalizasyon:  sum(kalan) → 1.0  (tek zayıf kanal %100'e fırlar)
        Anti-inflate:  denom = sum(kalan) + AMBIENT_FLOOR  → zayıf sinyal korunur

        Örnek (Fix-2):
            İs/Duman=0.05, diğerleri VETO → standart: 1.00  kötü
            ambient=0.20 → 0.05/(0.05+0.20) = 0.20            iyi
        """
        if ambient is None:
            ambient = float(
                self._cfg.get("veto", {}).get("ambient_floor", 0.20)
            )
        raw_total = sum(v for s, v in scores.items() if s not in vetoed)
        # Zero-sum blackout: tüm kanallar VETO veya sıfır → temiz sıfır şablonu
        if raw_total <= 1e-12:
            return {s: 0.0 for s in scores}
        denom = raw_total + ambient
        return {
            s: (v / denom if s not in vetoed else 0.0)
            for s, v in scores.items()
        }

    # ------------------------------------------------------------------
    # VCV-1 — Bağlam Ekseni Hesaplama
    # YOLO + CLIP sinyallerinden sürekli (0-1) kentsel/doğa ekseni türet.
    # ------------------------------------------------------------------
    def _compute_context_axes(
        self,
        detections:     list[dict],
        ovl_validation: dict[str, float],
        clip_has_run:   bool,
        ovl_conf:       float,
        sky_freshness:  float = 0.0,
    ) -> tuple[float, float]:
        """
        Sahnedeki kentsel/doğa bağlamını sürekli sinyal olarak hesapla.
        Döndürür: (urban_signal, nature_signal) — toplamı 1.0.
          YOLO kanıt sayısı + CLIP outdoor/water doğrulaması kombinasyonu.
          sky_freshness: açık gök → doğa sinyalini hafifçe artırır.
        """
        det_lbls = {d["label"].lower() for d in detections}

        # YOLO kanıt sayısı → normalize edilmiş ön sinyal
        urban_count  = sum(1 for lb in det_lbls if lb in _VCV_URBAN_LABELS)
        nature_count = sum(1 for lb in det_lbls if lb in _VCV_NATURE_LABELS)
        yolo_urban   = min(1.0, urban_count  * 0.35)
        yolo_nature  = min(1.0, nature_count * 0.30)

        # ORTA-3: ağırlıkları config'den oku (geriye uyumlu default değerler)
        vcv_cfg  = self._cfg.get("vcv", {})
        clip_w   = float(vcv_cfg.get("context_axis_clip_weight",   0.6))
        yolo_w   = float(vcv_cfg.get("context_axis_yolo_weight",   0.4))
        sky_w    = float(vcv_cfg.get("sky_freshness_weight",       0.08))

        # CLIP doğrulama sinyali (güven ağırlıklı)
        if clip_has_run and ovl_conf > 0.30:
            w           = min(1.0, ovl_conf * 1.5)   # güven arttıkça CLIP etkisi ağır basar
            clip_nature = (
                ovl_validation.get("outdoor", 0.5) * 0.55
                + ovl_validation.get("water",   0.5) * 0.45
            )
            clip_urban  = 1.0 - clip_nature

            # Büyük ekran koruması: tv/laptop alan>%40 → CLIP kesin kentsel çıkarımını sınırla
            # (ekranın içeriği gerçek sahneyi yansıtmıyor olabilir)
            _large_screen = any(
                d.get("label", "").lower() in _VCV_SCREEN_LABELS
                and float(d.get("area_ratio", 0.0)) > 0.40
                for d in detections
            )
            if _large_screen:
                clip_urban  = min(clip_urban,  0.55)   # max %55 kentsel sinyal
                clip_nature = 1.0 - clip_urban          # yeniden dengele

            urban_raw   = (1 - w) * yolo_urban  + w * (yolo_w * yolo_urban  + clip_w * clip_urban)
            nature_raw  = (1 - w) * yolo_nature + w * (yolo_w * yolo_nature + clip_w * clip_nature)
        else:
            urban_raw  = yolo_urban
            nature_raw = yolo_nature

        # Açık gök → hafif doğa yönelimi (ferahlık, açık alan hissi)
        nature_raw = min(1.0, nature_raw + sky_freshness * sky_w)

        # Normalize: toplam = 1.0
        total = urban_raw + nature_raw
        if total < 0.01:
            return 0.5, 0.5   # belirsiz sahne → nötr eksen
        return urban_raw / total, nature_raw / total

    # ------------------------------------------------------------------
    # VCV-2 — Vektörel Bağlam Sönümleme (Weight Decay)
    # Kural tabanlı sabit çarpanların (0.10 / 0.05 / 0.15) yerini alır.
    # Tüm kural bloklarının matematiksel birleşimi: tek geçişte hesaplanır.
    # ------------------------------------------------------------------
    def _vector_context_decay(
        self,
        blended:       dict[str, float],
        urban_signal:  float,
        nature_signal: float,
        lam:           float | None = None,
    ) -> dict[str, float]:
        """
        Anlamsal eksen çatışmasına göre üstel sönümleme.

        alignment  = p_nature*nature_sig + p_urban*urban_sig   (iç çarpım)
        conflict   = 1.0 - alignment
        decay      = exp(-λ * conflict)

        Örnekler (λ=2.5):
          • Kentsel sahne (u=0.8) + Çam (p_nat=0.92): decay ≈ 0.15  → güçlü ceza
          • Kentsel sahne (u=0.8) + Vanilya (p_urb=0.90): decay ≈ 0.52 → hafif ceza
          • Doğa sahnesi (n=0.9) + Vanilya (p_urb=0.90): decay ≈ 0.13  → güçlü ceza
          • Nötr sahne   (0.5/0.5): tüm kokular aynı decay ≈ 0.29   → oran korunur

        Normalizasyon adımı otomatik olarak "akışkan yeniden dağıtım" sağlar:
        sönümlenen kütlesi bağlam-uyumlu kokulara orantılı geçer.
        """
        # ORTA-3: lambda config'den oku (varsayılan 2.5 — davranış değişmez)
        if lam is None:
            lam = float(self._cfg.get("vcv", {}).get("lambda", 2.5))
        _hardening_thr   = float(self._cfg.get("vcv", {}).get("urban_hardening_threshold", 0.60))
        _hardening_scale = float(self._cfg.get("vcv", {}).get("urban_hardening_scale", 5.0))
        result = {}
        decays = {}
        for scent, score in blended.items():
            p_nat, p_urb = _SCENT_AXIS[scent]
            alignment    = p_nat * nature_signal + p_urb * urban_signal
            conflict     = 1.0 - alignment
            # Kentsel baskın sertleştirme: urban > eşik VE doğa kokusu (p_nat>0.50)
            if urban_signal > _hardening_thr and p_nat > 0.50:
                effective_lam = lam * (1.0 + (urban_signal - _hardening_thr) * _hardening_scale)
            else:
                effective_lam = lam
            # Limon Kuzey/Karasal Sertleştirme: Çam+Toprak+İs/Duman baskınsa lambda artar
            if scent == "Limon":
                _karasal = blended.get("Çam", 0.0) + blended.get("Toprak", 0.0) + blended.get("İs/Duman", 0.0)
                _kar_thr   = float(self._cfg.get("vcv", {}).get("limon_karasal_threshold", 0.45))
                _kar_scale = float(self._cfg.get("vcv", {}).get("limon_karasal_scale",     2.0))
                if _karasal > _kar_thr:
                    effective_lam = effective_lam * (1.0 + (_karasal - _kar_thr) * _kar_scale)
            decay        = math.exp(-effective_lam * conflict)
            result[scent] = score * decay
            decays[scent] = decay          # EMA tonu için ham VCV çarpanı
        return result, decays

    # ------------------------------------------------------------------
    # VCV-3 — Karasal/Sucul Ayrımı (Terrestrial/Aquatic Suppression)
    # Karasal doğa baskınsa (orman, dağ, kaya) Deniz kokusunu söndür.
    # ------------------------------------------------------------------
    # Karasal sahnede YOLO nesne sinyal kategorisi
    _YOLO_TERRESTRIAL_LABELS: frozenset = frozenset({
        "mountain", "rock", "cliff", "hill",
        "tree", "palm tree", "bush", "plant", "potted plant",
        "house", "building", "skyscraper", "bridge", "wall",
        "road", "sidewalk", "fence",
    })

    # ------------------------------------------------------------------
    # VCV-G — Karasal Güvence Katmanı (Context Guard)
    # K-Means baskın rengi yeşil/doğa ise, CLIP'in hatalı ürettiği
    # yüksek aquatic / düşük terrestrial sinyalini düzelt.
    # ovl_validation shallow-copy üzerinde çalışır → thread-safe.
    # ------------------------------------------------------------------
    def _vcv_terrestrial_guard(
        self,
        colors_bgr:     list[tuple],
        ovl_validation: dict[str, float],
        clip_has_run:   bool,
    ) -> dict[str, float]:
        if not clip_has_run or not colors_bgr:
            return ovl_validation

        cfg_vcv   = self._cfg.get("vcv", {})
        min_terr  = float(cfg_vcv.get("min_terrestrial_guard",      0.65))
        max_aq    = float(cfg_vcv.get("max_aquatic_limit_on_green",  0.20))
        h_low     = float(cfg_vcv.get("vcv_guard_green_h_low",       55.0))
        h_high    = float(cfg_vcv.get("vcv_guard_green_h_high",     140.0))
        s_min     = float(cfg_vcv.get("vcv_guard_green_s_min",       40.0))

        green_hits = 0
        for bgr in colors_bgr[:3]:
            try:
                b, g, r = int(bgr[0]), int(bgr[1]), int(bgr[2])
            except (IndexError, TypeError, ValueError):
                continue
            pixel = np.array([[[b, g, r]]], dtype=np.uint8)
            hsv   = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0, 0]
            h_deg = float(hsv[0]) * 2.0   # OpenCV H (0-179) → derece (0-358)
            s_val = float(hsv[1])          # S (0-255)
            if h_low <= h_deg <= h_high and s_val > s_min:
                green_hits += 1

        if green_hits == 0:
            return ovl_validation

        result = dict(ovl_validation)
        old_terr = result.get("terrestrial", 0.5)
        old_aq   = result.get("aquatic",     0.5)
        result["terrestrial"] = max(old_terr, min_terr)
        result["aquatic"]     = min(old_aq,   max_aq)
        _log.debug(
            "[VCV Guard] Yeşil renk hakimiyeti tespit edildi."
            " Karasal bağlam %.2f değerine sabitlendi."
            " (terrestrial: %.2f→%.2f, aquatic: %.2f→%.2f)",
            min_terr,
            old_terr, result["terrestrial"],
            old_aq,   result["aquatic"],
        )
        return result

    def _apply_aquatic_suppression(
        self,
        blended:        dict[str, float],
        ovl_validation: dict[str, float],
        clip_has_run:   bool = False,
    ) -> dict[str, float]:
        """Absorber mimarisi kaydolduğundan tek katman yeterli.
        Karasal CLIP sinyali baskınsa (terrestrial > eşik) Deniz ek penalty alır."""
        cfg_vcv    = self._cfg.get("vcv", {})
        result     = dict(blended)
        terrestrial = float(ovl_validation.get("terrestrial", 0.5))
        thr        = float(cfg_vcv.get("terrestrial_suppress_threshold", 0.60))
        scale      = float(cfg_vcv.get("terrestrial_suppress_scale",     4.0))
        floor_mult = float(cfg_vcv.get("aquatic_floor_multiplier",       0.30))
        if terrestrial > thr:
            penalty        = min(1.0, (terrestrial - thr) * scale)
            effective_mult = max(floor_mult, 1.0 - penalty)
            result["Deniz"] = result.get("Deniz", 0.0) * effective_mult
        return result

    # ------------------------------------------------------------------
    # VCV-4 — Karasal İs/Duman Yumuşatma (Terrestrial İs/Duman Mitigation)
    # Dağ/orman/kaya baskınsa İs/Duman arka plana çekilir;
    # Çam ve Nane'nin en az %40 altında kalır.
    # ------------------------------------------------------------------
    def _apply_terrestrial_is_mitigation(
        self,
        blended:        dict[str, float],
        ovl_validation: dict[str, float],
    ) -> dict[str, float]:
        terrestrial = float(ovl_validation.get("terrestrial", 0.5))
        cfg_vcv     = self._cfg.get("vcv", {})
        thr         = float(cfg_vcv.get("is_terr_threshold",   0.55))
        scale       = float(cfg_vcv.get("is_terr_scale",       2.0))
        max_pen     = float(cfg_vcv.get("is_terr_max_penalty", 0.55))
        ceil_ratio  = float(cfg_vcv.get("is_terr_green_ceil",  0.60))
        if terrestrial <= thr:
            return blended
        # Oransal sönümleme
        penalty = min(max_pen, (terrestrial - thr) * scale)
        result  = dict(blended)
        result["İs/Duman"] = blended.get("İs/Duman", 0.0) * (1.0 - penalty)
        # Göreli tavan: İs/Duman ≤ max(Çam, Nane) × ceil_ratio
        green_ref = max(result.get("Çam", 0.0), result.get("Nane", 0.0))
        if green_ref > 0.05:
            ceiling = green_ref * ceil_ratio
            if result["İs/Duman"] > ceiling:
                result["İs/Duman"] = ceiling
        return result

    # ------------------------------------------------------------------
    # Modül A — Nesne Veto Sistemi (Object over Color)
    # K-Means yüksek skor verdi ama YOLO desteklemiyor → cezalandır
    #
    # Ezici Çoğunluk İstisnası (Overwhelming Dominance Bypass):
    # FORMLESS_SCENTS listesindeki kokular (Deniz, Çam, Toprak) için
    # K-Means skoru bypass_threshold üzerindeyse YOLO vetosunu atla.
    # YOLO'nun sınıf olarak tanımlayamadığı formsuz ortamlarda
    # (açık deniz, orman, çayır) gerçek koku haksız yere veto edilmez.
    # ------------------------------------------------------------------
    def _color_veto_penalty(
        self,
        yolo_s:   dict[str, float],
        kmeans_s: dict[str, float],
        blended:  dict[str, float],
    ) -> dict[str, float]:
        cfg_veto = self._cfg.get("color_veto", {})
        if not cfg_veto.get("enabled", True):
            return blended

        kmeans_thresh    = float(cfg_veto.get("kmeans_dominant_threshold", 0.30))
        yolo_support     = float(cfg_veto.get("yolo_support_min",          0.05))
        blend_thresh     = float(cfg_veto.get("min_blend_to_trigger",      0.12))
        penalty          = float(cfg_veto.get("penalty",                   0.50))
        bypass_threshold = float(cfg_veto.get("formless_bypass_threshold", 0.60))

        result = dict(blended)
        for scent in SCENTS:
            b = result.get(scent, 0.0)
            if b < blend_thresh:
                continue

            km_score = kmeans_s.get(scent, 0.0)

            # Ezici Çoğunluk İstisnası: formsuz ortam + baskın renk → veto atla
            if (scent in FORMLESS_SCENTS
                    and km_score >= bypass_threshold):
                continue

            if km_score > kmeans_thresh and yolo_s.get(scent, 0.0) < yolo_support:
                result[scent] = b * penalty

        return self._safe_normalize(result)

    # ------------------------------------------------------------------
    # Modül B — Makro Bağlam Sınıflandırıcı
    # YOLO tespitleri → INDOOR / NATURE / WATER / KITCHEN / CITY / UNKNOWN
    # ------------------------------------------------------------------
    def _classify_macro_context(self, detections: list[dict]) -> str:
        if not detections:
            return "UNKNOWN"

        det_labels = {d["label"].lower() for d in detections}
        ctx_hits: dict[str, int] = {}
        for ctx, labels in _MACRO_CTX_LABELS.items():
            hits = len(det_labels & labels)
            if hits > 0:
                ctx_hits[ctx] = hits

        if not ctx_hits:
            return "UNKNOWN"

        # Eşitlik durumunda öncelik sırası
        priority = ["INDOOR", "KITCHEN", "NATURE", "WATER", "CITY"]
        max_hits  = max(ctx_hits.values())
        candidates = [ctx for ctx, h in ctx_hits.items() if h == max_hits]
        for p in priority:
            if p in candidates:
                return p
        return candidates[0]

    # ------------------------------------------------------------------
    # Modül C — Makro Bağlam Sahne Cezası
    # Sahne bağlamına uymayan kokuları yumuşak çarpanlarla baskıla
    # ------------------------------------------------------------------
    def _apply_macro_context_penalty(
        self,
        blended:       dict[str, float],
        macro_context: str,
    ) -> dict[str, float]:
        adjustments = _CTX_SCENT_ADJUSTMENTS.get(macro_context, {})
        if not adjustments:
            return blended

        result = {
            s: blended.get(s, 0.0) * adjustments.get(s, 1.0)
            for s in SCENTS
        }
        return self._safe_normalize(result)

    # ------------------------------------------------------------------
    # Modül D' — Kesin Veto (Hard Veto)
    # (a) Bağlam tablosu: NATURE/WATER → Vanilya sıfır
    # (b) Çapa nesne: herhangi bir dış mekan nesnesi varsa iç mekan kokuları veto
    # Normalizasyon tuzağına düşmemek için _safe_normalize ile kapanır.
    # ------------------------------------------------------------------
    def _apply_hard_veto(
        self,
        blended:    dict[str, float],
        macro_ctx:  str,
        detections: list[dict] | None = None,
    ) -> dict[str, float]:
        veto_set = set(_HARD_VETO_TABLE.get(macro_ctx, frozenset()))

        # Çapa nesne kontrolü — tek bir dış mekan nesnesi sahneyi kilitle
        if detections:
            det_labels = {d["label"].lower() for d in detections}
            if det_labels & _ANCHOR_OUTDOOR_OBJECTS:
                veto_set |= _ANCHOR_VETO_SCENTS

        if not veto_set:
            return blended
        result = {s: (0.0 if s in veto_set else v) for s, v in blended.items()}
        return self._safe_normalize(result)

    # ------------------------------------------------------------------
    # Modül E — Bağlamsal Skor Yönlendirme (Contextual Score Routing)
    # Aydınlatma + sahne bağlamı uyumsuzluğunu skor transferiyle düzelt.
    # GOLDEN_HOUR + WATER/NATURE : Vanilya/İs yanımaları → Deniz/Toprak/Limon
    # NIGHT_DARK  + NATURE/UNKNOWN : belirsiz skorlar → Çam/Toprak
    # ------------------------------------------------------------------
    def _contextual_score_routing(
        self,
        blended:   dict[str, float],
        lighting:  str,
        macro_ctx: str,
    ) -> dict[str, float]:
        cfg_sr = self._cfg.get("score_routing", {})
        if not cfg_sr.get("enabled", True):
            return blended

        result = dict(blended)

        if lighting == "GOLDEN_HOUR":
            if macro_ctx == "WATER":
                # YOLO onaylanmış dış mekan sahili: gün batımı yansımaları
                transfer = sum(result.get(s, 0.0) for s in _GOLDEN_HOUR_VETO_SCENTS)
                for s in _GOLDEN_HOUR_VETO_SCENTS:
                    result[s] = 0.0
                result["Deniz"]  = min(1.0, result.get("Deniz", 0.0)  + transfer * 0.70)
                result["Toprak"] = min(1.0, result.get("Toprak", 0.0) + transfer * 0.30)
            else:
                # İç mekan / boş kare: turuncu ışık → Deniz'e kesinlikle aktarma
                is_val = result.get("İs/Duman", 0.0)
                result["İs/Duman"] = 0.0
                if macro_ctx == "UNKNOWN":
                    # Boş kare nötr dağılım: Toprak + Çam
                    result["Toprak"] = min(1.0, result.get("Toprak", 0.0) + is_val * 0.60)
                    result["Çam"]    = min(1.0, result.get("Çam", 0.0)    + is_val * 0.40)
                else:
                    # İç mekan sıcak ışık: ahşap/toprak tonu + vanilya
                    result["Toprak"]  = min(1.0, result.get("Toprak", 0.0)  + is_val * 0.50)
                    result["Vanilya"] = min(1.0, result.get("Vanilya", 0.0) + is_val * 0.50)

        elif lighting == "NIGHT_DARK" and macro_ctx in {"NATURE", "UNKNOWN"}:
            transfer = (
                result.get("Vanilya", 0.0) * 0.80
                + result.get("Limon", 0.0)   * 0.50
            )
            result["Vanilya"] = result.get("Vanilya", 0.0) * 0.20
            result["Limon"]   = result.get("Limon", 0.0)   * 0.50
            result["Çam"]    = min(1.0, result.get("Çam", 0.0)    + transfer * 0.65)
            result["Toprak"] = min(1.0, result.get("Toprak", 0.0) + transfer * 0.35)

        # Doğa Beyazı Filtresi — tüm ışık koşullarında NATURE bağlamı
        # Bulut/sis/kar beyazlıkları K-Means'i Vanilya'ya iter; bunu Çam/Toprak'a aktar
        if macro_ctx == "NATURE":
            van_val = result.get("Vanilya", 0.0)
            is_val  = result.get("İs/Duman", 0.0)
            transfer_n = van_val * 0.80 + is_val * 0.80
            result["Vanilya"]   = van_val * 0.20
            result["İs/Duman"]  = is_val  * 0.20
            result["Çam"]    = min(1.0, result.get("Çam", 0.0)    + transfer_n * 0.65)
            result["Toprak"] = min(1.0, result.get("Toprak", 0.0) + transfer_n * 0.35)

        return self._safe_normalize(result)

    # ------------------------------------------------------------------
    # Doğa Gökyzü Veto (Uzamsal Beyaz Filtresi)
    # Üst %40 alan beyaz/açık piksellerle doluysa (gökyzü/bulut)
    # bu bölgeden gelen Vanilya puanını buda, Çam/Toprak'a transfer et.
    # ------------------------------------------------------------------
    def _nature_sky_veto(
        self, frame: np.ndarray, blended: dict[str, float]
    ) -> dict[str, float]:
        h = frame.shape[0]
        sky = frame[: int(h * 0.40), :, :]
        hsv_sky = cv2.cvtColor(sky, cv2.COLOR_BGR2HSV).astype(np.float32)
        s_ch = hsv_sky[:, :, 1]
        v_ch = hsv_sky[:, :, 2]
        white_mask  = (s_ch < 50.0) & (v_ch > 180.0)
        white_ratio = float(np.mean(white_mask))

        if white_ratio < 0.30:
            return blended

        result  = dict(blended)
        van_val = result.get("Vanilya", 0.0)
        cut     = min(van_val * 0.80, van_val * white_ratio)
        result["Vanilya"] = van_val - cut
        result["Çam"]    = min(1.0, result.get("Çam", 0.0)    + cut * 0.60)
        result["Toprak"] = min(1.0, result.get("Toprak", 0.0) + cut * 0.40)
        return self._safe_normalize(result)

    # ------------------------------------------------------------------
    # Modül D — Koku Baskınlık Hiyerarşisi
    # Güçlü koku baskın olduğunda zayıf kokuları matematiksel maskele
    # ------------------------------------------------------------------
    def _apply_dominance_hierarchy(self, blended: dict[str, float]) -> dict[str, float]:
        cfg_dom = self._cfg.get("dominance_hierarchy", {})
        if not cfg_dom.get("enabled", True):
            return blended

        min_dom_score   = float(cfg_dom.get("min_dominant_score",        0.22))
        min_dom_power   = float(cfg_dom.get("min_dominant_power",        0.50))
        max_suppress    = float(cfg_dom.get("max_suppression",           0.45))
        nature_scents   = set(cfg_dom.get("nature_scents",              ["Çam", "Toprak", "Nane"]))
        coalition_mult  = float(cfg_dom.get("nature_coalition_suppress", 0.60))

        dominant_scent = max(blended, key=blended.get)
        dominant_score = blended[dominant_scent]
        dominant_power = SCENT_DOMINANCE.get(dominant_scent, 0.5)

        if dominant_score < min_dom_score or dominant_power < min_dom_power:
            return blended

        dom_is_nature = dominant_scent in nature_scents
        result = dict(blended)
        for scent in SCENTS:
            if scent == dominant_scent:
                continue
            power_diff = dominant_power - SCENT_DOMINANCE.get(scent, 0.5)
            if power_diff <= 0:
                continue
            suppression = min(max_suppress, power_diff * dominant_score * 0.50)
            if dom_is_nature and scent in nature_scents:
                # Doğa Koalisyonu: doğa kokusu başka bir doğa kokusunu yok etmesin.
                # Baskılama sonrası skor, orijinalin coalition_mult katından aşağı düşemez.
                protected_min  = blended[scent] * coalition_mult
                result[scent]  = max(result[scent] * (1.0 - suppression), protected_min)
                _log.debug(
                    "[Doğa Koalisyonu] %s nedeniyle %s korundu, çarpan: %.2f",
                    dominant_scent, scent, coalition_mult,
                )
            else:
                result[scent] *= (1.0 - suppression)

        return self._safe_normalize(result)

    # ------------------------------------------------------------------
    # 8 Kanallı Seri Komut (7 koku + PurgeSpeed)
    # ------------------------------------------------------------------
    def get_serial_command_8ch(
        self,
        scores: dict[str, float],
        attenuation: dict[str, float],
        purge_speed: float = 0.0,
        intensity:   float = 1.0,
    ) -> str:
        """
        Format: !ch0,ch1,...,ch6,ch7|checksum#

        Tahliye protokolü (purge_speed >= 1.0):
            ch0-ch6 = 0  (tüm koku kanalları kapalı)
            ch7     = pwm_max  (tahliye fanı maksimum güç)
        Normal çalışma:
            ch0-ch6 = skora × attenuation × intensity × pwm_max
            ch7     = purge_speed × pwm_max
        """
        pwm_max = int(self._cfg.get("hardware", {}).get("pwm_max", 4095))
        if purge_speed >= 1.0:
            # PRD 5.4: Tahliye aktif — tüm koku kanalları kapalı, ch7 tam güç
            vals = ["0"] * len(SCENTS)
        else:
            vals = [
                str(int(max(0, min(pwm_max,
                    scores.get(s, 0.0) * attenuation.get(s, 1.0) * pwm_max * intensity
                ))))
                for s in SCENTS
            ]
        vals.append(str(int(purge_speed * pwm_max)))
        checksum = sum(int(v) for v in vals) % 256
        cmd = f"!{','.join(vals)}|{checksum}#"
        # Terminal debug: her 30 karede bir ham seri komutu yazdır
        # Arayüz donsa bile donanıma gerçekten koku gidip gitmediği görülsün.
        if self._ovl_frame_counter % 30 == 0:
            print(
                f"[SERIAL DBG] frame={self._ovl_frame_counter:>5} "
                f"purge={purge_speed:.2f} {cmd}",
                flush=True,
            )
        return cmd

    # ------------------------------------------------------------------
    # Ana Karar Metodu
    # Döndürür: (ui_scores, metadata) — DecisionEngine ile uyumlu arayüz
    # ------------------------------------------------------------------
    def decide(
        self,
        frame: np.ndarray,
        detections: list[dict],
        colors_bgr: list[tuple],
        motion_score:  float = 0.0,
        is_scene_cut:  bool  = False,
    ) -> tuple[dict[str, float], dict]:
        t0  = time.perf_counter()
        with self._cfg_lock:   # KRİTİK-2: shallow copy — in-place mutasyona karşı izole snapshot
            cfg = dict(self._cfg)

        # ------------------------------------------------------------------
        # WARMUP LOCK: CLIP ilk anlamsal onayını vermeden önce tüm PWM çıkışı
        # sıfırlanır. K-Means ve YOLO çalışsa bile Absorber kalkanı aktif
        # değilken histerezi/PWM kanalına hiçbir koku açma yetkisi gitmesin.
        # CLIP thread'i warmup sırasında da tetiklenir; ilk sonuç gelir gelmez kilit açılır.
        # ------------------------------------------------------------------
        self._ovl_frame_counter += 1

        # CLIP worker'ı warmup dahi olsa arka planda çalıştır (interval'e uyarak)
        _wu_ovl_cfg = cfg.get("ovl", {})
        _wu_ivl     = int(_wu_ovl_cfg.get("analysis_interval", 15))
        if (
            self._ovl_pipe is not None
            and self._ovl_frame_counter % _wu_ivl == 0
            and self._ovl_run_lock.acquire(blocking=False)
        ):
            threading.Thread(
                target=self._ovl_worker, args=(frame.copy(),), daemon=True
            ).start()

        with self._ovl_lock:
            _warmup_ovl_scores = dict(self._ovl_cache)
        if not self._clip_ever_run:
            if any(v > 0 for v in _warmup_ovl_scores.values()):
                self._clip_ever_run = True
                _log.info("[WARMUP] CLIP ilk inference tamamlandı (kare=%d) — PWM kilidi açıldı.",
                          self._ovl_frame_counter)
            else:
                _zero = {s: 0.0 for s in SCENTS}
                return _zero, {
                    "warmup_lock":      True,
                    "warmup_frame":     self._ovl_frame_counter,
                    "attenuation":      {s: 1.0 for s in SCENTS},
                    "purge_speed":      0.0,
                    "hysteresis_state": {s: False for s in SCENTS},
                    "dominant_scent":   "-",
                    "pdm_elapsed_ms":   round((time.perf_counter() - t0) * 1000, 2),
                }

        # ── V3 Katman B: VLM semantik maske — asenkron frame gönder ─────────────
        if self._vlm is not None:
            self._vlm.submit_frame(frame, self._ovl_frame_counter)

        # ── V3 Semantik Boşluk Kontrolü — tek atomik okuma (race-free) ──────────
        # get_mask_and_void_state() tek lock altında mask+void+trigger okur.
        # Üç ayrı çağrıda araya giren VLM güncellemesi mask≠void_state
        # tutarsızlığını (Bayes uniform dağılım bug'ı) önler.
        _semantic_mask, _is_void, _void_purge, _strict_ok = (
            self._vlm.get_mask_and_void_state()
            if self._vlm is not None
            else (dict(DEFAULT_MASK), False, False, True)
        )
        self._last_semantic_mask = _semantic_mask

        # ── Bypass Geçiş Tespiti: kısıtlı mask → tam açık ───────────────────────
        # SOFT_SUPPRESSION/HARD_VETO döneminde Kalman P küçülmüş (K≈20.52) olabilir.
        # Bypass/strict=True anında Kalman'1 sıfırlayarak P=1.0 → K≈20.91 →
        # Katman A ilk karede histerezis eşiğini (÷0.20) aşabilir.
        _mask_now_open = not _is_void and all(v >= 0.99 for v in _semantic_mask.values())
        if self._prev_mask_was_restricted and _mask_now_open:
            self._kalman.reset()
            self._bayes.reset_prior()
            for _sv in SCENTS:
                self._hysteresis_state[_sv] = False
                self._vote_buffer[_sv].clear()
            _log.info(
                "[VLM] Mask kısıtlı→açık GEÇİŞ — Kalman(P→1.0)+Bayes+Histerezis+Oylama sıfırlandı"
            )
        self._prev_mask_was_restricted = any(v < 0.5 for v in _semantic_mask.values())

        if _void_purge:
            # Yükselen kenar: Kalman + Bayes prior + histerezis tamamen sıfırla
            self._kalman.reset()
            self._bayes.reset_prior()        # void öncesi prior skor sıçramasını önle
            for _sv in SCENTS:
                self._hysteresis_state[_sv] = False
                self._vote_buffer[_sv].clear()
            _log.info("[VOID] Semantik Boşluk aktif — Kalman+Bayes sıfırlandı, PWM=0")
        if _is_void:
            _zero = {s: 0.0 for s in SCENTS}
            return _zero, {
                "is_abstract_or_void":  True,
                "void_purge_triggered": _void_purge,
                "attenuation":          {s: 1.0 for s in SCENTS},
                "purge_speed":          0.0,
                "hysteresis_state":     dict(self._hysteresis_state),
                "dominant_scent":       "—",
                "pdm_elapsed_ms":       round((time.perf_counter() - t0) * 1000, 2),
                "motion_score":         motion_score,
                "v3_vlm_status":        self._vlm.get_status() if self._vlm is not None
                                        else {"vlm_status": "disabled"},
                "v3_semantic_mask":     {s: round(v, 3) for s, v in _semantic_mask.items()},
                "v3_kalman_states":     self._kalman.get_states(),
                "v3_bayes_posterior":   {},
                "v3_module_pdfs":       {},
                "blended_4way":         {},
                "module_scores":        {},
                "voted":                _zero,
                "blended_pre_vote":     _zero,
            }

        # 1. CLAHE ile ışık normalizasyonu
        norm_frame = self.apply_clahe(frame)

        # 1b. Aydınlatma profili — orijinal kare üzerinde (CLAHE öncesi HSV doğru)
        lighting = self._detect_lighting_profile(frame)

        # 2. Temporal memory güncelle (FPS-bağımsız nesne hafizası)
        for det in detections:
            self._memory.update(
                det["label"].lower(), float(det.get("confidence", 0.0))
            )
        if self._ovl_frame_counter % 30 == 0:  # ORTA-4: throttle — 30fps*30 = 900 iter/sn önler
            self._memory.cleanup()

        # 3. YOLO modülü (canlı tespit + zayıflayan hafiza katkısı)
        yolo_s, _yolo_penalty_log = self._yolo_module_scores(detections)
        for label, mapping in OBJECT_SCENT_MAP.items():
            mem_w = self._memory.get_weight(label)
            if mem_w > 0.02:
                for scent, w in mapping.items():
                    yolo_s[scent] = min(1.0, yolo_s[scent] + mem_w * w * 0.45)
        yolo_s = self._safe_normalize(yolo_s, min_total=0.0)  # KRİTİK-4

        # 3b. Ağaç/Bitki Çam Bonusu — YOLO nesne alanına orantılı, zero-sum transfer
        tree_area = sum(
            float(d.get("area_ratio", 0.0))
            for d in detections
            if d["label"].lower() in {"tree", "plant", "potted plant", "palm tree", "bush"}
        )
        if tree_area > 0:
            boost  = min(0.25, tree_area * 1.5)    # maks %25
            donor  = boost / 2.0
            yolo_s["Çam"]      = min(1.0, yolo_s["Çam"]      + boost)
            yolo_s["Toprak"]   = max(0.0, yolo_s["Toprak"]   - donor)
            yolo_s["İs/Duman"] = max(0.0, yolo_s["İs/Duman"] - donor)
            yolo_s = self._safe_normalize(yolo_s, min_total=0.0)  # KRİTİK-4

        # 4. Dinamik ROI → K-Means renk analizi odaklı bölgede
        roi_frame  = self._dynamic_roi_crop(norm_frame, detections)
        roi_colors = self._run_kmeans_on_roi(roi_frame)
        kmeans_s, _kmeans_neutral_ct = self._kmeans_module_scores(roi_colors if roi_colors else colors_bgr)

        # 5. Doku analizi
        texture_s, lap_var = self._texture_module_scores(norm_frame)

        # 6. Optik akış — renk bölgesi hareket yönü bonusu/baskısı
        flow_s = self._optical_flow_scores(frame)
        for scent in SCENTS:
            kmeans_s[scent] = max(0.0, kmeans_s[scent] + flow_s.get(scent, 0.0))
        kmeans_s = self._safe_normalize(kmeans_s, min_total=0.0)  # KRİTİK-4

        # 4b. Yeşil Piksel Bonusu — yeşil oran >%15 ise Toprak→Çam transfer
        green_ratio = self._compute_green_ratio_frame(roi_frame)
        if green_ratio > 0.15:
            transfer = min(
                kmeans_s.get("Toprak", 0) * 0.70,
                (green_ratio - 0.15) * 3.0 * kmeans_s.get("Toprak", 0),
            )
            kmeans_s["Çam"]    = min(1.0, kmeans_s.get("Çam",    0) + transfer)
            kmeans_s["Toprak"] = max(0.0, kmeans_s.get("Toprak", 0) - transfer)
            km_gn = sum(kmeans_s.values())
            if km_gn > 0:
                kmeans_s = {k: v / km_gn for k, v in kmeans_s.items()}

        # 6b. OVL Modülü — warmup'tan sonra normal interval'de sürer
        ovl_cfg = cfg.get("ovl", {})
        ovl_ivl = int(ovl_cfg.get("analysis_interval",     15))
        ovl_thr = float(ovl_cfg.get("confidence_threshold", 0.70))
        # KRİTİK-1: _ovl_run_lock.acquire(blocking=False) atomik — iki thread başlamaz
        # Not: Warmup bloğunda aynı kare için zaten tetiklendiyse acquire başarısız olur (non-blocking)
        if (
            self._ovl_pipe is not None
            and self._ovl_frame_counter % ovl_ivl == 0
            and self._ovl_run_lock.acquire(blocking=False)
        ):
            threading.Thread(
                target=self._ovl_worker, args=(frame.copy(),), daemon=True
            ).start()

        # ── PHASE-1: Atomik Per-Kare Snapshot ────────────────────────────────
        # ovl_scores, ovl_conf, vlm_conf bu karede sadece BU BLOKTA okunur.
        # _kmeans_tamer() ve Post-Fusion Guard bu dondurulmuş veriyi kullanır;
        # asenkron dict'lere tekrar sorgu atmaz → lag/race-condition yok.
        with self._ovl_lock:
            ovl_scores     = dict(self._ovl_cache)
            ovl_conf       = self._ovl_confidence
            ovl_validation = dict(self._ovl_validation)
            ovl_debug      = dict(self._ovl_debug)
        # VLM güveni: Ollama JSON 'confidence' — GIL-atomik property.
        # CLIP snapshot ile eş zamanlı dondurulur → tutarsızlık yok.
        _vlm_conf = self._vlm.vlm_confidence if self._vlm is not None else 0.0

        # Semantik analiz takip değişkenleri (metadata için)
        grid_deltas    = {s: 0.0 for s in SCENTS}
        sky_blue_ratio = 0.0
        sky_freshness  = 0.0   # VCV: açık gök ferahlık sinyali
        _yolo_clip_adj: dict[str, float] = {}
        _urban_sig     = 0.5   # VCV: bağlam ekseni (sonraki kare için)
        _nature_sig    = 0.5

        # ------------------------------------------------------------------
        # 7. V3 Katman C — BayesFuser Log-Uzayı Füzyon
        #    Prior × ∏ P(data_i | s)^w_i × mask[s]  →  normalize posterior
        #    VLM semantik maske (_semantic_mask) likelihood'u ölçekler.
        #    CLIP hazır değilse bilgisiz uniform PDF girer (log katkısı sıfır).
        # ------------------------------------------------------------------
        clip_has_run = self._ovl_pipe is not None and any(
            v > 0 for v in ovl_scores.values()
        )

        # ── K-Means Tamer: Dinamik per-kanal ağırlıklandırma ─────────────────────
        # BayesFuser sabit w_kmeans=0.15 + power-scaling ≡ dinamik ağırlık:
        #   w_base * log(pdf^(w_dyn/w_base)) = w_dyn * log(pdf)  [log-uzayı eşdeğerli]
        # _vlm_conf + ovl_scores: kare başında dondurulmuş snapshot (Phase-1 yukarıda).
        _kmeans_tamed, _tamer_mode, _tamer_weights = self._kmeans_tamer(
            kmeans_s, _semantic_mask, ovl_scores,
            clip_conf=ovl_conf, vlm_conf=_vlm_conf,
        )
        if _tamer_mode != "Normal" or self._ovl_frame_counter % 30 == 1:
            _log.info(
                "[K-Means Tamer] Mod: %s | Dinamik Agirliklar -> "
                "Nane: %.3f, Vanilya: %.3f, Cam: %.3f",
                _tamer_mode,
                _tamer_weights.get("Nane",    0.0),
                _tamer_weights.get("Vanilya", 0.0),
                _tamer_weights.get("\u00c7am",     0.0),
            )

        _n_scents = len(SCENTS)
        _module_pdfs: dict[str, dict[str, float]] = {
            "yolo":    yolo_s,
            "kmeans":  _kmeans_tamed,   # Tamer: per-kanal dinamik ağırlıklı
            "texture": texture_s,
            # CLIP hazır değilse uniform dağılım → log(1/7) tüm kanallar için eşit → normalize'de etkisiz
            "clip":    ovl_scores if clip_has_run else {s: 1.0 / _n_scents for s in SCENTS},
        }

        # ── Feature 1: VLM Master Prior (Bilişsel Önsel Çivileme) ────────────────
        # Bayes füzyonu başlamadan önce, VLM yüksek güvenle (≥ eşik) sahne modunu
        # tanımladıysa yasaklanan kanalları prior'da log-uzayında baskıla.
        # BayesFuser.fuse(prior=...) parametresini zaten destekler → bayes_fuser.py değişmez.
        _VLM_PRIOR_THRESH = float(
            self._cfg.get("cognitive_gate", {}).get("vlm_prior_threshold", 0.70)
        )
        _vlm_prior_suppress: set[str] = set()
        if _tamer_mode == "Aksiyon":
            _vlm_prior_suppress = {"Vanilya", "Nane", "Limon"}
        elif _tamer_mode == "Kurtarıcı+Emniyet":
            _vlm_prior_suppress = {"Vanilya", "Nane"}
        elif _tamer_mode == "Kentsel":
            _vlm_prior_suppress = {"Vanilya"}

        _vlm_master_prior: dict[str, float] | None = None
        if _vlm_conf >= _VLM_PRIOR_THRESH and _vlm_prior_suppress:
            _rp       = {s: (1e-9 if s in _vlm_prior_suppress else 1.0) for s in SCENTS}
            _rp_total = sum(_rp.values())
            _vlm_master_prior = {s: v / _rp_total for s, v in _rp.items()}
            _log.debug(
                "[VLM Master Prior] conf=%.2f \u2265 %.2f \u2192 suppress: %s",
                _vlm_conf, _VLM_PRIOR_THRESH, sorted(_vlm_prior_suppress),
            )

        blended = self._bayes.fuse(
            module_pdfs   = _module_pdfs,
            semantic_mask = _semantic_mask,
            prior         = _vlm_master_prior,  # None \u2192 BayesFuser dahili prior\u2019unu kullanır
        )

        blended = self._safe_normalize(blended, min_total=0.0)  # KRİTİK-4
        blended_4way = dict(blended)  # DEBUG: Bayes posterior snapshotu (veto öncesi ham değer)

        # ── Post-Fusion Hard Veto (Tamer mod garantisi) ──────────────────────────
        # Power-scaling sıfır kanalları kurtaramazsa bu blok nihai kalkan olarak
        # Aksiyon/Kentsel modlarda belirlenen kanalları kodsal olarak sıfırlar.
        # Normalize: sıfırlanan kütleyi kalan kanallara dağıt.
        # ── Fix-2: Anti-Normalization Inflation Guard ────────────────────────────
        # Standart _safe_normalize → zayıf sinyal %100'e fırlar (şişme).
        # _anti_inflate_normalize → ambient floor denom'a eklenir; sızıntı yok.
        if _tamer_mode == "Aksiyon":
            _pf_veto = {"Vanilya", "Nane", "Limon"}
            for _sv in _pf_veto:
                blended[_sv] = 0.0
            _pf_raw = sum(v for s, v in blended.items() if s not in _pf_veto)
            blended = self._anti_inflate_normalize(blended, _pf_veto)
            _log.debug(
                "[PostFusion Veto] Aksiyon → Vanilya/Nane/Limon sıfırlandı | "
                "raw_total=%.4f → anti-inflate uygulandı", _pf_raw
            )
        elif _tamer_mode == "Kurtarıcı+Emniyet":
            _pf_veto = {"Vanilya", "Nane"}
            for _sv in _pf_veto:
                blended[_sv] = 0.0
            _pf_raw = sum(v for s, v in blended.items() if s not in _pf_veto)
            blended = self._anti_inflate_normalize(blended, _pf_veto)
            _log.debug(
                "[PostFusion Veto] Kurtarıcı+Emniyet → Vanilya/Nane sıfırlandı | "
                "raw_total=%.4f → anti-inflate uygulandı", _pf_raw
            )
        elif _tamer_mode == "Kentsel":
            _pf_veto = {"Vanilya"}
            blended["Vanilya"] = 0.0
            _pf_raw = sum(v for s, v in blended.items() if s not in _pf_veto)
            blended = self._anti_inflate_normalize(blended, _pf_veto)
            _log.debug(
                "[PostFusion Veto] Kentsel → Vanilya sıfırlandı | "
                "raw_total=%.4f → anti-inflate uygulandı", _pf_raw
            )

        # 7-GR. Çevresel Filtre — Zemin doku sinyali + Gökyüzü ferahlık
        grid_deltas, sky_blue_ratio, sky_freshness = self._grid_sky_penalty(frame)
        for _gs in SCENTS:
            blended[_gs] = max(0.0, blended[_gs] + grid_deltas.get(_gs, 0.0))
        blended = self._safe_normalize(blended, min_total=0.0)  # KRİTİK-4

        # 7-SF. Gökyüzü Yanlış Pozitif Kapısı — üst %40 mavi+pürüzsüz → Deniz/Nane baskıla
        blended = self._sky_false_positive_gate(frame, blended, detections)

        # 7-TX. Doku Tabanlı Sucul Filtresi
        # Yüksek Laplacian (orman/dağ/kayalık) → sahne makroskobik açık su içermiyor.
        # CLIP/K-Means ne derse desin Deniz baskıla; aquatik YOLO nesnesi varsa bypass.
        _tex_cfg     = cfg.get("texture_aquatic_filter", {})
        _tex_thr     = float(_tex_cfg.get("high_texture_threshold", 600.0))
        _tex_penalty = float(_tex_cfg.get("penalty", 0.10))
        _tex_aquatic = {"boat", "ship", "sailboat", "surfboard", "canoe"}
        if lap_var > _tex_thr and not any(
            d.get("label", "").lower() in _tex_aquatic for d in detections
        ):
            blended["Deniz"] = blended.get("Deniz", 0.0) * _tex_penalty
            _log.debug(
                "[TEX-AQ] lap=%.0f > %.0f → Deniz×%.2f",
                lap_var, _tex_thr, _tex_penalty,
            )

        # 7-FO. Orman Baskın Gécme Engeli — CLIP orman güveni > eşik → Toprak Çam'ı geçemez
        _forest_dom_thr = float(cfg.get("ovl", {}).get("forest_dominance_threshold",   0.60))
        _forest_rcl_fac = float(cfg.get("ovl", {}).get("forest_recalibration_factor",  0.80))
        if clip_has_run and ovl_scores.get("Çam", 0) > _forest_dom_thr:
            toprak_v = blended.get("Toprak", 0)
            cam_v    = blended.get("Çam",    0)
            if toprak_v > cam_v:
                excess = toprak_v - cam_v
                blended["Çam"]    = min(1.0, cam_v    + excess * _forest_rcl_fac)
                blended["Toprak"] = max(0.0, toprak_v - excess * _forest_rcl_fac)
                blended = self._safe_normalize(blended, min_total=0.0)  # KRİTİK-4

        # 7-VCV. Vektörel Bağlam Doğrulaması
        # ----------------------------------------------------------------
        # Şu blokların yerini alır:
        #   7a  (Uyumsuzluk Matrisi  — sabit 0.10/0.15 çarpanlar)
        #   7-US (Kentsel Nane Filtresi — sabit %5 tavan)
        #   7-CI (Kimya Uyumsuzluğu — sabit sıfırlama)
        #   7b  (CLIP Çapraz Doğrulama — sabit eşik çarpanları)
        #   7c  (Güvenlik Tabanı — sabit %5 tavan)
        #   7d  (İs/Duman Doğa Sınırı — sabit %18 tavan)
        # Mekanizma: koku-eksen iç çarpımı → üstel sönümleme (exp(-λ·conflict))
        # Normalizasyon sonrası sönümlenen kütle bağlam-uyumlu kokulara akar.
        # ----------------------------------------------------------------

        # VCV-G: Karasal Güvence Katmanı — K-Means yeşil baskınlıkta CLIP
        # hatalı aquatic skoru ürettiyse ovl_validation yerel kopyasını düzelt.
        # self._ovl_validation değiştirilmez → thread-safe.
        ovl_validation = self._vcv_terrestrial_guard(
            colors_bgr, ovl_validation, clip_has_run
        )

        _urban_sig, _nature_sig = self._compute_context_axes(
            detections, ovl_validation, clip_has_run, ovl_conf, sky_freshness
        )
        _blended_pre_vcv = dict(blended)   # VCV öncesi snapshot — VCV debug ve Bayes referansı için
        if clip_has_run:
            blended, _vcv_raw_decays = self._vector_context_decay(blended, _urban_sig, _nature_sig)
            blended = self._apply_aquatic_suppression(blended, ovl_validation, clip_has_run=True)
            blended = self._apply_terrestrial_is_mitigation(blended, ovl_validation)
            blended = self._safe_normalize(blended, min_total=0.0)  # KRİTİK-4
        else:
            # CLIP henüz çalışmamış → VCV bypass: nötr (0.5/0.5) eksenin yarattığı
            # düzleştirme (flatline) ve sahte koku tetiklemesi engellenir.
            # blended doğrudan bir sonraki aşamaya (histerezis) iletilir.
            _vcv_raw_decays = {s: 1.0 for s in SCENTS}
        _prev_u = getattr(self, "_vcv_prev_urban", 0.5)
        if abs(_urban_sig - _prev_u) > 0.15 or self._ovl_frame_counter % 60 == 0:
            top_s = max(blended, key=blended.get) if blended else "-"
            _log.debug(
                "[VCV] kare=%d | kentsel=%.2f  dogal=%.2f | gok=%.2f | en_yuksek=%s(%.2f)",
                self._ovl_frame_counter, _urban_sig, _nature_sig, sky_freshness,
                top_s, blended.get(top_s, 0),
            )
        self._vcv_prev_urban = _urban_sig

        # VCV debug — per-scent sönümleme değerlerini yakala (dashboard debug sekmesi için)
        _lam_dbg        = float(self._cfg.get("vcv", {}).get("lambda", 2.5))
        _hard_thr_dbg   = float(self._cfg.get("vcv", {}).get("urban_hardening_threshold", 0.60))
        _hard_scale_dbg = float(self._cfg.get("vcv", {}).get("urban_hardening_scale",     5.0))
        _vcv_debug: dict[str, dict] = {}
        for _s in SCENTS:
            _p_nat, _p_urb = _SCENT_AXIS[_s]
            _align    = _p_nat * _nature_sig + _p_urb * _urban_sig
            _conflict = round(1.0 - _align, 3)
            if _urban_sig > _hard_thr_dbg and _p_nat > 0.50:
                _eff_lam = round(_lam_dbg * (1.0 + (_urban_sig - _hard_thr_dbg) * _hard_scale_dbg), 3)
            else:
                _eff_lam = _lam_dbg
            _vcv_debug[_s] = {
                "conflict": _conflict,
                "decay":    round(math.exp(-_eff_lam * _conflict), 3),
                "eff_lam":  _eff_lam,
            }

        # VCV Urban Boost — kentsel baskın sahne → İs/Duman taban bonus
        _vcv_urban_boost_factor = float(self._cfg.get("vcv", {}).get("urban_is_duman_boost", 0.15))
        _urban_is_bonus: float = 0.0
        if _urban_sig > _hard_thr_dbg:   # _hard_thr_dbg → _vcv_debug bloğundan geliyor
            _urban_is_bonus = round(_urban_sig * _vcv_urban_boost_factor, 3)
            blended["\u0130s/Duman"] = min(1.0, blended.get("\u0130s/Duman", 0.0) + _urban_is_bonus)
            blended = self._safe_normalize(blended, min_total=0.0)

        # 7-YC. YOLO-CLIP Nesne İlişkilendirme (Korunuyor — spesifik nesne-koku)
        _det_yc = {d["label"].lower() for d in detections}
        if _det_yc & {"boat", "ship", "sailboat"} and clip_has_run and ovl_scores.get("Deniz", 0) > 0.25:
            _boost = min(0.80, blended.get("Deniz", 0) + 0.25)
            _yolo_clip_adj["Deniz"] = round(_boost - blended.get("Deniz", 0), 3)
            blended["Deniz"] = _boost
        if _det_yc & {"car", "truck", "bus", "motorcycle", "train"}:
            _floor = float(cfg.get("score_routing", {}).get("is_duman_vehicle_floor", 0.15))
            if blended.get("İs/Duman", 0) < _floor:
                _yolo_clip_adj["İs/Duman"] = round(_floor - blended.get("İs/Duman", 0), 3)
                blended["İs/Duman"] = _floor
        if _yolo_clip_adj:
            blended = self._safe_normalize(blended, min_total=0.0)  # KRİTİK-4

        # 7b. Nesne Veto — saf renk eşleşmelerini YOLO desteği olmadan tırpanla
        blended = self._color_veto_penalty(yolo_s, kmeans_s, blended)

        # 7c. Makro Bağlam: sahne sınıflandır → uyumsuz kokuları baskıla
        macro_ctx = self._classify_macro_context(detections)

        # Dış Mekan Kilidi — YOLO + CLIP Onayı (Yumuşak Geçiş)
        # CLIP aktifse: Deniz güveni >= 0.25 olmadan WATER kilidi uygulanmaz.
        # CLIP henüz çalışmamışsa (conf=0): eski YOLO tabanlı davranış korunur.
        if lighting == "GOLDEN_HOUR" and macro_ctx not in {"WATER", "NATURE"}:
            det_labels  = {d["label"].lower() for d in detections}
            has_outdoor = bool(det_labels & _ANCHOR_OUTDOOR_OBJECTS)
            has_indoor  = bool(det_labels & _ANCHOR_INDOOR_OBJECTS)
            if has_outdoor and not has_indoor:
                clip_active  = self._ovl_pipe is not None and ovl_conf >= ovl_thr
                clip_sea_ok  = ovl_scores.get("Deniz", 0.0) >= float(cfg.get("ovl", {}).get("min_sea_confidence_for_golden", 0.25))
                # CLIP çalışıyor ama Deniz görmüyor → hard-lock uygulanmaz
                if not clip_active or clip_sea_ok:
                    macro_ctx = "WATER"

        blended   = self._apply_macro_context_penalty(blended, macro_ctx)

        # 7c-2. Kesin Veto — bağlam tablosu + çapa nesneler → iç mekan kokuları sıfır
        blended = self._apply_hard_veto(blended, macro_ctx, detections)

        # 7c-3. Bağlamsal Skor Yönlendirme — yanlış ışık/renk atamaası → doğru koku transfer
        blended = self._contextual_score_routing(blended, lighting, macro_ctx)

        # 7c-4. Doğa Gökyzü Veto — üst %40 beyaz piksel oranı yüksekse Vanilya buda
        if macro_ctx == "NATURE":
            blended = self._nature_sky_veto(frame, blended)

        # 7d. Baskınlık Hiyerarşisi — güçlü koku zayıf kokuları maskele
        blended = self._apply_dominance_hierarchy(blended)

        # 8. Spesifik kural bazlı bağlam süzgeci (araç, deniz aracı vb.)
        blended = self._context_processor(blended, norm_frame, detections)

        # Final normalizasyon — tüm kural adımları sonrası toplam %100 garantisi
        blended = self._safe_normalize(blended, min_total=0.0)  # KRİTİK-4 final

        # 9. Histerezis kapısı — STATE: ham blended_4way, ÇIKIŞ: post-VCV blended
        blended = self._apply_hysteresis(blended, raw_scores=blended_4way)

        # 10. N-kare oylama
        self._update_voting_buffer(blended)
        voted = self._get_voted_scores(blended)

        # ── 10b. PHASE-3: Final Tamer Guard (Kalman Öncesi Kesin Bariyer) ─────────
        # SORUN: Post-Fusion Hard VETO (Bayes sonrası) doğru çalışır, ancak
        #   7-VCV / 7-YC / 7b / 7c / 7d / 8 adımları VETO kanallarını yeniden canlandırabilir.
        #   Kalman filtresi geçmiş hafıza (x, P) ile sızıntı yaratır ("residual smell").
        #   Histerezis active_floor VETO kanalına PWM tabanı ekler → koku sızar.
        # ÇÖZÜM: Kalman'ın gördüğü ölçüm ve tüm hafıza kaynakları burada mutlak sıfırlanır.
        _vetoed_scents: set[str] = set()
        if _tamer_mode == "Aksiyon":
            _vetoed_scents = {"Vanilya", "Nane", "Limon"}
        elif _tamer_mode == "Kurtarıcı+Emniyet":
            _vetoed_scents = {"Vanilya", "Nane"}
        elif _tamer_mode == "Kentsel":
            _vetoed_scents = {"Vanilya"}

        if _vetoed_scents:
            _fg_raw_voted = dict(voted)  # Soft-gate için VETO öncesi kopyası
            # Kalman hafızası + histerezis her zaman sıfırlanır (VETO sızıntı kesme)
            for _vs in _vetoed_scents:
                blended[_vs] = 0.0                   # histerezis floor baypas
                self._kalman.filters[_vs].reset()     # x=0, P=1 → geçmiş silindi
                self._hysteresis_state[_vs] = False   # active_floor uygulanmasın

            # Hard-VETO: seçili kanallar sıfır + anti-inflate normalize
            _voted_hard = {s: (0.0 if s in _vetoed_scents else _fg_raw_voted[s]) for s in SCENTS}
            _voted_hard = self._anti_inflate_normalize(_voted_hard, _vetoed_scents)

            # ── Feature 3: Cognitive Soft-Gate ───────────────────────────────────
            # Final_Blended = (vlm_conf × VLM_Masked) + ((1 - vlm_conf) × Bayes_Output)
            # vlm_conf=1.0 → tam VETO (VLM hâkim)
            # vlm_conf=0.0 → saf Bayes çıkışı (istatistiksel füzyon hâkim)
            _gate_w = max(0.0, min(1.0, _vlm_conf))
            voted = {
                s: _gate_w       * _voted_hard.get(s, 0.0)
                 + (1.0 - _gate_w) * _fg_raw_voted.get(s, 0.0)
                for s in SCENTS
            }
            voted = self._safe_normalize(voted, min_total=0.0)

            # ── [MATH GUARD] — ham/maskeli/soft-gate değer izleme logu ──────────
            _top_fg = max(voted, key=voted.get) if any(voted.values()) else "—"
            _log.info(
                "[MATH GUARD] kare=%d | Mod: %-20s | VETO: %-26s | "
                "raw_voted: Is/Duman=%.3f Cam=%.3f Toprak=%.3f | "
                "soft_gate(w=%.2f): Is/Duman=%.3f Cam=%.3f Toprak=%.3f | "
                "top=%s(%.3f) | ZeroBlackout=%s",
                self._ovl_frame_counter, _tamer_mode,
                str(sorted(_vetoed_scents)),
                _fg_raw_voted.get("\u0130s/Duman", 0.0),
                _fg_raw_voted.get("\u00c7am",      0.0),
                _fg_raw_voted.get("Toprak",         0.0),
                _gate_w,
                voted.get("\u0130s/Duman", 0.0),
                voted.get("\u00c7am",      0.0),
                voted.get("Toprak",         0.0),
                _top_fg,
                voted.get(_top_fg, 0.0) if _top_fg != "—" else 0.0,
                not any(voted.values()),
            )

        # ── PHASE-4: [FUSION CORE] Birleşik Debug Logu ──────────────────────────
        if _tamer_mode != "Normal" or self._ovl_frame_counter % 30 == 1:
            _top_pv = max(voted, key=voted.get) if voted else "—"
            _log.info(
                "[FUSION CORE] kare=%d | Mod: %-20s | VETO: %-26s | "
                "PostVeto_Top: %s(%.3f) | Vanilya: %.3f Nane: %.3f Is/Duman: %.3f | "
                "vlm_conf: %.2f clip_conf: %.2f",
                self._ovl_frame_counter, _tamer_mode,
                str(sorted(_vetoed_scents)) if _vetoed_scents else "—",
                _top_pv, voted.get(_top_pv, 0.0) if _top_pv != "—" else 0.0,
                voted.get("Vanilya",   0.0),
                voted.get("Nane",      0.0),
                voted.get("\u0130s/Duman",  0.0),
                _vlm_conf, ovl_conf,
            )

        # ── Feature 2: VLM-Driven Adaptive Kalman R-Scaling ─────────────────────
        # Gürültülü sahnelerde (Aksiyon/Kurtarıcı+Emniyet) K-Means piksel gürültüsü yüksek.
        # R artır → Kalman anlık değişime daha yavaş tepki verir, VLM bağlamını korur.
        # Sakin sahne → R normale döner (ölçüme daha çok güven, hızlı tepki).
        if not hasattr(self, "_kalman_base_r"):
            self._kalman_base_r = {s: float(self._kalman.filters[s].R) for s in SCENTS}
        _r_mult = float(
            self._cfg.get("cognitive_gate", {}).get("kalman_noise_r_multiplier", 10.0)
        )
        _is_noisy_scene = _tamer_mode in {"Aksiyon", "Kurtarıcı+Emniyet"}
        _r_scale        = _r_mult if _is_noisy_scene else 1.0
        for _ks in SCENTS:
            self._kalman.filters[_ks].R = self._kalman_base_r[_ks] * _r_scale

        # ── [COGNITIVE CORE] Log ─────────────────────────────────────────────────
        if _tamer_mode != "Normal" or self._ovl_frame_counter % 30 == 1:
            _cog_lbl = (
                "VLM_Dominant"   if _vlm_conf >= 0.70 else
                "Mixed"          if _vlm_conf >  0.30 else
                "Bayes_Dominant"
            )
            _log.info(
                "[COGNITIVE CORE] kare=%d | VLM G\u00fcven: %.2f | Gate: %-16s | "
                "Kalman R-Multiplier: %.0fx | Master Prior: %s | Mod: %s",
                self._ovl_frame_counter, _vlm_conf, _cog_lbl, _r_scale,
                "Aktif" if _vlm_master_prior is not None else "Pasif",
                _tamer_mode,
            )

        # 11. V3 Katman D — Kalman Filtresi (EMA'nın yerini alır)
        #   Q (işlem gürültüsü) + R (ölçüm gürültüsü) ayrımı → jitter < %3
        #   Hot-reload: config değişince reload_config() Q/R'yi günceller.
        smoothed = self._kalman.update(voted)

        # 11b. Histerezis aktif kanallar için dinamik PWM taban garantisi
        # floor = base_floor + voted[s] × scale
        # Yüksek matris skoru → yüksek taban; düşük matris skoru → düşük taban.
        _hyst_floor_base  = float(cfg.get("hysteresis", {}).get("active_floor_score", 0.15))
        _hyst_floor_scale = float(cfg.get("hysteresis", {}).get("active_floor_scale", 1.0))
        _hyst_act         = float(cfg.get("hysteresis", {}).get("activate_threshold", 0.20))
        if _hyst_floor_base > 0:
            for _sf in SCENTS:
                # Floor sadece aktif VE post-VCV skoru eşiğin üzerindeyken uygulanır.
                # Ölü-band (0.08-0.20) veya baskılanmış kanallar için floor'u engelle → latch önleme.
                if self._hysteresis_state[_sf] and blended.get(_sf, 0.0) >= _hyst_act:
                    _dynamic_floor = _hyst_floor_base + voted.get(_sf, 0.0) * _hyst_floor_scale
                    smoothed[_sf]  = max(smoothed[_sf], _dynamic_floor)

        # 12. Doygunluk filtresi (UI tam göster, hardware attenuation ayrı)
        ui_scores, attenuation = self._saturation.update_and_apply(smoothed)

        # ── Post-Bayes Clamping: strict_scent_allowed=False ──────────────────────
        # SORUN: uniform SOFT_SUPPRESSION(0.02) Bayes log-uzayında tüm kanallara
        # eşit log(0.02) ekler → normalizasyon sırasında iptal olur → sıfır etki!
        # ÇÖZÜM: Bayes+Kalman hesaplaması bittikten SONRA çıkış kapısında mutlak boğma.
        # strict_ok=False → VLM 'koku yok' kararı → PWM çıkışına ulaşmasın.
        _clamp_applied = False
        if not _strict_ok:
            ui_scores     = {s: v * _SOFT_SUPPRESSION for s, v in ui_scores.items()}
            attenuation   = dict(attenuation)  # kopyala — referans paylaşımı olmasın
            _clamp_applied = True
        if _clamp_applied or (self._ovl_frame_counter % 90 == 1):
            _log.info(
                "[Post-Clamp Debug] Sinyal Bogma Uygulandı: %s | Nihai Nane: %.3f | strict_ok: %s",
                _clamp_applied, ui_scores.get("Nane", 0.0), _strict_ok,
            )

        # ── PHASE-4b: [FUSION CORE PWM] Nihai Donanım Değerleri ────────────────────
        if _tamer_mode != "Normal" or self._ovl_frame_counter % 30 == 1:
            _log.info(
                "[FUSION CORE PWM] kare=%d | Mod: %-20s | "
                "Smoothed: Vanilya=%.3f Nane=%.3f Is/Duman=%.3f | "
                "UI(PWM~): Vanilya=%.3f Nane=%.3f Is/Duman=%.3f",
                self._ovl_frame_counter, _tamer_mode,
                smoothed.get("Vanilya",  0.0),
                smoothed.get("Nane",     0.0),
                smoothed.get("\u0130s/Duman", 0.0),
                ui_scores.get("Vanilya",  0.0),
                ui_scores.get("Nane",     0.0),
                ui_scores.get("\u0130s/Duman", 0.0),
            )

        # 13. Purge hızı — sahne kesimi veya ani skor atlaması
        purge_speed = self._purge_ctrl.check(
            smoothed,
            is_scene_cut,
            float(cfg.get("purge_controller", {}).get("score_jump_threshold", 0.50)),
        )

        elapsed = time.perf_counter() - t0

        dominant_scent = max(ui_scores, key=ui_scores.get) if ui_scores else "-"

        # ── Metadata NaN/Inf Guard ────────────────────────────────────────────
        # Grafik kütüphanesi (Plotly) saf finite float bekler; Kalman/Bayes
        # zincirinde teorik olarak NaN/Inf üretilebilir → dashboard çöker.
        # _sfd(): dict[str,float] → tüm değerleri finite'a döndürür.
        def _sfd(d: dict, default: float = 0.0) -> dict:
            return {
                k: (float(v) if math.isfinite(float(v)) else default)
                for k, v in d.items()
            }

        _safe_b4w      = _sfd(blended_4way)
        _safe_smoothed = _sfd(smoothed)
        _safe_ui_sc    = _sfd(ui_scores)

        metadata = {
            "module_scores": {
                "yolo":          _sfd(yolo_s),
                "kmeans":        _sfd(kmeans_s),
                "kmeans_tamed":  _sfd(_kmeans_tamed),
                "texture":       _sfd(texture_s),
                "flow":          _sfd(flow_s),
                "ovl":           _sfd(ovl_scores),
            },
            "kmeans_tamer_mode":    _tamer_mode,
            "kmeans_dyn_weights":   {s: round(v, 4) for s, v in _tamer_weights.items()},
            "score_detail": {
                s: {
                    "clip_pct":   round(ovl_scores.get(s, 0.0) * 100),
                    "yolo_pct":   round(yolo_s.get(s, 0.0) * 100),
                    "kmeans_pct": round(kmeans_s.get(s, 0.0) * 100),
                    "grid_delta": round(grid_deltas.get(s, 0.0) * 100),
                    "adj_pct":    round(_yolo_clip_adj.get(s, 0.0) * 100),
                }
                for s in SCENTS
            },
            "sky_blue_ratio":  round(sky_blue_ratio, 3),
            "sky_freshness":   round(sky_freshness,  3),
            "vcv_urban_sig":       round(_urban_sig,     3),
            "vcv_nature_sig":      round(_nature_sig,    3),
            "vcv_terrestrial":     round(ovl_validation.get("terrestrial", 0.5), 3),
            "vcv_aquatic":         round(ovl_validation.get("aquatic",     0.5), 3),
            "yolo_clip_adj":   _yolo_clip_adj,
            "ovl_confidence":  round(ovl_conf, 3),
            "ovl_coherence":   ovl_debug.get("coherence", {}),
            "green_ratio":     round(green_ratio, 3),
            "ovl_validation":  ovl_validation,
            "blended_pre_vote": blended,
            "voted":             voted,
            "hysteresis_state":  dict(self._hysteresis_state),
            "attenuation":       attenuation,
            "purge_speed":       purge_speed,
            "motion_score":      motion_score,
            "pdm_elapsed_ms":    round(elapsed * 1000, 2),
            "macro_context":     macro_ctx,
            "lighting_profile":  lighting,
            "dominant_scent":    dominant_scent,
            "blended_4way":      blended_4way,
            "vcv_debug":         _vcv_debug,
            "yolo_top3":         sorted(
                                     detections,
                                     key=lambda d: d.get("confidence", 0.0),
                                     reverse=True,
                                 )[:3],
            "clip_top3":         sorted(
                                     [(s, round(ovl_scores.get(s, 0.0), 3)) for s in SCENTS],
                                     key=lambda x: x[1],
                                     reverse=True,
                                 )[:3],
            "roi_colors_bgr":          list(roi_colors if roi_colors else colors_bgr)[:5],
            "yolo_penalty_log":        _yolo_penalty_log,
            "kmeans_neutral_filtered": _kmeans_neutral_ct,
            "urban_is_duman_bonus":    _urban_is_bonus,
            # ── V3 telemetri alanları ──────────────────────────────────────
            "v3_kalman_states":    self._kalman.get_states(),
            "v3_kalman_smoothed":  {s: round(v, 4) for s, v in _safe_smoothed.items()},
            "v3_bayes_posterior":  _safe_b4w,
            "v3_semantic_mask":    {s: round(v, 3) for s, v in _semantic_mask.items()},
            "is_abstract_or_void":  False,
            "void_purge_triggered": False,
            "vlm_strict_ok":        _strict_ok,
            "post_clamp_applied":   _clamp_applied,
            "v3_vlm_status":      self._vlm.get_status() if self._vlm is not None else {"vlm_status": "disabled"},
            "v3_module_pdfs":     {
                "yolo":    {s: round(yolo_s.get(s, 0.0), 3) for s in SCENTS},
                "kmeans":  {s: round(kmeans_s.get(s, 0.0), 3) for s in SCENTS},
                "texture": {s: round(texture_s.get(s, 0.0), 3) for s in SCENTS},
                "clip":    {s: round(ovl_scores.get(s, 0.0), 3) for s in SCENTS},
            },
        }

        return ui_scores, metadata

    # ------------------------------------------------------------------
    # Durum Sıfırlama
    # ------------------------------------------------------------------
    def soft_reset(self):
        """Sahne kesimi için yumuşak sıfırlama.

        V3: Kalman filtre durumu (x, P) ve Bayes prior sıfırlanır (PRD 5.4).
        _clip_ever_run KORUNUR → warmup lock yeniden devreye girmez.
        """
        self._hysteresis_state = {s: False for s in SCENTS}
        for buf in self._vote_buffer.values():
            buf.clear()
        self._prev_flow_gray      = None
        self._memory.reset()
        self._vcv_prev_urban      = 0.5
        self._kmeans_prev_centers = None
        with self._ovl_lock:
            self._ovl_cache      = {s: 0.0 for s in SCENTS}
            self._ovl_confidence = 0.0
            self._ovl_validation = {"water": 0.5, "outdoor": 0.5, "terrestrial": 0.5, "aquatic": 0.5}
            self._ovl_debug      = {}
        self._ovl_frame_counter = 0
        # V3 sıfırlamalar (PRD 5.4 — on_scene_cut)
        self._kalman.reset()
        self._bayes.reset_prior()
        self._last_semantic_mask     = dict(DEFAULT_MASK)
        self._prev_mask_was_restricted = False  # bypass geçiş takibini sıfırla
        try:
            self._ovl_run_lock.release()
        except RuntimeError:
            pass

    def reset(self):
        self._hysteresis_state  = {s: False for s in SCENTS}
        for buf in self._vote_buffer.values():
            buf.clear()
        self._prev_flow_gray = None
        self._memory.reset()
        self._saturation.reset()
        self._purge_ctrl.reset()
        with self._ovl_lock:
            self._ovl_cache      = {s: 0.0 for s in SCENTS}
            self._ovl_confidence = 0.0
            self._ovl_validation = {"water": 0.5, "outdoor": 0.5, "terrestrial": 0.5, "aquatic": 0.5}
            self._ovl_debug      = {}
        self._ovl_frame_counter  = 0
        self._clip_ever_run      = False  # Warmup Lock: yeniden başlatmada kilitle
        self._vcv_prev_urban     = 0.5
        self._kmeans_prev_centers = None  # ORTA-2: warm start sıfırla
        # V3 tam sıfırlamalar
        self._kalman.reset()
        self._bayes.reset_prior()
        self._last_semantic_mask       = dict(DEFAULT_MASK)
        self._prev_mask_was_restricted = False  # bypass geçiş takibini sıfırla
        # ORTA-2: _ovl_run_lock'u serbest konuma getir (reset sonrası yeni thread başlayabilsin)
        try:
            self._ovl_run_lock.release()
        except RuntimeError:
            pass  # zaten serbest ise ignore
