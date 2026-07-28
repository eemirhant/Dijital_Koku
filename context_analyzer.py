"""
context_analyzer.py — DijitalKoku V3 Katman B: VLM Semantik Kilit

ContextAnalyzerThread:
    Asenkron Ollama/moondream2 bağlam analizcisi.
    Ana video döngüsünü bloklamamak için daemon thread olarak çalışır.

Fallback davranışı:
    VLM 'fallback_timeout_seconds' kadar yanıt vermezse DEFAULT_MASK devreye girer.
    Bu durum metadata'ya 'vlm_status: stale' olarak yazılır.
    Sistem çalışmaya devam eder — VLM olmadan Bayes füzyon çalışır.

Teknoloji: Ollama API (tamamen yerel, http://localhost:11434) — moondream2
"""

from __future__ import annotations

import base64
import json
import logging
import threading
import time
from queue import Empty, Full, Queue

import cv2
import numpy as np

from engine import SCENTS

_log = logging.getLogger(__name__)

try:
    import requests as _req
    _REQUESTS_AVAILABLE = True
except ImportError:
    _req = None                 # type: ignore[assignment]
    _REQUESTS_AVAILABLE = False

# ---------------------------------------------------------------------------
# Varsayılan maske — tüm koku kanalları izinli
# ---------------------------------------------------------------------------
DEFAULT_MASK: dict[str, float] = {s: 1.0 for s in SCENTS}

# ---------------------------------------------------------------------------
# VLM Prompt — moondream2 için kısa ve yapısal (kısa prompt = daha tutarlı JSON)
# ---------------------------------------------------------------------------
CONTEXT_PROMPT = (
    'Analyze this image and respond ONLY with a valid JSON object. '
    'No explanations, no markdown, no extra text.\n\n'
    'Required JSON structure:\n'
    '{\n'
    '  "environment": "<one of: urban_historic, urban_modern, indoor_domestic, '
    'indoor_commercial, nature_forest, nature_water, nature_open, vehicle_interior, unknown>",\n'
    '  "organic_vegetation": <true/false>,\n'
    '  "water_body": <true/false>,\n'
    '  "fire_smoke": <true/false>,\n'
    '  "is_abstract_or_void": <true/false>,\n'
    '  "strict_scent_allowed": <true/false>,\n'
    '  "dominant_materials": ["<material1>", "<material2>"],\n'
    '  "confidence": <0.0-1.0>\n'
    '}\n\n'
    'Set is_abstract_or_void to true ONLY if the scene is: purely abstract art, '
    'a dream/surreal sequence with no real objects, entirely graphics or text overlays, '
    'a dramatic black screen, or any scene with NO connection to natural or urban odors. '
    'For all normal real-world scenes set it to false.\n\n'
    'Set strict_scent_allowed to true ONLY if the scene contains at least one concrete, '
    'physical odor source matching our 7 scents: Pine trees/forest, Sea/ocean/river, '
    'Mint/herbs/grass, Earth/soil/forest floor, Smoke/fire/soot, Grilled food/BBQ, '
    'or Spicy food/spices/market. '
    'If the scene is mostly empty sky, clouds, a blank road, an abstract background, '
    'a plain wall, or any element with NO definitive odor — set strict_scent_allowed '
    'to false. Do NOT force a scent association. When in doubt, set to false.'
)


# ---------------------------------------------------------------------------
# Semantik Maske Hesaplayıcı
# ---------------------------------------------------------------------------
# Bayes log-uzayında kesin baskı (≈ -16.1 log birim) — soyut/grafik sahnede kullanılır
_HARD_VETO: float = 1e-7
# Güçlü sönümleme — strict=false (VLM koku kaynağı bulamadı) ama saha sinyaline kapı açık
# Katman A %80+ sinyal üretirse bu çarpan aşılabilir → Bayesçi yaklaşım korunur
_SOFT_SUPPRESSION: float = 0.02
# VLM tazelik penceresi: bu süre geçerse eski VLM kararı Katman A'yı bloke etmemeli
# VLM yeni analiz üretmedi → mask zorunlu 1.0 (tam açık, veto yok)
_VLM_FRESHNESS_TIMEOUT: float = 3.0


def compute_semantic_mask(context: dict) -> dict[str, float]:
    """
    VLM JSON bağlam çıktısını koku kanalı çarpanlarına dönüştür.

    Çarpan anlamları:
        1.0        = tam izin (maske yok)
        _HARD_VETO = kesin veto — log(≈1e-7) ≈ -16.1 ile Bayes posterior'ı sıfırlar
        0 < x < 1  = yumuşak baskı

    PRD Bölüm 3.4 mantığını uygular.
    Semantik Boşluk (is_abstract_or_void=true) → tüm 7 kanal _HARD_VETO.
    """
    try:
        # ── Seviye 1: Mutlak Boşluk (soyut/grafik/rüya sahne) → HARD_VETO ─────────
        # Sadece bu durumda tüm kanallar pratikte imkansız (log ≈ -16.1) olarak işaretlenir.
        _is_abstract = bool(context.get("is_abstract_or_void", False))
        if _is_abstract:
            _log.debug("[VLM] HARD_VETO — soyut/grafik sahne, tüm 7 kanal sıfırlandı")
            return {scent: _HARD_VETO for scent in SCENTS}

        # ── Seviye 2: Strict Koku Denetimi — VLM koku kaynağı bulamadı ─────────────
        # Mutlak imkansızlık DEĞİL; Katman A (YOLO/%80+ orman vb.) kuvvetli sinyal
        # üretirse _SOFT_SUPPRESSION çarpanı aşılabilir — Bayesçi esneklik korunur.
        _strict_ok = bool(context.get("strict_scent_allowed", False))  # fail-safe: False
        if not _strict_ok:
            _log.debug("[VLM] SOFT_SUPPRESSION — strict=false, tüm 7 kanal x%.2f", _SOFT_SUPPRESSION)
            return {scent: _SOFT_SUPPRESSION for scent in SCENTS}

        mask      = {scent: 1.0 for scent in SCENTS}
        env       = context.get("environment", "unknown")
        has_veg   = context.get("organic_vegetation", True)   # şüphede izin ver
        has_water = context.get("water_body", False)
        has_fire  = context.get("fire_smoke", False)

        # ── Botanik filtreler ────────────────────────────────────────────────
        if not has_veg:
            mask["Çam"]    = _HARD_VETO   # bitki yok → çam kesin imkansız
            mask["Nane"]   *= 0.15        # yumuşak baskı (nane iç mekanda olabilir)
            mask["Toprak"] *= 0.20        # orta baskı (toprak her yerde)

        # ── Kentsel çevre düzeltmeleri ──────────────────────────────────────
        if env in ("urban_historic", "urban_modern"):
            if not has_water:
                mask["Deniz"] = _HARD_VETO  # kentsel + su yok → Deniz kesin imkansız
            if not has_veg:
                mask["Çam"]    = _HARD_VETO  # bitki yok → çam kesin imkansız
                mask["Toprak"] *= 0.10       # beton/kaldırım → toprak yok
                mask["Nane"]   *= 0.10       # bitki yok → nane yok
            else:
                # Kentsel + organik bitki (şehir parkı / ağaçlı cadde)
                # Kaldırım kenarındaki birkaç ağaç baskın doğa kokusu üretmemeli.
                # Güçlü sönümleme: YOLO/K-Means kuvvetli sinyal olmadan eşik aşılamaz.
                mask["Çam"]    *= 0.10   # şehir içi ağaçlar → orman kokusu değil
                mask["Toprak"] *= 0.15   # beton/asfalt → toprak kokusu yok
                mask["Nane"]   *= 0.40   # nane kentsel çevrede sınırlı
                mask["Deniz"]  *= 0.10   # kentsel + su yok ama has_water kontrolü geçtiyse bile sınırla

        # ── Yangın / Duman doğrulaması ────────────────────────────────────────
        if has_fire:
            mask["İs/Duman"] = min(mask["İs/Duman"] * 2.0, 1.0)  # güçlendir
        elif env == "nature_water":
            mask["İs/Duman"] = _HARD_VETO   # su ortamında duman kesin imkansız

        # ── Su ortamı güçlendirmesi ─────────────────────────────────────────
        if has_water or env == "nature_water":
            if mask["Deniz"] > _HARD_VETO:    # hard veto'yu güçlendirme ezmesin
                mask["Deniz"] = min(mask["Deniz"] * 1.5, 1.0)

        # ── Kentsel Semantik Kapılar (Vanilya + Nane) ─────────────────────────
        # Outdoor urban: bej bina/kaldırım Vanilya'yı, kaldırım ağaçları Nane'yi tetiklememeli.
        # Bakery/café → indoor_commercial sınıflanır, bu blok çalışmaz.
        if env in ("urban_historic", "urban_modern"):
            mask["Vanilya"] = min(mask["Vanilya"], 0.01)  # bina yüzeyi → Vanilya ≠
            mask["Nane"]    = min(mask["Nane"],    0.05)  # kaldırım ağacı → Nane ≠

        # ── İç mekan düzeltmeleri (yumuşak baskı) ────────────────────────────
        if env in ("indoor_domestic", "indoor_commercial"):
            mask["Çam"]    *= 0.45
            mask["Deniz"]  *= 0.40
            mask["Toprak"] *= 0.50

        return mask

    except Exception as exc:  # Hatalı VLM çıktısında sistem durmamalı
        _log.warning("[VLM] compute_semantic_mask beklenmeyen hata: %s — DEFAULT_MASK", exc)
        return dict(DEFAULT_MASK)


# ---------------------------------------------------------------------------
# ContextAnalyzerThread
# ---------------------------------------------------------------------------
class ContextAnalyzerThread(threading.Thread):
    """
    Asenkron VLM bağlam analizcisi — Katman B.

    Sadece en güncel kareyi analiz eder (kuyruk doluysa eski kare atılır).
    Ollama yanıt vermezse son geçerli maske korunur; timeout aşılırsa
    DEFAULT_MASK devreye girer.
    """

    def __init__(
        self,
        ollama_url:       str   = "http://localhost:11434",
        model:            str   = "moondream2",
        interval_frames:  int   = 20,
        fallback_timeout: float = 10.0,
        max_retries:      int   = 2,
        enabled:          bool  = True,
    ):
        super().__init__(daemon=True, name="ContextAnalyzer")

        self.ollama_url      = ollama_url.rstrip("/")
        self.model           = model
        self.interval_frames = max(1, interval_frames)
        self.fallback_timeout = fallback_timeout
        self.max_retries     = max_retries
        self.enabled         = enabled and _REQUESTS_AVAILABLE

        self._lock           = threading.RLock()
        self._stop_event     = threading.Event()
        self._frame_queue: Queue = Queue(maxsize=1)

        # Paylaşılan durum (thread-safe — _lock altında okunur/yazılır)
        self._context_mask:    dict[str, float] = dict(DEFAULT_MASK)
        self._last_context:    dict             = {}
        self._last_valid_ts:   float            = 0.0
        self._start_ts:        float            = time.perf_counter()  # başlangıç stale için
        self._vlm_status:      str              = "disabled" if not self.enabled else "init"
        self._parse_errors:    int              = 0
        self._total_calls:     int              = 0
        self._successful_calls: int             = 0
        # Birleşik Kokusuz Mod durumu
        self._is_void:           bool = False  # is_abstract_or_void
        self._is_strict_allowed: bool = True   # strict_scent_allowed (warmup'ta True = izinli)
        self._void_trigger:      bool = False  # birleşik yükselen kenar — tek seferlik purge
        # GIL-atomik snapshot — lock meşgulken non-blocking okuma için
        # Python referans ataması GIL altında atomiktir; lock gerektirmez.
        self._snapshot: tuple = (dict(DEFAULT_MASK), False, True)  # (mask, is_void, strict_ok)
        # Debug log hız sınırlayıcı — her ~90 çağrıda bir konsola yazdır (~3s @ 30fps)
        self._dbg_ctr: int = 0

        if not _REQUESTS_AVAILABLE:
            _log.warning(
                "[VLM] 'requests' paketi bulunamadı — ContextAnalyzer devre dışı. "
                "Yüklemek için: pip install requests"
            )

    # ------------------------------------------------------------------
    # Dış API
    # ------------------------------------------------------------------
    def submit_frame(self, frame: np.ndarray, frame_count: int) -> None:
        """
        Ana döngüden çağrılır.
        interval_frames modulo 0 olduğunda kareyi kuyruğa ekler.
        Kuyruk doluysa eski kare atılır (en güncel kare her zaman kazanır).
        """
        if not self.enabled:
            return
        if frame_count % self.interval_frames != 0:
            return
        # Kuyruk doluysa eski kareyi çıkar
        try:
            self._frame_queue.get_nowait()
        except Empty:
            pass
        try:
            self._frame_queue.put_nowait(frame.copy())
        except Full:
            pass

    def get_void_state(self) -> bool:
        """Thread-safe (non-blocking): mutlak boşluk (is_abstract_or_void) durumunu döndür.

        NOT: strict_scent_allowed bu durumu etkilemez — o sadece maskeyi yumuşatır,
        PDM erken çıkışı tetiklemez.
        """
        # GIL-atomik snapshot okuma — lock beklememek için
        return self._snapshot[1]

    def consume_void_trigger(self) -> bool:
        """
        Yükselen kenar sinyalini atomik oku-ve-temizle.
        Boşluk ilk aktif olduğu anda True döner, sonraki çağrılarda False.
        """
        with self._lock:
            triggered = self._void_trigger
            self._void_trigger = False
            return triggered

    def get_mask_and_void_state(self) -> tuple[dict[str, float], bool, bool, bool]:
        """
        ATOMIK dörtlü okuma: semantik maske + is_void + void trigger + strict_scent_allowed.

        PDMEngine bunu tek lock altında okur —
        get_current_mask() / get_void_state() / consume_void_trigger() üç ayrı
        çağrıdan kaynaklanan race condition'u ortadan kaldırır.

        Stale durumda: (DEFAULT_MASK, False, False, True)
            → VLM zaman aşımında void modundan da güvenli çıkış

        Returns:
            (mask, is_void, void_triggered, strict_scent_allowed)
        """
        if not self.enabled:
            return dict(DEFAULT_MASK), False, False, True

        now = time.perf_counter()

        # ── Başlangıç stale: VLM hiç yanıt vermedi ────────────────────────────────
        # _last_valid_ts == 0 ise fallback_timeout'u başlangıçtan ölç
        if self._last_valid_ts == 0 and (now - self._start_ts) > self.fallback_timeout:
            return dict(DEFAULT_MASK), False, False, True

        # ── Tazelik bypass: VLM eski kararı Katman A'yı bloke etmesin ────────────
        # Son VLM yanıtı _VLM_FRESHNESS_TIMEOUT saniyeden eskiyse mask=1.0 (tam açık).
        # VLM hâlâ eski 'urban' kararında donmuşsa Katman A serbest kalır.
        if self._last_valid_ts > 0:
            _age = now - self._last_valid_ts
            if _age > _VLM_FRESHNESS_TIMEOUT:
                _bypass_mask = {s: 1.0 for s in SCENTS}
                _log.info(
                    "[VLM Mask Debug] FRESHNESS BYPASS — VLM %.1fs eski, mask=1.0 zorlandı"
                    " | Cam:1.00 Deniz:1.00 Toprak:1.00 is_void:False",
                    _age,
                )
                return _bypass_mask, False, False, True  # bypass = tam açık = strict_ok=True

        # ── Non-blocking acquire: ana döngü VLM thread'ini asla BEKLEMESİN ──────
        # Lock meşgulse (VLM state yazıyor) — GIL-atomik snapshot döndür
        if not self._lock.acquire(blocking=False):
            snap = self._snapshot          # GIL altında atomik referans okuma
            return dict(snap[0]), snap[1], False, snap[2]  # trigger tüketilmez

        try:
            if self._last_valid_ts > 0:
                age = now - self._last_valid_ts
                if age > self.fallback_timeout:
                    if self._vlm_status != "stale":
                        _log.warning(
                            "[VLM] Stale timeout (·%.0fs) — DEFAULT_MASK + void=False",
                            self.fallback_timeout,
                        )
                        self._vlm_status    = "stale"
                    self._is_void           = False
                    self._is_strict_allowed = True
                    self._void_trigger      = False
                    self._snapshot          = (dict(DEFAULT_MASK), False, True)
                    return dict(DEFAULT_MASK), False, False, True

            triggered          = self._void_trigger
            self._void_trigger = False
            # Trigger tüketildi — snapshot'u senkronize et (3-tuple: mask, is_void, strict_ok)
            self._snapshot = (dict(self._context_mask), self._is_void, self._is_strict_allowed)

            # ── Debug log (hız sınırlı: ~3s aralıklı) ─────────────────────────────
            self._dbg_ctr += 1
            if self._dbg_ctr % 90 == 1:
                _m = self._context_mask
                _log.info(
                    "[VLM Mask Debug] Cam:%.3f Deniz:%.3f Vanilya:%.3f "
                    "Is:%.3f Nane:%.3f Toprak:%.3f Limon:%.3f | "
                    "is_void:%s strict:%s vlm_age:%.1fs",
                    _m.get("Çam",      1.0), _m.get("Deniz",    1.0),
                    _m.get("Vanilya",   1.0), _m.get("İs/Duman",  1.0),
                    _m.get("Nane",      1.0), _m.get("Toprak",   1.0),
                    _m.get("Limon",     1.0),
                    self._is_void, self._is_strict_allowed,
                    (now - self._last_valid_ts) if self._last_valid_ts > 0 else -1,
                )

            return dict(self._context_mask), self._is_void, triggered, self._is_strict_allowed
        finally:
            self._lock.release()

    def get_current_mask(self) -> dict[str, float]:
        """
        Bayes motoru tarafından her karede çağrılır. Non-blocking.

        Stale timeout aşılmışsa DEFAULT_MASK döner ve 'vlm_status = stale'
        olarak işaretlenir.
        """
        if not self.enabled:
            return dict(DEFAULT_MASK)

        now = time.perf_counter()

        # Tazelik bypass — get_mask_and_void_state ile tutarlı davranış
        if self._last_valid_ts > 0 and (now - self._last_valid_ts) > _VLM_FRESHNESS_TIMEOUT:
            return {s: 1.0 for s in SCENTS}

        if not self._lock.acquire(blocking=False):
            return dict(self._snapshot[0])  # GIL-atomik fallback

        try:
            if self._last_valid_ts > 0:
                age = now - self._last_valid_ts
                if age > self.fallback_timeout:
                    if self._vlm_status != "stale":
                        _log.warning(
                            "[VLM] Stale timeout aşıldı (%.0fs) — DEFAULT_MASK devreye girdi",
                            self.fallback_timeout,
                        )
                        self._vlm_status = "stale"
                    return dict(DEFAULT_MASK)
            return dict(self._context_mask)
        finally:
            self._lock.release()

    @property
    def vlm_confidence(self) -> float:
        """
        VLM son analiz güveni — GIL-atomik okuma, lock gerektirmez.
        Ollama JSON yanıtındaki 'confidence' alanını döndürür.
        VLM hiç çalışmadıysa veya ayrıştırma başarısızsa 0.0 döner.
        """
        return float(self._last_context.get("confidence", 0.0))

    def get_status(self) -> dict:
        """Telemetri / dashboard için VLM durum özeti."""
        with self._lock:
            age = (
                round(time.perf_counter() - self._last_valid_ts, 1)
                if self._last_valid_ts > 0
                else -1
            )
            parse_rate = (
                round(self._successful_calls / max(1, self._total_calls), 3)
                if self._total_calls > 0
                else 0.0
            )
            return {
                "vlm_status":         self._vlm_status,
                "last_context":       dict(self._last_context),
                "context_mask":       dict(self._context_mask),
                "parse_errors":       self._parse_errors,
                "total_calls":        self._total_calls,
                "successful_calls":   self._successful_calls,
                "parse_success_rate": parse_rate,
                "last_valid_age_s":   age,
                "enabled":            self.enabled,
                "model":              self.model,
                "is_abstract_or_void":  self._is_void,
                "strict_scent_allowed": self._is_strict_allowed,
            }

    def stop(self) -> None:
        self._stop_event.set()

    # ------------------------------------------------------------------
    # Thread döngüsü
    # ------------------------------------------------------------------
    def run(self) -> None:
        if not self.enabled:
            return
        _log.info(
            "[VLM] ContextAnalyzerThread başladı — model=%s url=%s interval=%d kare",
            self.model, self.ollama_url, self.interval_frames,
        )
        while not self._stop_event.is_set():
            try:
                frame = self._frame_queue.get(timeout=1.0)
            except Empty:
                continue
            self._analyze_frame(frame)

    # ------------------------------------------------------------------
    # Analiz motoru
    # ------------------------------------------------------------------
    def _analyze_frame(self, frame: np.ndarray) -> None:
        """Kareyi Ollama'ya gönder, JSON yanıtı parse et, maskeyi güncelle."""
        self._total_calls += 1

        img_b64 = self._encode_frame(frame)
        if img_b64 is None:
            return

        context = None
        try:
            context = self._call_ollama(img_b64)
            if context is None:
                return

            mask         = compute_semantic_mask(context)
            _new_void    = bool(context.get("is_abstract_or_void", False))
            _new_strict  = bool(context.get("strict_scent_allowed", False))  # fail-safe: False
            with self._lock:
                self._context_mask     = mask
                self._last_context     = context
                self._last_valid_ts    = time.perf_counter()
                self._vlm_status       = "ok"
                self._successful_calls += 1

                # Purge yalnızca MUTLAK BOŞ (soyut/grafik) sahne geçişinde tetiklenir.
                # strict_scent_allowed dalgalanmaları purge açmaz — sadece soft mask uygular.
                if _new_void and not self._is_void:
                    self._void_trigger = True
                    _log.info("[VLM] Mutlak Boşluk TETİKLENDİ (abstract) — purge sinyali")
                elif not _new_void and self._is_void:
                    _log.info("[VLM] Mutlak Boşluk sona erdi — normal mod")

                if not _new_strict and self._is_strict_allowed:
                    _log.debug("[VLM] Katı Koku Denetimi → SOFT_SUPPRESSION devreye girdi")
                elif _new_strict and not self._is_strict_allowed:
                    _log.debug("[VLM] Katı Koku Denetimi sona erdi — normal mask")

                self._is_void           = _new_void
                self._is_strict_allowed = _new_strict
                # GIL-atomik snapshot güncelle — non-blocking okuyucular tutarlı görür
                self._snapshot = (dict(mask), _new_void, _new_strict)

            _log.debug(
                "[VLM] Bağlam — env=%s veg=%s water=%s fire=%s void=%s strict=%s conf=%.2f",
                context.get("environment", "?"),
                context.get("organic_vegetation", "?"),
                context.get("water_body", "?"),
                context.get("fire_smoke", "?"),
                _new_void,
                _new_strict,
                float(context.get("confidence", 0.0)),
            )

        except Exception as exc:
            # Herhangi bir istisnada lock açık kalmaz (with-block garantisi).
            # Sistem fail-safe moda çekilir: koku akışı durmasın.
            _log.warning(
                "[VLM] _analyze_frame beklenmeyen hata: %s — fail-safe state sıfırlandı", exc
            )
            with self._lock:
                self._vlm_status        = "error"
                self._is_void           = False
                self._is_strict_allowed = True
                self._void_trigger      = False
                self._context_mask      = dict(DEFAULT_MASK)
                self._snapshot          = (dict(DEFAULT_MASK), False, True)

    @staticmethod
    def _encode_frame(frame: np.ndarray) -> str | None:
        """BGR kareyi 512×512 JPEG base64 string'e dönüştür."""
        try:
            small = cv2.resize(frame, (512, 512), interpolation=cv2.INTER_LINEAR)
            rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            ok, buf = cv2.imencode(".jpg", rgb, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                raise ValueError("JPEG encode başarısız")
            return base64.b64encode(buf.tobytes()).decode("utf-8")
        except Exception as exc:
            _log.warning("[VLM] Kare encode hatası: %s", exc)
            return None

    def _call_ollama(self, img_b64: str) -> dict | None:
        """
        Ollama /api/generate uç noktasına istek gönder.
        Başarısız olursa max_retries kadar yeniden dene.
        """
        url = f"{self.ollama_url}/api/generate"
        payload = {
            "model":  self.model,
            "prompt": CONTEXT_PROMPT,
            "images": [img_b64],
            "stream": False,
            "options": {
                "temperature":  0.1,
                "num_predict":  256,
                "stop":         ["\n\n"],
            },
        }

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = _req.post(
                    url, json=payload,
                    timeout=(5.0, float(self.fallback_timeout)),  # (bağlantı, okuma)
                )
                resp.raise_for_status()
                raw = resp.json().get("response", "")
                ctx = _parse_json_response(raw)
                if ctx is not None:
                    return ctx
                with self._lock:
                    self._parse_errors += 1
                _log.debug(
                    "[VLM] JSON parse hatası (deneme %d/%d): %.120r",
                    attempt, self.max_retries, raw,
                )
            except Exception as exc:
                _log.debug(
                    "[VLM] Ollama isteği başarısız (deneme %d/%d): %s",
                    attempt, self.max_retries, exc,
                )

        with self._lock:
            self._vlm_status = "error"
        return None


# ---------------------------------------------------------------------------
# JSON parse yardımcısı (modül düzeyinde — paylaşılabilir)
# ---------------------------------------------------------------------------
def _parse_json_response(text: str) -> dict | None:
    """Ham Ollama yanıtından geçerli JSON bağlam dict'i çıkar."""
    if not text:
        return None

    text = text.strip()

    # Markdown kod bloğu temizle
    if "```" in text:
        start = text.find("{")
        end   = text.rfind("}") + 1
        if start >= 0 < end:
            text = text[start:end]

    # Doğrudan parse dene
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Ekstra metin varsa JSON bloğunu bul
        start = text.find("{")
        end   = text.rfind("}") + 1
        if start < 0 or end <= start:
            return None
        try:
            data = json.loads(text[start:end])
        except json.JSONDecodeError:
            return None

    # Zorunlu alan kontrolü
    if "environment" not in data:
        return None

    # Bool alanlarını normalleştir (bazen string gelir)
    for field in ("organic_vegetation", "water_body", "fire_smoke"):
        val = data.get(field)
        if isinstance(val, str):
            data[field] = val.lower() in ("true", "1", "yes")
        elif not isinstance(val, bool):
            data[field] = bool(val) if val is not None else True
    # is_abstract_or_void: None/eksik → False (şüphede void ETKİNLEŞMEİN)
    _void_raw = data.get("is_abstract_or_void")
    if isinstance(_void_raw, str):
        data["is_abstract_or_void"] = _void_raw.lower() in ("true", "1", "yes")
    elif not isinstance(_void_raw, bool):
        data["is_abstract_or_void"] = False  # None veya bilinmeyen → void DEGİL
    # strict_scent_allowed: None/eksik → False (fail-safe: şüphede koku izni VERME)
    _strict_raw = data.get("strict_scent_allowed")
    if isinstance(_strict_raw, str):
        data["strict_scent_allowed"] = _strict_raw.lower() in ("true", "1", "yes")
    elif not isinstance(_strict_raw, bool):
        data["strict_scent_allowed"] = False  # None veya bilinmeyen → koku izni YOK

    # confidence [0,1] aralığına zorla
    if "confidence" in data:
        try:
            data["confidence"] = float(max(0.0, min(1.0, data["confidence"])))
        except (TypeError, ValueError):
            data["confidence"] = 0.5

    return data
