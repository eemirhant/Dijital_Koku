"""
window_worker.py — Arka plan pencere yakalama ve headless analiz.

Windows PrintWindow API'si ile hedef pencereyi ekranda görünmeden yakalar
(minimize edilmiş veya başka sekmenin arkasında olsa bile).
Sonuçları thread-safe Queue üzerinden Dashboard'a iletir.

Gereksinim: pip install pywin32
"""

import ctypes
import logging
import queue
import threading
import time
from typing import Optional

import cv2
import numpy as np

try:
    import win32gui
    import win32ui
    _WIN32_AVAILABLE = True
except ImportError:
    _WIN32_AVAILABLE = False

PW_RENDERFULLCONTENT = 0x00000002  # GPU hızlandırmalı pencereler (Chrome, Edge, vb.)
_MAX_CAPTURE_FAILS   = 10           # Ardışık bu kadar hata olursa worker otomatik durur

log = logging.getLogger("window_worker")


# ---------------------------------------------------------------------------
# Yardımcı fonksiyonlar
# ---------------------------------------------------------------------------

def capture_window_hwnd(hwnd: int) -> Optional[np.ndarray]:
    """
    PrintWindow API ile bir pencereyi arka planda yakalar.

    Parameters
    ----------
    hwnd : int
        Win32 pencere tanımlayıcısı.

    Returns
    -------
    np.ndarray | None
        BGR formatında görüntü veya hata durumunda None.
    """
    if not _WIN32_AVAILABLE or not win32gui.IsWindow(hwnd):
        return None

    left, top, right, bot = win32gui.GetWindowRect(hwnd)
    w, h = right - left, bot - top
    if w <= 0 or h <= 0:
        return None

    hwnd_dc = save_dc = mfc_dc = save_bmp = None
    try:
        hwnd_dc  = win32gui.GetWindowDC(hwnd)
        mfc_dc   = win32ui.CreateDCFromHandle(hwnd_dc)
        save_dc  = mfc_dc.CreateCompatibleDC()
        save_bmp = win32ui.CreateBitmap()
        save_bmp.CreateCompatibleBitmap(mfc_dc, w, h)
        save_dc.SelectObject(save_bmp)

        # PW_RENDERFULLCONTENT: tarayıcılar ve GPU hızlandırmalı pencereler için
        ctypes.windll.user32.PrintWindow(hwnd, save_dc.GetSafeHdc(), PW_RENDERFULLCONTENT)

        bmpstr = save_bmp.GetBitmapBits(True)
        img = np.frombuffer(bmpstr, dtype="uint8").reshape(h, w, 4).copy()
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    except Exception:
        return None
    finally:
        if save_bmp:
            try:
                win32gui.DeleteObject(save_bmp.GetHandle())
            except Exception:
                pass
        if save_dc:
            try:
                save_dc.DeleteDC()
            except Exception:
                pass
        if mfc_dc:
            try:
                mfc_dc.DeleteDC()
            except Exception:
                pass
        if hwnd_dc:
            try:
                win32gui.ReleaseDC(hwnd, hwnd_dc)
            except Exception:
                pass


def list_capturable_windows() -> list[tuple[int, str]]:
    """
    Tüm görünür ve başlıklı pencereleri listeler.

    Returns
    -------
    list[tuple[int, str]]
        [(hwnd, title), ...] formatında pencere listesi.
    """
    if not _WIN32_AVAILABLE:
        return []
    windows: list[tuple[int, str]] = []

    def _cb(hwnd: int, _: object) -> None:
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd).strip()
            if title:
                windows.append((hwnd, title))

    win32gui.EnumWindows(_cb, None)
    return windows


def find_hwnd_by_title(title: str) -> Optional[int]:
    """Başlığa göre (kısmi eşleşme) hwnd döndürür.
    Birden fazla eşleşme varsa ilki seçilir ve uyarı loglanır."""
    title_lower = title.lower()
    matches = [(h, t) for h, t in list_capturable_windows() if title_lower in t.lower()]
    if not matches:
        return None
    if len(matches) > 1:
        log.warning(
            "multiple_hwnd_matches",
            extra={"query": title, "count": len(matches),
                   "titles": [t for _, t in matches[:5]]},
        )
    return matches[0][0]


# ---------------------------------------------------------------------------
# WindowWorker
# ---------------------------------------------------------------------------

class WindowWorker:
    """
    Arka planda bir pencereyi yakalar ve analiz eder.

    Attributes
    ----------
    result_queue : queue.Queue(maxsize=1)
        Her zaman en güncel analiz sonucunu tutar.
    status : dict
        {"frame": int, "running": bool, "error": str|None, "fps": float}

    Sonuç dict anahtarları
    ----------------------
    frame, scores, metadata, detections, colors,
    motion_score, is_scene_cut, frame_idx, total(-1), native_fps
    """

    _QUEUE_SIZE = 1

    def __init__(
        self,
        hwnd: int,
        window_title: str,
        analyzer,
        engine,
        conf_threshold: float = 0.35,
        capture_fps: float = 5.0,
        analyze_every_n: int = 3,
    ) -> None:
        self.hwnd             = hwnd
        self.window_title     = window_title
        self.analyzer         = analyzer
        self.engine           = engine
        self.conf_threshold   = conf_threshold
        self.capture_fps      = max(0.5, min(capture_fps, 60.0))
        self._analyze_every_n = max(1, analyze_every_n)

        self.result_queue: queue.Queue = queue.Queue(maxsize=self._QUEUE_SIZE)
        self.status: dict = {
            "frame":   0,
            "running": False,
            "error":   None,
            "fps":     self.capture_fps,
        }
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Arka plan yakalama thread'ini başlat."""
        if self.is_alive():
            log.warning("window_worker_already_running", extra={"hwnd": self.hwnd})
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="WindowWorker"
        )
        self._thread.start()

    def stop(self) -> None:
        """Thread'i durdur (en fazla 5 sn bekle)."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        with self._lock:
            self.status["running"] = False

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def get_status(self) -> dict:
        with self._lock:
            return dict(self.status)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _push(self, result: dict) -> None:
        """Kuyruğa ekle; dolu ise en eski öğeyi at."""
        try:
            self.result_queue.put_nowait(result)
        except queue.Full:
            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.result_queue.put_nowait(result)
            except queue.Full:
                pass

    def _run(self) -> None:
        if not _WIN32_AVAILABLE:
            with self._lock:
                self.status["error"] = (
                    "pywin32 kurulu değil. Lütfen: pip install pywin32"
                )
            log.error("window_worker_no_pywin32")
            return

        interval       = 1.0 / self.capture_fps
        frame_count    = 0
        _fail_count    = 0
        _skip_counter  = 0
        _cached_dets:  list = []
        _cached_colors: list = []
        prev_frame: Optional[np.ndarray] = None
        _fps_ema: float = self.capture_fps

        with self._lock:
            self.status.update(running=True, error=None)
        log.info("window_worker_started",
                 extra={"hwnd": self.hwnd,
                        "title": self.window_title[:50],
                        "fps": self.capture_fps,
                        "analyze_every_n": self._analyze_every_n})

        while not self._stop_event.is_set():
            t0 = time.perf_counter()

            frame = capture_window_hwnd(self.hwnd)
            if frame is None:
                _fail_count += 1
                with self._lock:
                    self.status["error"] = (
                        f"Pencere yakalanamıyor: {self.window_title[:50]}"
                    )
                log.warning("capture_failed",
                            extra={"hwnd": self.hwnd,
                                   "consecutive_fails": _fail_count})
                if _fail_count >= _MAX_CAPTURE_FAILS:
                    log.error("window_worker_auto_stop",
                              extra={"hwnd": self.hwnd,
                                     "reason": "max_fails_reached"})
                    break
                self._stop_event.wait(timeout=1.0)
                continue
            _fail_count = 0

            frame_count += 1
            with self._lock:
                self.status.update(frame=frame_count, error=None)

            try:
                _skip_counter += 1
                if _skip_counter >= self._analyze_every_n or not _cached_colors:
                    # Tam analiz: YOLO tam kare, K-Means 160x120
                    _skip_counter  = 0
                    _small = cv2.resize(frame, (160, 120), interpolation=cv2.INTER_NEAREST)
                    _cached_dets, _cached_colors, motion, scene_cut, _ = self.analyzer.analyze(
                        frame,
                        prev_frame=prev_frame,
                        conf_threshold=self.conf_threshold,
                        kmeans_frame=_small,
                    )
                else:
                    # Hızlı kare: gerçek motion hesapla, YOLO/K-Means atla
                    motion    = float(self.analyzer.get_motion_score(frame))
                    scene_cut = False

                dets   = _cached_dets
                colors = _cached_colors

                if scene_cut:
                    self.engine.soft_reset()
                    log.debug("scene_cut_detected", extra={"frame": frame_count})

                scores, meta = self.engine.decide(
                    frame, dets, colors,
                    motion_score=motion,
                    is_scene_cut=scene_cut,
                )

                self._push({
                    "frame":        frame,
                    "scores":       scores,
                    "metadata":     meta,
                    "detections":   dets,
                    "colors":       colors,
                    "motion_score": motion,
                    "is_scene_cut": scene_cut,
                    "frame_idx":    frame_count,
                    "total":        -1,
                    "native_fps":   _fps_ema,
                })
            except Exception as exc:
                log.error("window_worker_frame_error", extra={"error": str(exc)})
                with self._lock:
                    self.status["error"] = str(exc)

            prev_frame = frame

            elapsed  = time.perf_counter() - t0
            if elapsed > 0:
                _fps_ema = 0.9 * _fps_ema + 0.1 * (1.0 / elapsed)
                with self._lock:
                    self.status["fps"] = round(_fps_ema, 1)
            to_sleep = interval - elapsed
            if to_sleep > 0:
                self._stop_event.wait(timeout=to_sleep)

        with self._lock:
            self.status["running"] = False
        log.info("window_worker_stopped",
                 extra={"hwnd": self.hwnd, "frames": frame_count})
