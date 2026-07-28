"""
video_worker.py — Headless arka plan video analiz iş parçacığı.

VideoWorker, bir video dosyasını cv2.VideoCapture ile okur,
her kareye YOLO + K-Means + Optical Flow analizini uygular ve
sonuçları thread-safe bir Queue üzerinden Dashboard'a iletir.

Kullanım:
    worker = VideoWorker(path, analyzer, engine)
    worker.start()
    ...
    result = worker.result_queue.get_nowait()  # fragment içinde
    worker.stop()
"""

import logging
import math
import queue
import threading
import time

import cv2
import numpy as np

log = logging.getLogger("video_worker")


class VideoWorker:
    """
    Thread tabanlı arka plan video analiz motoru.

    Attributes
    ----------
    result_queue : queue.Queue(maxsize=1)
        Her zaman en güncel analiz sonucunu tutar.
        Dict anahtarları: frame, scores, metadata, detections,
                          colors, motion_score, is_scene_cut,
                          frame_idx, total, native_fps
    status : dict
        {"frame": int, "total": int, "fps": float,
         "running": bool, "error": str|None}
    """

    _QUEUE_SIZE = 1

    def __init__(
        self,
        video_path: str,
        analyzer,
        engine,
        conf_threshold: float = 0.35,
        analyze_every_n: int = 10,
    ) -> None:
        self.video_path       = video_path
        self.analyzer         = analyzer
        self.engine           = engine
        self.conf_threshold   = conf_threshold
        self._analyze_every_n = max(1, analyze_every_n)

        self.result_queue: queue.Queue = queue.Queue(maxsize=self._QUEUE_SIZE)
        self.status: dict = {
            "frame":   0,
            "total":   0,
            "fps":     0.0,
            "running": False,
            "error":   None,
            "loops":   0,
        }
        self._stop_event   = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock         = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Arka plan analiz thread'ini başlat."""
        if self.is_alive():
            log.warning("video_worker_already_running", extra={"path": self.video_path})
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="VideoWorker"
        )
        self._thread.start()

    def stop(self) -> None:
        """Thread'i durdur ve join et (en fazla 5 sn bekle)."""
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
    # Internal helpers
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

    # ------------------------------------------------------------------
    # Worker loop
    # ------------------------------------------------------------------
    def _run(self) -> None:
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            err = f"Video açılamadı: {self.video_path}"
            with self._lock:
                self.status["error"] = err
            log.error("video_open_failed", extra={"path": self.video_path})
            return

        raw_fps    = cap.get(cv2.CAP_PROP_FPS)
        native_fps = raw_fps if math.isfinite(raw_fps) and raw_fps > 0 else 25.0
        interval   = 1.0 / native_fps
        raw_total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total      = raw_total if raw_total > 0 else -1

        with self._lock:
            self.status.update(
                total=total, fps=native_fps, running=True, error=None
            )
        log.info("video_worker_started",
                 extra={"path": self.video_path,
                        "fps": native_fps,
                        "total": total,
                        "analyze_every_n": self._analyze_every_n})

        prev_frame:    np.ndarray | None = None
        _skip_counter: int   = 0
        _cached_dets:  list  = []
        _cached_colors: list = []
        _fps_ema:      float = native_fps
        _loop_count:   int   = 0

        try:
            while not self._stop_event.is_set():
                t0 = time.perf_counter()

                try:
                    ret, frame = cap.read()
                except Exception as exc:
                    log.error("cap_read_error", extra={"error": str(exc)})
                    with self._lock:
                        self.status["error"] = str(exc)
                    break

                if not ret:
                    _loop_count += 1
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.engine.reset()
                    prev_frame     = None
                    _skip_counter  = 0
                    _cached_dets   = []
                    _cached_colors = []
                    with self._lock:
                        self.status["loops"] = _loop_count
                    log.info("video_looped",
                             extra={"loop": _loop_count,
                                    "path": self.video_path})
                    continue

                frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                with self._lock:
                    self.status["frame"] = frame_idx

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
                        log.debug("scene_cut_detected", extra={"frame": frame_idx})

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
                        "frame_idx":    frame_idx,
                        "total":        total,
                        "native_fps":   _fps_ema,
                    })
                except Exception as exc:
                    log.error("video_worker_frame_error", extra={"error": str(exc)})
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

        finally:
            cap.release()
            with self._lock:
                self.status["running"] = False
            log.info("video_worker_stopped",
                     extra={"path": self.video_path,
                            "frames": self.status.get("frame", 0),
                            "loops": _loop_count})
