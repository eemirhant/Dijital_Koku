"""
scent_controller.py — DijitalKoku Asenkron Donanım Kontrolörü

Yazılım analiz döngüsünü bloklamadan Arduino + PCA9685'e komut iletir.
Back-flush ve purge siklüslerini otomatik olarak yönetir.

Kullanım:
    ctrl = ScentController(port="COM3", baudrate=9600)
    ctrl.send(scores, intensity=0.8)
    ctrl.purge()
    ctrl.close()
"""

import time
import queue
import logging
import threading
from dataclasses import dataclass, field
from typing import Optional

from engine import SCENTS

# ---------------------------------------------------------------------------
# Komut Veri Sınıfı
# ---------------------------------------------------------------------------

@dataclass
class ScentCommand:
    scores:    dict[str, float]
    intensity: float
    timestamp: float = field(default_factory=time.monotonic)


# ---------------------------------------------------------------------------
# ScentController
# ---------------------------------------------------------------------------

class ScentController:
    """
    Analiz döngüsüyle asenkron çalışan donanım kontrolörü.

    Mimari:
        - Ana iş parçacığı  → send() ile kuyruğa koyar
        - Worker iş parçacığı → kuyruğu okur, bayat komut atar, seri yazar
        - Purge / Back-flush → kilit üzerinden talep edilir, worker yürütür

    Port verilmezse "simülasyon modu" aktifleşir (log çıktısı üretir).
    """

    DEFAULT_CONFIG: dict = {
        "back_flush_ms":        200,
        "purge_ms":             500,
        "min_change_threshold": 0.05,
        "pwm_max":              4095,
        "command_queue_size":   4,
        "stale_command_ms":     300,
    }

    def __init__(
        self,
        port:     Optional[str] = None,
        baudrate: int           = 9600,
        config:   Optional[dict] = None,
    ):
        self._port     = port
        self._baudrate = baudrate
        self._cfg      = {**self.DEFAULT_CONFIG, **(config or {})}

        self._serial: Optional[object] = None
        self._queue:  queue.Queue = queue.Queue(maxsize=self._cfg["command_queue_size"])

        self._last_scores: dict[str, float] = {s: 0.0 for s in SCENTS}

        self._lock              = threading.Lock()
        self._running           = False
        self._thread: Optional[threading.Thread] = None

        self._purge_requested = False
        self._flush_requested = False

        self._log = logging.getLogger("ScentController")

        if port:
            self._connect()
        self._start_worker()

    # ------------------------------------------------------------------
    # Bağlantı
    # ------------------------------------------------------------------

    def _connect(self):
        try:
            import serial
            self._serial = serial.Serial(self._port, self._baudrate, timeout=0.5)
            time.sleep(2.0)  # Arduino reset bekleme süresi
            self._log.info(f"Seri bağlantı kuruldu: {self._port} @ {self._baudrate}")
        except Exception as exc:
            self._log.warning(f"Seri bağlantı başarısız ({exc}) — simülasyon modu aktif")
            self._serial = None

    # ------------------------------------------------------------------
    # Worker İş Parçacığı
    # ------------------------------------------------------------------

    def _start_worker(self):
        self._running = True
        self._thread  = threading.Thread(
            target=self._worker,
            daemon=True,
            name="ScentWorker",
        )
        self._thread.start()

    def _worker(self):
        """Arka plan döngüsü: kuyruktan komut alır, timing yönetir, seri yazar."""
        while self._running:
            # Öncelikli: donanım bakım istekleri
            with self._lock:
                if self._purge_requested:
                    self._do_purge()
                    self._purge_requested = False
                    continue
                if self._flush_requested:
                    self._do_flush()
                    self._flush_requested = False
                    continue

            # Kuyruktan komut al (50ms timeout — bakım isteklerini kaçırma)
            try:
                cmd: ScentCommand = self._queue.get(timeout=0.05)
            except queue.Empty:
                continue

            # Bayat komut kontrolü: analiz döngüsü geç kaldıysa atla
            age_ms = (time.monotonic() - cmd.timestamp) * 1000.0
            if age_ms > self._cfg["stale_command_ms"]:
                self._log.debug(f"Bayat komut atlandı ({age_ms:.0f} ms)")
                continue

            # Minimum değişim eşiği: gereksiz seri trafiği engelle
            if not self._has_significant_change(cmd.scores):
                continue

            raw = self._build_command(cmd.scores, cmd.intensity)
            self._write(raw)
            self._last_scores = dict(cmd.scores)

    # ------------------------------------------------------------------
    # Değişim Tespiti
    # ------------------------------------------------------------------

    def _has_significant_change(self, scores: dict[str, float]) -> bool:
        thr = self._cfg["min_change_threshold"]
        return any(
            abs(scores.get(s, 0.0) - self._last_scores.get(s, 0.0)) >= thr
            for s in SCENTS
        )

    # ------------------------------------------------------------------
    # Seri Protokol
    # Format: !<ch0>,<ch1>,...,<ch6>,<ch7>|<checksum>#   (8 kanal)
    # ch0-ch6 = 7 koku kanalı, ch7 = tahliye fanı
    # ------------------------------------------------------------------

    def _build_command(
        self,
        scores:    dict[str, float],
        intensity: float,
        ch7_pwm:   int = 0,
    ) -> str:
        pwm_max = int(self._cfg["pwm_max"])
        vals    = [
            str(int(max(0, min(pwm_max, scores.get(s, 0.0) * pwm_max * intensity))))
            for s in SCENTS
        ]
        vals.append(str(int(max(0, min(pwm_max, ch7_pwm)))))
        checksum = sum(int(v) for v in vals) % 256
        return f"!{','.join(vals)}|{checksum}#"

    def _write(self, command: str):
        if self._serial is not None:
            try:
                self._serial.write((command + "\n").encode("ascii"))
                self._serial.flush()
            except Exception as exc:
                self._log.error(f"Seri yazma hatası: {exc}")
        else:
            self._log.debug(f"[SIM] {command}")

    # ------------------------------------------------------------------
    # Back-flush — Pompa Geri Emiş Siklüsü
    # Tüm kanalları sıfırlar, kısa bekleme sonrası normale döner.
    # ------------------------------------------------------------------

    def _do_flush(self):
        ms  = self._cfg["back_flush_ms"]
        self._log.info(f"Back-flush başlıyor ({ms} ms)")
        self._write(self._build_command({s: 0.0 for s in SCENTS}, 1.0, ch7_pwm=0))
        time.sleep(ms / 1000.0)
        self._log.info("Back-flush tamamlandı")

    # ------------------------------------------------------------------
    # Purge — Genel Tahliye Siklüsü
    # Tüm kanalları sıfırlar, ch7 fan ile boşaltır, hafızayı temizler.
    # ------------------------------------------------------------------

    def _do_purge(self):
        ms      = self._cfg["purge_ms"]
        pwm_max = int(self._cfg["pwm_max"])
        self._log.info(f"Purge başlıyor ({ms} ms) — ch7={pwm_max}")
        self._write(self._build_command({s: 0.0 for s in SCENTS}, 1.0, ch7_pwm=pwm_max))
        time.sleep(ms / 1000.0)
        self._write(self._build_command({s: 0.0 for s in SCENTS}, 1.0, ch7_pwm=0))
        self._last_scores = {s: 0.0 for s in SCENTS}
        self._log.info("Purge tamamlandı")

    # ------------------------------------------------------------------
    # Dışa Açık API
    # ------------------------------------------------------------------

    def send(self, scores: dict[str, float], intensity: float = 1.0):
        """
        Analiz döngüsünden çağrılır — bloklamaz.
        Kuyruk doluysa en eski komutu atar ve yenisini ekler.
        """
        cmd = ScentCommand(scores=scores, intensity=intensity)
        try:
            self._queue.put_nowait(cmd)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(cmd)
            except queue.Full:
                pass

    def purge(self):
        """Genel tahliye siklüsü başlat (kapanışta)."""
        with self._lock:
            self._purge_requested = True

    def scene_cut_purge(self) -> None:
        """
        PRD 5.4 Sahne Kesimi Tahliye Protokolü:
            1500 ms boyunca ch0-ch6 = 0, ch7 = pwm_max (4095).
        Ayrı daemon thread'de çalışır — ana akışı bloklamaz.
        """
        def _run() -> None:
            pwm_max = int(self._cfg.get("pwm_max", 4095))
            self._log.info("[PURGE] Sahne kesimi tahliye: ch7=%d, 1500 ms", pwm_max)
            self._write(self._build_command({s: 0.0 for s in SCENTS}, 1.0, ch7_pwm=pwm_max))
            time.sleep(1.5)
            self._write(self._build_command({s: 0.0 for s in SCENTS}, 1.0, ch7_pwm=0))
            self._last_scores = {s: 0.0 for s in SCENTS}
            self._log.info("[PURGE] Sahne kesimi tahliyesi tamamlandı")

        threading.Thread(target=_run, daemon=True, name="PurgeThread").start()

    def flush(self):
        """Back-flush siklüsü başlat."""
        with self._lock:
            self._flush_requested = True

    def get_preview_command(self, scores: dict[str, float], intensity: float = 1.0) -> str:
        """Dashboard görüntüsü için komut string'i döndürür — donanıma göndermez."""
        return self._build_command(scores, intensity)

    def close(self):
        """Kontrolörü düzgünce kapat."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._serial is not None:
            try:
                self._serial.close()
            except Exception:
                pass
        self._log.info("ScentController kapatıldı")

    @property
    def is_connected(self) -> bool:
        """Gerçek seri bağlantı aktif mi?"""
        return self._serial is not None and getattr(self._serial, "is_open", False)

    @property
    def queue_size(self) -> int:
        """Bekleyen komut sayısı."""
        return self._queue.qsize()
