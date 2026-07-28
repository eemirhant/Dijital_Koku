"""
kalman.py — DijitalKoku V3 Katman D: 1D Kalman Filtresi

Her koku kanalı için bağımsız 1D Kalman filtresi.
EMA'nın yerini alır: Q (işlem gürültüsü) ve R (ölçüm gürültüsü) ayrımı
yaparak donanım çıkışını stabilize eder.

Parametre mantığı:
    Yüksek Q  → hızlı değişen sahne kokuları için hızlı tepki (İs/Duman, Deniz)
    Yüksek R  → ölçüme az güven → daha pürüzsüz çıkış (Nane, Çam)
"""

from __future__ import annotations

from engine import SCENTS


class KalmanFilter1D:
    """
    Tek boyutlu Kalman filtresi.
    Durum: koku yoğunluğu [0, 1]

    Q: İşlem gürültüsü — düşük = daha yavaş adaptasyon
    R: Ölçüm gürültüsü — yüksek = ölçüme daha az güven
    """

    def __init__(self, Q: float = 0.010, R: float = 0.10):
        self.Q = Q
        self.R = R
        self.x = 0.0          # Durum tahmini
        self.P = 1.0          # Hata kovaryansı
        self._last_K: float = 0.0  # Son gerçek Kalman kazancı (telemetri)

    def update(self, measurement: float) -> float:
        """
        Kalman güncelleme adımı.
        Döndürür: kararlı koku skoru [0, 1]
        """
        # 1. Tahmin adımı
        x_pred = self.x
        P_pred = self.P + self.Q

        # 2. Kalman kazancı
        K = P_pred / (P_pred + self.R)
        self._last_K = K      # telemetri için sakla

        # 3. Güncelleme adımı
        self.x = x_pred + K * (measurement - x_pred)
        self.P = (1.0 - K) * P_pred

        return max(0.0, min(1.0, self.x))

    def reset(self):
        """Sahne kesimi için sıfırla."""
        self.x = 0.0
        self.P = 1.0
        self._last_K = 0.0

    @property
    def state(self) -> dict:
        """Dashboard telemetri için anlık filtre durumu."""
        return {
            "x": round(self.x, 4),
            "P": round(self.P, 4),
            "Q": self.Q,
            "R": self.R,
            "K": round(self._last_K, 4),  # son gerçek P_pred/(P_pred+R)
        }


class KalmanFilterBank:
    """
    7 koku kanalı için bağımsız KalmanFilter1D koleksiyonu.
    Parametreler config.json'daki 'kalman_filters' tablosundan gelir.
    """

    _DEFAULT_Q = 0.010
    _DEFAULT_R = 0.10

    def __init__(self, params: dict[str, dict[str, float]] | None = None):
        """
        params: {"Çam": {"Q": 0.008, "R": 0.12}, ...}
        None verilirse tüm kanallar varsayılan parametreyle başlatılır.
        """
        params = params or {}
        self.filters: dict[str, KalmanFilter1D] = {}
        for scent in SCENTS:
            p = params.get(scent, {})
            self.filters[scent] = KalmanFilter1D(
                Q=float(p.get("Q", self._DEFAULT_Q)),
                R=float(p.get("R", self._DEFAULT_R)),
            )

    def update(self, scores: dict[str, float]) -> dict[str, float]:
        """Tüm kanalları günceller, kararlı çıktı döndürür."""
        return {s: self.filters[s].update(float(scores.get(s, 0.0))) for s in SCENTS}

    def reset(self):
        """Tüm filtreleri sıfır durumuna döndür (sahne kesimi için)."""
        for kf in self.filters.values():
            kf.reset()

    def get_states(self) -> dict[str, dict]:
        """Dashboard telemetri için anlık filtre durumları."""
        return {s: self.filters[s].state for s in SCENTS}

    def reload_params(self, params: dict[str, dict[str, float]]) -> None:
        """
        Hot-reload: Q/R değerlerini güncelle, filtre durumu (x, P) korunur.
        config.json değiştiğinde çağrılır.
        """
        for scent, p in params.items():
            if scent in self.filters:
                self.filters[scent].Q = float(p.get("Q", self._DEFAULT_Q))
                self.filters[scent].R = float(p.get("R", self._DEFAULT_R))
