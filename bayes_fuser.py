"""
bayes_fuser.py — DijitalKoku V3 Katman C: Bayes Log-Uzayı Füzyon Motoru

Matematiksel model:
    P(s | data) ∝ P(s) × ∏ P(data_i | s)^w_i

Log-uzayında (sayısal kararlılık için):
    log_post(s) = log_prior(s) + Σ w_i × log(P(data_i | s)) + log(mask[s])

Normalizasyon sonrası posterior [0,1] aralığına taşınır.
Bu posterior bir sonraki karenin prior'u olur (zaman serisi sürekliliği).

Katman A modülleri (YOLO, K-Means, Doku, CLIP) likelihood kaynağıdır.
Katman B VLM semantik maskesi likelihood'u ölçekler (veto veya güçlendir).
"""

from __future__ import annotations

import logging
import math

from engine import SCENTS

_log = logging.getLogger(__name__)

_N_SCENTS         = len(SCENTS)
_FLAT_PRIOR_VALUE = 1.0 / _N_SCENTS          # ~0.1429 — düz prior
_LOG_EPSILON      = 1e-7                      # sıfırdan kaçın


class BayesFuser:
    """
    Modül PDF vektörlerini log-uzayında ağırlıklı Bayes çarpımı ile birleştirir.

    Kullanım:
        fuser = BayesFuser(module_weights=cfg["module_weights"])
        posterior = fuser.fuse(
            module_pdfs   = {"yolo": yolo_s, "kmeans": kmeans_s, ...},
            semantic_mask = vlm_mask,   # None → tüm 1.0
        )
    """

    _DEFAULT_WEIGHTS: dict[str, float] = {
        "clip":     0.40,
        "yolo":     0.30,
        "kmeans":   0.15,
        "texture":  0.10,
        "vlm_bias": 0.05,
    }

    def __init__(self, module_weights: dict[str, float] | None = None):
        self.module_weights: dict[str, float] = (
            dict(module_weights) if module_weights else dict(self._DEFAULT_WEIGHTS)
        )
        self._prior: dict[str, float] = {s: _FLAT_PRIOR_VALUE for s in SCENTS}

    # ------------------------------------------------------------------
    # Ana füzyon metodu
    # ------------------------------------------------------------------
    def fuse(
        self,
        module_pdfs:   dict[str, dict[str, float]],
        semantic_mask: dict[str, float] | None = None,
        prior:         dict[str, float] | None = None,
    ) -> dict[str, float]:
        """
        Log-uzayı Bayes füzyon.

        Args:
            module_pdfs  : {"yolo": {...}, "kmeans": {...}, "clip": {...}, ...}
                           Her modülün 7-elemanlı normalize PDF sözlüğü.
            semantic_mask: {"Çam": 0.05, "Deniz": 1.0, ...}
                           VLM çarpanları — None ise tüm 1.0 (veto yok).
            prior        : Önceki karenin posterior'u.
                           None ise dahili prior kullanılır.

        Returns:
            Normalize edilmiş posterior {scent: float}
            Toplam ≈ 1.0 — bir sonraki prior olarak saklanır.
        """
        if prior is None:
            prior = self._prior

        if semantic_mask is None:
            semantic_mask = {s: 1.0 for s in SCENTS}

        log_posteriors: dict[str, float] = {}

        for scent in SCENTS:
            # Prior katkısı — tamamen log-uzayında
            prior_val = max(float(prior.get(scent, _FLAT_PRIOR_VALUE)), _LOG_EPSILON)
            log_post  = math.log(prior_val)

            # Modül likelihood katkıları — ağırlıklı log toplamı
            for module_name, pdf in module_pdfs.items():
                w   = self.module_weights.get(module_name, 0.25)
                val = max(float(pdf.get(scent, _LOG_EPSILON)), _LOG_EPSILON)
                log_post += w * math.log(val)

            # VLM semantik maske katkısı — log-uzayında ölçekleme
            mask_val  = max(float(semantic_mask.get(scent, 1.0)), _LOG_EPSILON)
            log_post += math.log(mask_val)

            log_posteriors[scent] = log_post

        # Log-Sum-Exp normalizasyonu — sayısal taşmayı ve underflow'u önler
        max_lp      = max(log_posteriors.values())
        exp_shifted = {s: math.exp(v - max_lp) for s, v in log_posteriors.items()}
        total       = sum(exp_shifted.values())
        normalized  = {s: v / total for s, v in exp_shifted.items()}

        # Bir sonraki kare için prior güncelle
        self._prior = dict(normalized)

        return normalized

    # ------------------------------------------------------------------
    # Durum yönetimi
    # ------------------------------------------------------------------
    def reset_prior(self) -> None:
        """Sahne kesimi sonrası prior'u düz dağılıma sıfırla."""
        self._prior = {s: _FLAT_PRIOR_VALUE for s in SCENTS}
        _log.debug("[BayesFuser] Prior sıfırlandı — düz dağılım (1/7)")

    def reload_weights(self, module_weights: dict[str, float]) -> None:
        """Hot-reload: modül ağırlıklarını güncelle (prior korunur)."""
        self.module_weights = dict(module_weights)
        _log.debug("[BayesFuser] Modül ağırlıkları güncellendi: %s", module_weights)

    @property
    def current_prior(self) -> dict[str, float]:
        """Anlık prior'un kopyası (debug / telemetri için)."""
        return dict(self._prior)

    def get_debug_info(
        self,
        module_pdfs:   dict[str, dict[str, float]],
        semantic_mask: dict[str, float] | None = None,
    ) -> dict[str, dict]:
        """
        Koku bazında log-likelihood detayını döndürür.
        Dashboard debug sekmesi için — fuse() sonrası çağrılabilir.
        """
        if semantic_mask is None:
            semantic_mask = {s: 1.0 for s in SCENTS}

        debug: dict[str, dict] = {}
        for scent in SCENTS:
            per_module: dict[str, float] = {}
            total_ll = 0.0
            for mod, pdf in module_pdfs.items():
                w   = self.module_weights.get(mod, 0.25)
                val = max(float(pdf.get(scent, _LOG_EPSILON)), _LOG_EPSILON)
                ll  = w * math.log(val)
                per_module[mod] = round(ll, 4)
                total_ll += ll
            mask_val = max(float(semantic_mask.get(scent, 1.0)), _LOG_EPSILON)
            debug[scent] = {
                "per_module":   per_module,
                "mask_log":     round(math.log(mask_val), 4),
                "total_log_ll": round(total_ll + math.log(mask_val), 4),
                "prior":        round(self._prior.get(scent, _FLAT_PRIOR_VALUE), 4),
            }
        return debug
