"""
analyzer.py — DijitalKoku Faz 1
Pencere yakalama (mss), YOLO nesne tespiti ve K-Means renk analizi.
"""

import json
import time
from pathlib import Path
import numpy as np
import cv2
from sklearn.cluster import KMeans
import pygetwindow as gw

_CONFIG_PATH = Path(__file__).parent / "config.json"

# YOLOv8 — yalnızca ihtiyaç anında yükle (geç başlatma)
_yolo_model = None


def _get_yolo():
    """YOLOv8 Nano modelini CUDA float16 ile geç başlatır."""
    global _yolo_model
    if _yolo_model is None:
        import torch
        from ultralytics import YOLO
        dev = "cuda:0" if torch.cuda.is_available() else "cpu"
        _yolo_model = YOLO("yolov8n.pt")
        _yolo_model.to(dev)
        if dev == "cuda:0":
            _yolo_model.model = _yolo_model.model.half()  # float16 → VRAM tasarrufu
            print(f"[YOLO] GPU yüklendi → {torch.cuda.get_device_name(0)} (float16)")
        else:
            print("[YOLO] CPU modunda çalışıyor")
    return _yolo_model


class Analyzer:
    def __init__(self, config_path: "Path | str | None" = None):
        self.prev_frame_gray = None
        try:
            _p = Path(config_path) if config_path else _CONFIG_PATH
            _az = json.loads(_p.read_text(encoding="utf-8")).get("analyzer", {})
        except Exception:
            _az = {}
        self._motion_diff_thr   = int(_az.get("motion_diff_threshold",       25))
        self._sc_sim_hard       = float(_az.get("scene_cut_similarity_hard",   0.25))
        self._sc_sim_mixed      = float(_az.get("scene_cut_similarity_mixed",  0.35))
        self._sc_motion_thr     = float(_az.get("scene_cut_motion_threshold",  0.35))
        self._sc_sim_low        = float(_az.get("scene_cut_similarity_low",    0.18))

    def get_motion_score(self, frame):
        """İki kare arasındaki farkı hesaplar ve değişen piksel oranını 0.0-1.0 arası skor olarak döndürür."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev_frame_gray is None:
            self.prev_frame_gray = gray
            return 0.0
        
        if self.prev_frame_gray.shape != gray.shape:
            self.prev_frame_gray = gray
            return 0.0
        diff = cv2.absdiff(self.prev_frame_gray, gray)
        _, thresh = cv2.threshold(diff, self._motion_diff_thr, 255, cv2.THRESH_BINARY)
        motion_score = np.sum(thresh) / (thresh.size * 255.0)
        self.prev_frame_gray = gray
        return float(motion_score)

    def analyze(
        self,
        frame: np.ndarray,
        prev_frame: np.ndarray = None,
        conf_threshold: float = 0.45,
        kmeans_frame: np.ndarray | None = None,
    ) -> tuple[list[dict], list[tuple], float, bool, float]:
        """
        YOLO + K-Means + Hareket + Sahne Kesme analizini birlikte çalıştırır.
        kmeans_frame: K-Means için küçültülmüş kare (None ise frame kullanılır).
        Döndürür: (detections, dominant_colors, motion_score, is_scene_cut, elapsed_seconds)
        """
        t0 = time.perf_counter()
        
        # 1. YOLO ve Renk
        detections = run_yolo(frame, conf_threshold=conf_threshold)
        colors = run_kmeans(kmeans_frame if kmeans_frame is not None else frame)
        
        # 2. Hareket ve Sahne Kesme Analizi
        motion_score = self.get_motion_score(frame)
        is_scene_cut = False
        
        if prev_frame is not None and prev_frame.shape == frame.shape:
            # Sahne kesme algılama (Histogram karşılaştırma)
            f1 = cv2.resize(cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY), (160, 90))
            f2 = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (160, 90))
            
            hist1 = cv2.calcHist([f1], [0], None, [256], [0, 256])
            hist2 = cv2.calcHist([f2], [0], None, [256], [0, 256])
            similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            
            is_scene_cut = (
                (similarity < self._sc_sim_hard) or
                ((similarity < self._sc_sim_mixed) and (motion_score > self._sc_motion_thr)) or
                ((motion_score > self._sc_motion_thr) and (similarity < self._sc_sim_low))
            )
            
        elapsed = time.perf_counter() - t0
        return detections, colors, motion_score, is_scene_cut, elapsed


# ---------------------------------------------------------------------------
# Pencere yardımcıları
# ---------------------------------------------------------------------------

def list_windows() -> list[str]:
    """Başlığı olan tüm aktif pencereleri döndürür."""
    try:
        titles = gw.getAllTitles()
        return [t for t in titles if t.strip()]
    except Exception:
        return []


def get_window_rect(title: str):
    """
    Verilen başlığa sahip pencerenin (left, top, right, bottom) dikdörtgenini döndürür.
    Bulunamazsa None döner.
    """
    try:
        wins = gw.getWindowsWithTitle(title)
        if not wins:
            return None
        w = wins[0]
        return (w.left, w.top, w.right, w.bottom)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Ekran yakalama — mss kullanır (dxcam'den çok daha stabil)
# ---------------------------------------------------------------------------

def capture_window(title: str, roi: tuple = None) -> np.ndarray | None:
    """
    Seçilen pencerenin ekran görüntüsünü mss ile yakalar.
    roi: (x_offset_pct, y_offset_pct, width_pct, height_pct) - 0.0 to 1.0
    """
    try:
        import mss

        rect = get_window_rect(title)

        with mss.mss() as sct:
            if rect:
                l, t, r, b = rect
                l, t = max(0, l), max(0, t)
                r, b = max(l + 32, r), max(t + 32, b)
                
                win_w = r - l
                win_h = b - t

                # ROI Uygula
                if roi:
                    rx, ry, rw, rh = roi
                    l = int(l + (win_w * rx))
                    t = int(t + (win_h * ry))
                    win_w = int(win_w * rw)
                    win_h = int(win_h * rh)

                monitor = {"left": l, "top": t, "width": win_w, "height": win_h}
            else:
                monitor = sct.monitors[1]

            screenshot = sct.grab(monitor)

        frame = np.array(screenshot, dtype=np.uint8)[:, :, :3]
        return frame

    except Exception as e:
        print(f"[analyzer] capture hatası: {e}")
        return None


# ---------------------------------------------------------------------------
# Görsel ön işleme — Bounding Box önizlemesi
# ---------------------------------------------------------------------------

def draw_preview(frame: np.ndarray, label: str = "İzlenen Alan") -> np.ndarray:
    """
    Çerçeve etrafına parlak yeşil dikdörtgen çizer ve etiket ekler.
    Dashboard'daki görsel doğrulama içindir.
    """
    preview = frame.copy()
    h, w = preview.shape[:2]

    # Dış çerçeve
    cv2.rectangle(preview, (4, 4), (w - 4, h - 4), (0, 230, 80), 3)

    # İlgi Merkezi (Center of Interest) Görselleştirmesi
    # Merkeze yaklaştıkça odak artar
    cx, cy = w // 2, h // 2
    cv2.circle(preview, (cx, cy), 10, (0, 230, 80), -1)
    # Odak halkaları
    for r in [40, 80, 120]:
        alpha = 1.0 - (r / 200)
        color = (0, int(230 * alpha), int(80 * alpha))
        cv2.circle(preview, (cx, cy), r, color, 1)

    # Köşe aksan çizgileri
    corner_len = min(w, h) // 8
    thickness = 4
    color_corner = (0, 255, 120)
    corners = [(4, 4), (w - 4, 4), (4, h - 4), (w - 4, h - 4)]
    for cx_c, cy_c in corners:
        dx = corner_len if cx_c < w // 2 else -corner_len
        dy = corner_len if cy_c < h // 2 else -corner_len
        cv2.line(preview, (cx_c, cy_c), (cx_c + dx, cy_c), color_corner, thickness)
        cv2.line(preview, (cx_c, cy_c), (cx_c, cy_c + dy), color_corner, thickness)

    # Etiket
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"[CANLI] {label}"
    (tw, th), _ = cv2.getTextSize(text, font, 0.55, 2)
    cv2.rectangle(preview, (4, 4), (tw + 14, th + 14), (0, 0, 0), -1)
    cv2.putText(preview, text, (8, th + 8), font, 0.55, (0, 230, 80), 2, cv2.LINE_AA)

    return preview


# ---------------------------------------------------------------------------
# Nesne Tespitlerini Çiz (Görsel Geri Bildirim)
# ---------------------------------------------------------------------------

def draw_color_heatmap(frame: np.ndarray, colors: list[tuple]) -> np.ndarray:
    """
    Ekranın altına baskın renklerin ısı haritasını (paletini) çizer.
    """
    h, w = frame.shape[:2]
    heatmap_h = 40
    canvas = frame.copy()
    
    if not colors:
        return canvas

    num_colors = len(colors)
    segment_w = w // num_colors
    
    for i, bgr in enumerate(colors):
        color = tuple(map(int, bgr))
        start_x = i * segment_w
        end_x = (i + 1) * segment_w if i < num_colors - 1 else w
        
        # Renk bloğunu çiz
        cv2.rectangle(canvas, (start_x, h - heatmap_h), (end_x, h), color, -1)
        # Sınır çizgisi
        if i > 0:
            cv2.line(canvas, (start_x, h - heatmap_h), (start_x, h), (255, 255, 255), 1)

    return canvas
def draw_detections(frame: np.ndarray, detections: list[dict], scent_map: dict = None) -> np.ndarray:
    """
    YOLO tespitlerini, güven skorlarını ve baskın koku eşleşmelerini çizer.
    scent_map: engine.OBJECT_SCENT_MAP (etiket -> koku eşleşmesi için)
    """
    canvas = frame.copy()
    
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        label = det["label"]
        conf = det["confidence"]
        
        # Renk seçimi (etikete göre sabit renk üret)
        color_hash = hash(label)
        color = ( (color_hash & 0xFF), (color_hash >> 8 & 0xFF), (color_hash >> 16 & 0xFF) )
        
        # Dikdörtgen
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        
        # Etiket metni (Nesne + Güven + Koku)
        scent_info = ""
        if scent_map and label in scent_map:
            primary_scent = max(scent_map[label], key=scent_map[label].get)
            scent_info = f" -> {primary_scent}"
            
        txt = f"{label} {conf:.2f}{scent_info}"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        
        # Metin arka planı
        cv2.rectangle(canvas, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
        cv2.putText(canvas, txt, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        
    return canvas


# ---------------------------------------------------------------------------
# YOLOv8 Nano analizi
# ---------------------------------------------------------------------------

def run_yolo(frame: np.ndarray, conf_threshold: float = 0.45) -> list[dict]:
    """
    YOLOv8 Nano ile nesne tespiti yapar.
    Performans için görüntüyü 640px'e sabitler.
    Döndürür: [{'label': str, 'confidence': float, 'box': (x1,y1,x2,y2)}]
    """
    model = _get_yolo()
    
    # 1. Performans Optimizasyonu: Frame'i 640px'e sınırla
    h, w = frame.shape[:2]
    max_dim = 640
    scale = 1.0
    
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        processed_frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
    else:
        processed_frame = frame

    import torch
    _dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    internal_conf = min(conf_threshold, 0.25)
    results = model(processed_frame, verbose=False, conf=internal_conf, device=_dev)[0]
    
    h0, w0 = frame.shape[:2]
    frame_area = max(1, w0 * h0)
    detections = []
    min_conf_by_label = {
        "donut": 0.65,
        "cake": 0.65,
        "pizza": 0.60,
        "hot dog": 0.60,
        "sandwich": 0.55,
        "fork": 0.60,
        "knife": 0.60,
        "spoon": 0.60,
        "bowl": 0.55,
        "cup": 0.55,
        "wine glass": 0.60,
    }
    max_area_ratio_by_label = {
        "donut": 0.55,
        "cake": 0.60,
    }
    allow_below_global = {"cow", "horse", "sheep", "elephant", "zebra", "giraffe", "bear"}
    for box in results.boxes:
        label = results.names[int(box.cls[0])]
        conf = float(box.conf[0])
        label_l = label.lower()
        if label_l in allow_below_global:
            required_conf = max(0.25, float(conf_threshold) - 0.12, float(min_conf_by_label.get(label_l, 0.0)))
        else:
            required_conf = max(float(conf_threshold), float(min_conf_by_label.get(label_l, 0.0)))
        if conf < required_conf:
            continue
        # Orijinal koordinatlara geri dön
        xyxy_orig = (box.xyxy[0] / scale).cpu().numpy().astype(int)
        x1, y1, x2, y2 = map(int, xyxy_orig)
        area_ratio = float(max(0, x2 - x1) * max(0, y2 - y1)) / frame_area
        if label_l in max_area_ratio_by_label:
            if area_ratio > float(max_area_ratio_by_label[label_l]):
                continue
        detections.append({"label": label, "confidence": conf, "box": (x1, y1, x2, y2), "area_ratio": area_ratio})
        
    return detections


# ---------------------------------------------------------------------------
# K-Means renk analizi
# ---------------------------------------------------------------------------

def _kmeans_gpu(pixels: np.ndarray, k: int, n_iter: int = 50) -> np.ndarray:
    """
    PyTorch tabanlı GPU K-Means (Lloyd algoritması).
    CUDA yoksa sklearn'a düşer.
    """
    import torch
    if not torch.cuda.is_available():
        from sklearn.cluster import KMeans as _KM
        km = _KM(n_clusters=k, n_init=3, max_iter=n_iter, random_state=42)
        km.fit(pixels)
        return km.cluster_centers_

    X = torch.from_numpy(pixels).to("cuda", non_blocking=True)  # (N, 3)
    idx = torch.randperm(X.shape[0], device="cuda")[:k]
    centroids = X[idx].clone()                                   # (k, 3)

    for _ in range(n_iter):
        dists = torch.cdist(X.unsqueeze(0), centroids.unsqueeze(0)).squeeze(0)  # (N, k)
        labels = dists.argmin(dim=1)                             # (N,)
        new_c = torch.zeros_like(centroids)
        for i in range(k):
            mask = labels == i
            new_c[i] = X[mask].mean(dim=0) if mask.sum() > 0 else centroids[i]
        if torch.allclose(centroids, new_c, atol=1.0):
            break
        centroids = new_c

    return centroids.cpu().numpy()


def run_kmeans(frame: np.ndarray, k: int = 5) -> list[tuple]:
    """
    Baskın renkleri K-Means ile bulur (GPU önce, CPU fallback).
    Çerçevenin ORTA %60'ını kullanır — kenar UI elementleri vanilya'ı sahte artırmasın.
    Döndürür: [(B, G, R), ...] — baskın olandan azalan sırayla.
    """
    h, w = frame.shape[:2]
    y1, y2 = int(h * 0.20), int(h * 0.80)
    x1, x2 = int(w * 0.20), int(w * 0.80)
    center = frame[y1:y2, x1:x2]

    small  = cv2.resize(center, (160, 90))
    pixels = small.reshape(-1, 3).astype(np.float32)

    try:
        centers = _kmeans_gpu(pixels, k=k, n_iter=50).astype(int)
    except Exception:
        from sklearn.cluster import KMeans as _KM
        km = _KM(n_clusters=k, n_init=3, max_iter=50, random_state=42)
        km.fit(pixels)
        centers = km.cluster_centers_.astype(int)

    labels = np.argmin(
        np.linalg.norm(pixels[:, None] - centers[None], axis=2), axis=1
    )
    counts = np.bincount(labels, minlength=k)
    order  = np.argsort(counts)[::-1]
    return [tuple(centers[i]) for i in order]

