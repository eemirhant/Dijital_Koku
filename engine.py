"""
engine.py — DijitalKoku
Paylaşılan sabitler: SCENTS listesi ve OBJECT_SCENT_MAP.
Hibrit Karar Motoru (DecisionEngine) pdm_engine.py içindeki
ProbabilisticDecisionMatrix ile değiştirilmiştir.
"""

SCENTS = ["Çam", "Deniz", "Vanilya", "İs/Duman", "Nane", "Toprak", "Limon"]


# ---------------------------------------------------------------------------
# Nesne → Koku Eşlemesi (COCO 80 Sınıfı Kapsanmıştır)
# ---------------------------------------------------------------------------
OBJECT_SCENT_MAP: dict[str, dict[str, float]] = {
    # --- Ulaşım / Şehir ---
    "car":          {"İs/Duman": 0.80, "Limon": 0.20},
    "truck":        {"İs/Duman": 0.85, "Limon": 0.15},
    "bus":          {"İs/Duman": 0.80, "Limon": 0.20},
    "motorcycle":   {"İs/Duman": 0.90, "Limon": 0.10},
    "airplane":     {"İs/Duman": 0.30, "Nane": 0.40, "Deniz": 0.30},
    "train":        {"İs/Duman": 0.60, "Toprak": 0.20, "Vanilya": 0.20},
    "bicycle":      {"Nane": 0.60, "Limon": 0.40},
    "fire hydrant": {"İs/Duman": 0.70, "Limon": 0.30},
    "stop sign":    {"Vanilya": 0.80, "Limon": 0.20},
    "traffic light":{"Vanilya": 0.70, "Nane": 0.30},

    # --- Su / Deniz ---
    "boat":         {"Deniz": 0.90, "Nane": 0.10},
    "surfboard":    {"Deniz": 0.85, "Nane": 0.15},
    "umbrella":     {"Deniz": 0.40, "Limon": 0.30, "Vanilya": 0.30},

    # --- Doğa / Hayvanlar ---
    "bird":         {"Çam": 0.50, "Deniz": 0.30, "Limon": 0.20},
    "cat":          {"Vanilya": 0.50, "Nane": 0.30, "Çam": 0.20},
    "dog":          {"Toprak": 0.60, "Çam": 0.30, "Vanilya": 0.10},
    "horse":        {"Toprak": 0.80, "Çam": 0.20},
    "sheep":        {"Toprak": 0.70, "Çam": 0.30},
    "cow":          {"Toprak": 0.75, "Çam": 0.25},
    "elephant":     {"Toprak": 0.90, "Çam": 0.10},
    "bear":         {"Çam": 0.70, "Toprak": 0.30},
    "zebra":        {"Toprak": 0.80, "Çam": 0.20},
    "giraffe":      {"Toprak": 0.75, "Çam": 0.25},
    "potted plant": {"Çam": 0.75, "Toprak": 0.25},
    "bench":        {"Çam": 0.40, "Toprak": 0.60},

    # --- Gıda / Mutfak ---
    "banana":       {"Limon": 0.70, "Vanilya": 0.30},
    "apple":        {"Limon": 0.60, "Vanilya": 0.40},
    "orange":       {"Limon": 0.95, "Vanilya": 0.05},
    "broccoli":     {"Çam": 0.60, "Nane": 0.40},
    "carrot":       {"Toprak": 0.70, "Limon": 0.30},
    "hot dog":      {"İs/Duman": 0.70, "Vanilya": 0.30},
    "pizza":        {"İs/Duman": 0.60, "Vanilya": 0.40},
    "donut":        {"Vanilya": 1.00},
    "cake":         {"Vanilya": 1.00},
    "sandwich":     {"Vanilya": 0.80, "Toprak": 0.20},
    "bottle":       {"Deniz": 0.40, "Limon": 0.30, "Vanilya": 0.30},
    "wine glass":   {"Limon": 0.60, "Vanilya": 0.40},
    "cup":          {"Limon": 0.50, "Vanilya": 0.50},
    "fork":         {"Vanilya": 1.00},
    "knife":        {"Vanilya": 1.00},
    "spoon":        {"Vanilya": 1.00},
    "bowl":         {"Vanilya": 1.00},

    # --- Ev / İç Mekan ---
    "couch":        {"Vanilya": 0.70, "Toprak": 0.30},
    "bed":          {"Vanilya": 0.80, "Nane": 0.20},
    "chair":        {"Vanilya": 0.75, "Toprak": 0.25},
    "dining table": {"Vanilya": 0.80, "Toprak": 0.20},
    "microwave":    {"Vanilya": 0.80, "İs/Duman": 0.20},
    "oven":         {"İs/Duman": 0.60, "Vanilya": 0.40},
    "toaster":      {"İs/Duman": 0.60, "Vanilya": 0.40},
    "sink":         {"Deniz": 0.60, "Nane": 0.40},
    "book":         {"Vanilya": 0.80, "Toprak": 0.20},
    "vase":         {"Çam": 0.60, "Limon": 0.40},
    "scissors":     {"Vanilya": 1.00},
    "teddy bear":   {"Vanilya": 0.90, "Toprak": 0.10},
    "hair drier":   {"Vanilya": 0.80, "İs/Duman": 0.20},
    "toothbrush":   {"Nane": 1.00},

    # --- Spor / Diğer ---
    "sports ball":  {"Çam": 0.40, "Toprak": 0.60},
    "frisbee":      {"Deniz": 0.50, "Nane": 0.50},
    "skis":         {"Çam": 0.70, "Nane": 0.30},
    "snowboard":    {"Çam": 0.60, "Nane": 0.40},
    "kite":         {"Deniz": 0.70, "Nane": 0.30},
    "person":       {"Vanilya": 0.40, "Nane": 0.40, "Limon": 0.20},
    "toilet":       {"Nane": 0.90, "Limon": 0.10},
    "backpack":     {"Vanilya": 0.85, "Toprak": 0.15},
    "handbag":      {"Vanilya": 0.70, "Limon": 0.30},
    "tie":          {"Vanilya": 0.80, "Nane": 0.20},
    "suitcase":     {"Vanilya": 0.70, "Deniz": 0.30},
    "baseball bat": {"Çam": 0.60, "Toprak": 0.40},
    "baseball glove":{"Toprak": 0.80, "Çam": 0.20},
    "skateboard":   {"Toprak": 0.50, "Nane": 0.50},
    "tennis racket":{"Nane": 0.60, "Vanilya": 0.40},
}
