"""
dashboard.py — DijitalKoku Faz 1
Streamlit kontrol paneli: pencere seçici, canlı önizleme, koku telemetri.
Çalıştır: streamlit run dashboard.py
"""

import atexit
import collections
import logging
import os
import subprocess
import sys
import io
import base64
import queue
import time
import tempfile
import datetime
import html
import json
import webbrowser
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
from PIL import Image

import analyzer
from analyzer import Analyzer
from pdm_engine import ProbabilisticDecisionMatrix, SCENTS
from engine import OBJECT_SCENT_MAP
from video_worker import VideoWorker
from window_worker import WindowWorker, list_capturable_windows

# ---------------------------------------------------------------------------
# Logging konfigürasyonu — dk_server.log'a INFO+ seviyesinde yaz
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

# ---------------------------------------------------------------------------
# Sayfa yapılandırması
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="DijitalKoku — Kontrol Paneli",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Session state başlatma (KRİTİK: Sidebar'dan önce olmalı)
# ---------------------------------------------------------------------------
if "running" not in st.session_state:
    st.session_state.running = False
if "engine" not in st.session_state:
    st.session_state.engine = ProbabilisticDecisionMatrix()
if "log" not in st.session_state:
    st.session_state.log = []  # [(zaman, pencere, {koku: pct})]
if "fps" not in st.session_state:
    st.session_state.fps = 0.0
if "last_scores" not in st.session_state:
    st.session_state.last_scores = {s: 0.0 for s in SCENTS}
if "current_window" not in st.session_state:
    st.session_state.current_window = None
if "prev_frame" not in st.session_state:
    st.session_state.prev_frame = None
if "trend_data" not in st.session_state:
    st.session_state.trend_data = [] # List of (timestamp, scores_dict)
if "last_analysis_ts" not in st.session_state:
    st.session_state.last_analysis_ts = 0.0
if "last_detections" not in st.session_state:
    st.session_state.last_detections = []
if "last_colors" not in st.session_state:
    st.session_state.last_colors = []
if "last_motion_score" not in st.session_state:
    st.session_state.last_motion_score = 0.0
if "last_metadata" not in st.session_state:
    st.session_state.last_metadata = {}
if "analyzer" not in st.session_state:
    st.session_state.analyzer = Analyzer()
if "telemetry" not in st.session_state:
    st.session_state.telemetry = collections.deque(maxlen=600)
if "post_cut_until" not in st.session_state:
    st.session_state.post_cut_until = 0.0
if "kalman_history" not in st.session_state:
    st.session_state.kalman_history = collections.deque(maxlen=50)
if "void_mode_active" not in st.session_state:
    st.session_state.void_mode_active = False
if "strict_scent_allowed" not in st.session_state:
    st.session_state.strict_scent_allowed = True
if "_void_purge_sent" not in st.session_state:
    st.session_state["_void_purge_sent"] = False
if "video_cap" not in st.session_state:
    st.session_state.video_cap = None
if "video_tmp_path" not in st.session_state:
    st.session_state.video_tmp_path = None
if "video_name" not in st.session_state:
    st.session_state.video_name = None
if "video_worker" not in st.session_state:
    st.session_state.video_worker = None
if "window_worker" not in st.session_state:
    st.session_state.window_worker = None
if "selected_hwnd" not in st.session_state:
    st.session_state.selected_hwnd = 0
if "selected_wtitle" not in st.session_state:
    st.session_state.selected_wtitle = ""
if "pres_proc" not in st.session_state:
    st.session_state.pres_proc = None

def apply_film_mode_defaults():
    if not bool(st.session_state.get("film_mode", False)):
        return
    st.session_state["analysis_interval"] = 450
    st.session_state["refresh_rate"] = 250
    st.session_state["auto_refresh"] = True
    st.session_state["enable_scene_opt"] = True
    st.session_state["sensitivity"] = 0.55
    st.session_state["cut_boost_ms"] = 900
    st.session_state["cut_alpha"] = 0.85
    st.session_state.force_analyze = True
    st.session_state.last_analysis_ts = 0.0
    st.session_state.post_cut_until = 0.0

# ---------------------------------------------------------------------------
# CSS — koyu tema, premium görünüm
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0a0e1a;
    color: #e2e8f0;
}

/* Ana başlık */
.dk-header {
    background: linear-gradient(135deg, #0d1b2a 0%, #1a2744 50%, #0d2137 100%);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 20px 30px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 14px;
}
.dk-header h1 { margin: 0; font-size: 1.9rem; font-weight: 700;
    background: linear-gradient(90deg, #38bdf8, #818cf8);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.dk-header p { margin: 4px 0 0; color: #64748b; font-size: 0.85rem; }

/* Panel kartları */
.dk-card {
    background: #0f1929;
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 18px 20px;
    margin-bottom: 14px;
}
.dk-card-title {
    font-size: 0.78rem; font-weight: 600; letter-spacing: 0.1em;
    color: #38bdf8; text-transform: uppercase; margin-bottom: 12px;
}

/* Koku bar renkleri */
.scent-bar-wrap { margin-bottom: 10px; }
.scent-label { font-size: 0.82rem; color: #94a3b8; margin-bottom: 3px; }
.scent-pct { font-size: 0.75rem; color: #cbd5e1; float: right; }

/* Durum rozetleri */
.badge-live {
    display: inline-block; background: #052e16;
    border: 1px solid #16a34a; color: #4ade80;
    border-radius: 20px; padding: 2px 10px;
    font-size: 0.72rem; font-weight: 600;
    animation: pulse 2s infinite;
}
.badge-stop {
    display: inline-block; background: #1c1917;
    border: 1px solid #57534e; color: #78716c;
    border-radius: 20px; padding: 2px 10px;
    font-size: 0.72rem;
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.55} }

/* Log tablosu */
.log-row { font-size: 0.76rem; color: #64748b; padding: 4px 0;
           border-bottom: 1px solid #1e293b; }
.log-time { color: #38bdf8; margin-right: 8px; }

/* FPS chip */
.fps-chip {
    display: inline-block; background: #172033;
    border: 1px solid #1e3a5f; border-radius: 8px;
    padding: 3px 10px; font-size: 0.78rem; color: #64748b;
    margin-top: 6px;
}

/* Streamlit düğmesi özelleştirme */
div[data-testid="stButton"] > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
}

/* Sabit yükseklikli paneller: sayfa boyu dalgalanmasın */
.dk-fixed-panel {
    max-height: 320px;
    overflow-y: auto;
    padding-right: 6px;
}
.dk-fixed-panel::-webkit-scrollbar { width: 8px; }
.dk-fixed-panel::-webkit-scrollbar-thumb { background: #1e3a5f; border-radius: 10px; }
.dk-fixed-panel::-webkit-scrollbar-track { background: #0f1929; }
</style>
""", unsafe_allow_html=True)

components.html(
    """
    <script>
    (function () {
      const KEY = "dk_scrollY";
      try {
        const parent = window.parent;
        const y = parseInt(parent.localStorage.getItem(KEY) || "0", 10);
        setTimeout(() => parent.scrollTo(0, y), 0);
        parent.addEventListener("scroll", () => {
          parent.localStorage.setItem(KEY, String(parent.scrollY || 0));
        }, { passive: true });
      } catch (e) {}
    })();
    </script>
    """,
    height=0,
)

# ---------------------------------------------------------------------------
# Sidebar — Ayarlar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="dk-card-title">⚙️ Sistem Ayarları</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="dk-card-title" style="color:#94a3b8; font-size:0.7rem;">Film</div>', unsafe_allow_html=True)
    film_mode = st.toggle("Film Modu", value=False, key="film_mode", on_change=apply_film_mode_defaults)
    cut_boost_ms = st.slider(
        "Sahne Geçiş Tepkisi (ms)",
        min_value=200,
        max_value=2500,
        value=900,
        step=50,
        key="cut_boost_ms",
    )
    cut_alpha = st.slider(
        "Geçiş Hızı (Alpha)",
        min_value=0.50,
        max_value=0.95,
        value=0.85,
        step=0.05,
        key="cut_alpha",
    )

    st.markdown("---")

    sensitivity = st.slider(
        "EMA Hassasiyeti (Alpha)",
        min_value=0.05,
        max_value=0.95,
        value=0.45,
        step=0.05,
        help="PDM motoru için EMA hızı. config.json'dan ema_alpha olarak da ayarlanabilir.",
        key="sensitivity",
    )
    st.session_state["pdm_alpha_override"] = sensitivity

    conf_threshold = st.slider(
        "Nesne Güven Eşiği",
        min_value=0.10,
        max_value=0.90,
        value=0.45,
        step=0.05,
        help="Modelin bir nesneyi tanıması için gereken minimum güven skoru.",
        key="conf_threshold",
    )

    enable_scene_opt = st.toggle("Sahne Optimizasyonu", value=True, key="enable_scene_opt")
    show_debug = st.toggle("Detaylı Analiz", value=False, key="show_debug")
    
    st.markdown("---")
    st.markdown('<div class="dk-card-title">🎭 Sahne Hazır Ayarları</div>', unsafe_allow_html=True)
    
    preset = st.selectbox(
        "Hazır Ayar Seç:",
        ["Özel", "Aksiyon/Oyun", "Belgesel/Doğa", "Mutfak/Yemek", "Sinematik"],
        help="Farklı içerik türleri için optimize edilmiş hassasiyet ve koku dengeleri."
    )
    
    if preset == "Aksiyon/Oyun":
        st.session_state["pdm_alpha_override"] = 0.75
        st.session_state["pdm_multipliers"] = {s: 1.0 for s in SCENTS}
        st.session_state["pdm_multipliers"]["İs/Duman"] = 1.5
        st.session_state["pdm_multipliers"]["Nane"] = 1.3
    elif preset == "Belgesel/Doğa":
        st.session_state["pdm_alpha_override"] = 0.35
        st.session_state["pdm_multipliers"] = {s: 1.0 for s in SCENTS}
        st.session_state["pdm_multipliers"]["Çam"] = 1.6
        st.session_state["pdm_multipliers"]["Toprak"] = 1.4
    elif preset == "Mutfak/Yemek":
        st.session_state["pdm_alpha_override"] = 0.50
        st.session_state["pdm_multipliers"] = {s: 1.0 for s in SCENTS}
        st.session_state["pdm_multipliers"]["Vanilya"] = 1.5
        st.session_state["pdm_multipliers"]["Limon"] = 1.5
    elif preset == "Sinematik":
        st.session_state["pdm_alpha_override"] = 0.25
        st.session_state["pdm_multipliers"] = {s: 1.0 for s in SCENTS}

    st.markdown("---")
    st.markdown('<div class="dk-card-title">🔌 Donanım Simülasyonu</div>', unsafe_allow_html=True)
    show_hardware = st.toggle("Donanım Çıkışını Göster", value=True)
    
    st.info("Donanım bağlı değilse bile üretilen seri protokol komutlarını burada görebilirsiniz.")
    
    st.markdown("---")
    st.markdown('<div class="dk-card-title">📺 Arayüz Ayarları</div>', unsafe_allow_html=True)
    
    refresh_rate = st.slider(
        "Yenileme Gecikmesi (ms)",
        min_value=100,
        max_value=2000,
        value=300,
        step=50,
        help="Arayüzün ne sıklıkla yenileneceğini ayarlar. Yüksek değerler daha stabil bir görünüm sağlar.",
        key="refresh_rate",
    )

    auto_refresh = st.toggle("Otomatik Yenile", value=True, key="auto_refresh")

    analysis_interval = st.slider(
        "Analiz Aralığı (ms)",
        min_value=100,
        max_value=3000,
        value=600,
        step=50,
        help="Analiz hızını düşürmek değerlerin daha okunaklı olmasını sağlar.",
        key="analysis_interval",
    )

    if not auto_refresh:
        if st.button("⏱ Tek Kare Analiz Et", use_container_width=True):
            st.session_state.force_analyze = True
    
    st.markdown("---")
    st.markdown('<div class="dk-card-title">🧪 Koku Yoğunlukları</div>', unsafe_allow_html=True)
    
    multipliers = {}
    for scent in SCENTS:
        multipliers[scent] = st.slider(
            f"{scent} Çarpanı",
            0.1, 3.0, 1.0, 0.1,
            key=f"mul_{scent}"
        )
    if preset == "Özel":
        st.session_state["pdm_multipliers"] = multipliers

    st.markdown("---")
    st.markdown('<div class="dk-card-title">🔍 Bölge Seçici (ROI)</div>', unsafe_allow_html=True)
    
    roi_x = st.slider("Yatay Kaydırma %", 0, 80, 0, 5, key="roi_x_pct") / 100.0
    roi_y = st.slider("Dikey Kaydırma %", 0, 80, 0, 5, key="roi_y_pct") / 100.0
    roi_w = st.slider("Genişlik %", 20, 100, 100, 5, key="roi_w_pct") / 100.0
    roi_h = st.slider("Yükseklik %", 20, 100, 100, 5, key="roi_h_pct") / 100.0
    
    current_roi = (roi_x, roi_y, roi_w, roi_h) if (roi_x > 0 or roi_y > 0 or roi_w < 1.0 or roi_h < 1.0) else None
    st.session_state["current_roi"] = current_roi

# ---------------------------------------------------------------------------
# Tüp Simülasyonu HTML Üretici
# ---------------------------------------------------------------------------
def build_tube_html(scores: dict, attenuation: dict, purge_speed: float) -> str:
    COLORS = {
        "Çam": "#22c55e", "Deniz": "#38bdf8", "Vanilya": "#fbbf24",
        "İs/Duman": "#94a3b8", "Nane": "#34d399", "Toprak": "#b45309", "Limon": "#facc15",
    }
    fan_state  = "running" if purge_speed > 0.05 else "paused"
    fan_dur    = max(0.3, 2.0 - purge_speed * 1.7)
    tubes_html = ""
    for scent in SCENTS:
        val  = float(scores.get(scent, 0.0))
        att  = float(attenuation.get(scent, 1.0))
        col  = COLORS.get(scent, "#94a3b8")
        fill = int(val * 100)
        opacity = max(0.0, min(1.0, 0.25 + att * 0.75))
        spray = "" if val < 0.08 else "".join(
            f'<div style="position:absolute;bottom:{95+i*8}%;left:{20+i*15}%;'
            f'width:3px;height:14px;background:{col};border-radius:3px;opacity:.8;'
            f'animation:dk-spray {max(0.4, 1.5-val):.1f}s {i*0.18:.2f}s infinite;"></div>'
            for i in range(3)
        )
        tubes_html += f"""
        <div style="display:flex;flex-direction:column;align-items:center;gap:6px;">
          <div style="position:relative;width:54px;height:180px;
               background:rgba(255,255,255,0.04);border:1.5px solid rgba(255,255,255,0.15);
               border-radius:27px 27px 8px 8px;overflow:hidden;">
            {spray}
            <div style="position:absolute;bottom:0;width:100%;height:{fill}%;
                 background:linear-gradient(to top,{col}cc,{col}66);
                 border-radius:0 0 6px 6px;opacity:{opacity:.2f};
                 transition:height .6s ease,opacity .6s ease;"></div>
          </div>
          <div style="font-size:0.68rem;color:#94a3b8;text-align:center;width:60px;">{scent}</div>
          <div style="font-size:0.8rem;font-weight:600;color:{col};">{fill}%</div>
        </div>"""
    return f"""
    <style>
    @keyframes dk-spray {{0%{{transform:translateY(0);opacity:.9}}100%{{transform:translateY(-28px);opacity:0}}}}
    @keyframes dk-spin  {{from{{transform:rotate(0deg)}}to{{transform:rotate(360deg)}}}}
    </style>
    <div style="display:flex;flex-direction:column;align-items:center;gap:18px;padding:16px;">
      <div style="display:flex;flex-direction:column;align-items:center;gap:6px;">
        <svg width="72" height="72" viewBox="0 0 100 100"
             style="animation:dk-spin {fan_dur:.1f}s linear infinite;animation-play-state:{fan_state};filter:drop-shadow(0 0 8px #38bdf8aa);">
          <circle cx="50" cy="50" r="7" fill="#e2e8f0"/>
          <ellipse cx="50" cy="22" rx="11" ry="24" fill="#38bdf8cc" transform="rotate(0 50 50)"/>
          <ellipse cx="50" cy="22" rx="11" ry="24" fill="#38bdf8cc" transform="rotate(120 50 50)"/>
          <ellipse cx="50" cy="22" rx="11" ry="24" fill="#38bdf8cc" transform="rotate(240 50 50)"/>
          <circle cx="50" cy="50" r="5" fill="#0f1929"/>
        </svg>
        <div style="font-size:0.7rem;color:{'#38bdf8' if purge_speed>0.05 else '#475569'};">
          {'⚡ TAHLİYE AKTİF' if purge_speed>0.05 else 'Tahliye Fanı'}</div>
      </div>
      <div style="display:flex;gap:14px;align-items:flex-end;flex-wrap:wrap;justify-content:center;">
        {tubes_html}
      </div>
    </div>"""

# ---------------------------------------------------------------------------
# Koku kanalı renk paleti (bar için)
# ---------------------------------------------------------------------------
SCENT_COLORS = {
    "Çam":      "#22c55e",
    "Deniz":    "#38bdf8",
    "Vanilya":  "#fbbf24",
    "İs/Duman": "#94a3b8",
    "Nane":     "#34d399",
    "Toprak":   "#b45309",
    "Limon":    "#facc15",
}

SCENT_ICONS = {
    "Çam":      "🌲",
    "Deniz":    "🌊",
    "Vanilya":  "🍦",
    "İs/Duman": "🔥",
    "Nane":     "🌿",
    "Toprak":   "🌍",
    "Limon":    "🍋",
}

# ---------------------------------------------------------------------------
# Yardımcı: koku bar'ı render et
# ---------------------------------------------------------------------------
def render_scent_bars(scores: dict[str, float], placeholder):
    """Koku kanallarını renkli progress barlarıyla göster."""
    with placeholder.container():
        st.markdown('<div class="dk-card-title">🧪 Sistem Tahmini</div>', unsafe_allow_html=True)

        sorted_scents = sorted(SCENTS, key=lambda s: scores.get(s, 0.0), reverse=True)
        for scent in sorted_scents:
            val = scores.get(scent, 0.0)
            pct = int(val * 100)
            icon = SCENT_ICONS[scent]
            color = SCENT_COLORS[scent]

            # Renkli progress bar (HTML ile)
            bar_html = f"""
            <div class="scent-bar-wrap">
              <div>
                <span class="scent-label">{icon} {scent}</span>
                <span class="scent-pct">{pct}%</span>
              </div>
              <div style="
                background:#1e293b; border-radius:6px; height:10px;
                margin-top:4px; overflow:hidden;">
                <div style="
                  width:{pct}%; height:100%;
                  background: linear-gradient(90deg, {color}99, {color});
                  border-radius:6px;
                  transition: width 0.4s ease;
                "></div>
              </div>
            </div>
            """
            st.markdown(bar_html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Yardımcı: log satırı ekle
# ---------------------------------------------------------------------------
def add_log(window_title: str, scores: dict[str, float]):
    now = datetime.datetime.now().strftime("%H:%M:%S")
    top = max(scores, key=scores.get)
    pct = int(scores[top] * 100)
    st.session_state.log.insert(0, (now, window_title[:28], top, pct))
    if len(st.session_state.log) > 30:
        st.session_state.log.pop()

def add_telemetry(window_title: str, scores: dict[str, float], metadata: dict):
    now = datetime.datetime.now().isoformat(timespec="seconds")
    top = max(scores, key=scores.get)
    row = {
        "time": now,
        "window": window_title[:28],
        "scene": metadata.get("macro_context", ""),
        "scene_conf": float(metadata.get("scene_conf", 0.0) or 0.0),
        "top_scent": top,
        "top_pct": int(scores.get(top, 0.0) * 100),
        "motion": float(metadata.get("motion_score", 0.0) or 0.0),
        "intensity": float(metadata.get("intensity_factor", 1.0) or 1.0),
    }
    for s in SCENTS:
        row[s] = float(scores.get(s, 0.0))
    st.session_state.telemetry.append(row)


# ---------------------------------------------------------------------------
# BAŞLIK + SEKMELER
# ---------------------------------------------------------------------------
st.markdown("""
<div class="dk-header">
  <div>
    <h1>🌿 DijitalKoku</h1>
    <p>V3 Semantik-Olasılıksal Mimari — VLM · Bayes · Kalman · 8-Kanal PWM</p>
  </div>
</div>
""", unsafe_allow_html=True)

tab_main, tab_unified = st.tabs([
    "🎛️ Ana Kontrol",
    "� Birleşik Analiz Paneli",
])


# ---------------------------------------------------------------------------
# YARDIMCI: Canlı Analiz Overlay — YOLO + Optik Akış Vektörleri + ROI
# ---------------------------------------------------------------------------
def draw_analysis_overlay(frame: np.ndarray, detections: list, roi_box: tuple | None) -> np.ndarray:
    ov = frame.copy()
    # Bounding box + etiketler
    for det in detections:
        x1, y1, x2, y2 = det.get("box", (0, 0, 0, 0))
        lbl  = str(det.get("label", ""))
        conf = float(det.get("confidence", 0.0))
        cv2.rectangle(ov, (x1, y1), (x2, y2), (0, 200, 80), 2)
        cv2.putText(ov, f"{lbl} {conf:.2f}", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 230, 100), 1, cv2.LINE_AA)
    # Dinamik ROI çerçevesi (sarı)
    if roi_box:
        x1, y1, x2, y2 = roi_box
        cv2.rectangle(ov, (x1, y1), (x2, y2), (0, 220, 255), 2)
        cv2.putText(ov, "ROI", (x1 + 4, y1 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1, cv2.LINE_AA)
    # Optik akış vektörleri (grid örnekleme, 20×20)
    _FLOW_W, _FLOW_H = 320, 240
    gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_s = cv2.resize(gray, (_FLOW_W, _FLOW_H))
    prev_s = st.session_state.get("ov_prev_gray")
    if prev_s is not None and prev_s.shape == gray_s.shape:
        flow = cv2.calcOpticalFlowFarneback(prev_s, gray_s, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        h, w = gray.shape
        sx, sy = w / _FLOW_W, h / _FLOW_H
        step = max(h // 20, w // 20, 16)
        for y in range(step // 2, h, step):
            for x in range(step // 2, w, step):
                xs = min(int(x / sx), _FLOW_W - 1)
                ys = min(int(y / sy), _FLOW_H - 1)
                fx, fy = flow[ys, xs]
                fx *= sx; fy *= sy
                mag = (fx**2 + fy**2) ** 0.5
                if mag > 1.5:
                    ex, ey = int(x + fx * 3), int(y + fy * 3)
                    cv2.arrowedLine(ov, (x, y), (ex, ey), (255, 120, 40), 1,
                                    tipLength=0.35, line_type=cv2.LINE_AA)
    st.session_state["ov_prev_gray"] = gray_s
    return ov


def _get_roi_box(frame: np.ndarray, detections: list) -> tuple | None:
    focus = {"fire", "car", "truck", "boat", "fire hydrant", "train"}
    h, w  = frame.shape[:2]
    for det in detections:
        if det.get("label", "").lower() in focus:
            return tuple(det["box"])
    return None


# ---------------------------------------------------------------------------
# ANA ANALİZ DÖNGÜSÜ
# ---------------------------------------------------------------------------
def _render_log(placeholder):
    with placeholder.container():
        rows = ['<div class="dk-fixed-panel">']
        for ts, win, top_scent, pct in st.session_state.log[:15]:
            icon = SCENT_ICONS.get(top_scent, "🌸")
            rows.append(
                f'<div class="log-row">'
                f'<span class="log-time">{html.escape(str(ts))}</span>'
                f'{html.escape(str(icon))} <b>{html.escape(str(top_scent))}</b>'
                f' %{int(pct)} — {html.escape(str(win))}</div>'
            )
        rows.append("</div>")
        st.markdown("".join(rows), unsafe_allow_html=True)


# ─── Fragment: Ana Kontrol ────────────────────────────────────────────────────
@st.fragment(run_every=0.5)
def _main_fragment():
    # ── 1. Önceki widget değerlerini session_state'ten oku ──────────────────
    source_mode     = st.session_state.get("source_mode", "🖥️ Ekran Penceresi")
    selected_window = st.session_state.get("selected_wtitle", "")
    _log_label = "(video)" if source_mode == "🎬 Video Dosyası" else selected_window

    # ── 2. Queue'dan frame al + sonuçları işle ──────────────────────────────
    _worker_result: dict | None = None
    frame = None

    if st.session_state.running:
        if source_mode == "🎬 Video Dosyası":
            _vw = st.session_state.get("video_worker")
            if _vw and _vw.is_alive():
                try:
                    _worker_result = _vw.result_queue.get_nowait()
                except queue.Empty:
                    pass
        else:
            _ww = st.session_state.get("window_worker")
            if _ww and _ww.is_alive():
                try:
                    _worker_result = _ww.result_queue.get_nowait()
                except queue.Empty:
                    pass

        if _worker_result is not None:
            frame = _worker_result["frame"]
            _sc   = _worker_result["scores"]
            _md   = _worker_result["metadata"]
            _det  = _worker_result["detections"]
            _cl   = _worker_result["colors"]
            mults = st.session_state.get("pdm_multipliers", {})
            if mults:
                _sc = {s: _sc.get(s, 0.0) * mults.get(s, 1.0) for s in SCENTS}
                _mx = max(_sc.values()) if _sc else 1.0
                if _mx > 1.0:
                    _sc = {s: v / _mx for s, v in _sc.items()}
            st.session_state.last_scores     = dict(_sc)  # referans donmasını önle
            st.session_state.last_metadata   = _md
            st.session_state.last_detections = _det
            st.session_state.last_colors     = _cl
            st.session_state.trend_data.append(dict(_sc))
            if len(st.session_state.trend_data) > 60:
                st.session_state.trend_data.pop(0)
            _kal_st = _md.get("v3_kalman_states", {})
            if _kal_st:
                st.session_state.kalman_history.append(
                    {s: float(_kal_st.get(s, {}).get("x", 0.0)) for s in SCENTS}
                )
            # Semantik Boşluk + Strict Scent: durum ve purge izleme
            _void_now   = bool(_md.get("is_abstract_or_void", False))
            _strict_now = bool(_md.get("v3_vlm_status", {}).get("strict_scent_allowed", True))
            st.session_state.void_mode_active    = _void_now
            st.session_state.strict_scent_allowed = _strict_now
            if _md.get("void_purge_triggered") and not st.session_state.get("_void_purge_sent"):
                _sc_ctrl = st.session_state.get("scent_controller")
                if _sc_ctrl is not None:
                    try:
                        _sc_ctrl.scene_cut_purge()
                    except Exception:
                        pass
                st.session_state["_void_purge_sent"] = True
            elif not _void_now:
                st.session_state["_void_purge_sent"] = False
            add_log(_log_label, _sc)
            add_telemetry(_log_label, _sc, _md)
            st.session_state["_last_ts"] = time.perf_counter()
            roi_box = _get_roi_box(frame, _det)
            ov_frm  = draw_analysis_overlay(frame, _det, roi_box)
            _ov_buf = io.BytesIO()
            Image.fromarray(cv2.cvtColor(ov_frm, cv2.COLOR_BGR2RGB)).save(
                _ov_buf, format="JPEG", quality=75
            )
            st.session_state["ov_frame"] = _ov_buf.getvalue()
            st.session_state["ov_flow"]  = _md.get("module_scores", {}).get("flow", {})

    _active_worker = (
        st.session_state.get("video_worker") if source_mode == "🎬 Video Dosyası"
        else st.session_state.get("window_worker")
    )
    if _active_worker:
        _w_fps = _active_worker.get_status().get("fps", 0.0)
        if _w_fps > 0:
            st.session_state.fps = _w_fps

    # Son bilinen değerleri kullan — dict() ile değer kopyası al (referans donması önle)
    scores      = dict(st.session_state.last_scores)
    metadata    = st.session_state.last_metadata
    attenuation = metadata.get("attenuation", {s: 1.0 for s in SCENTS})
    purge_speed = float(metadata.get("purge_speed", 0.0))
    detections  = st.session_state.last_detections
    colors      = st.session_state.last_colors

    # ── 3. UI Layout ────────────────────────────────────────────────────────
    vid_col, sim_col = st.columns([2, 3], gap="large")

    with vid_col:
        st.markdown('<div class="dk-card-title">📡 Veri Kaynağı</div>', unsafe_allow_html=True)
        source_mode = st.radio(
            "Kaynak", ["🖥️ Ekran Penceresi", "🎬 Video Dosyası"],
            key="source_mode", horizontal=True,
        )

        if source_mode == "🎬 Video Dosyası":
            uploaded = st.file_uploader(
                "MP4 / AVI / MOV yükle", type=["mp4", "avi", "mov", "mkv"],
                key="video_upload",
            )
            if uploaded and uploaded.name != st.session_state.video_name:
                old_w = st.session_state.get("video_worker")
                if old_w and old_w.is_alive():
                    old_w.stop()
                old_tmp = st.session_state.get("video_tmp_path")
                if old_tmp and os.path.exists(old_tmp):
                    try:
                        os.unlink(old_tmp)
                    except OSError:
                        pass
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tmp.write(uploaded.read())
                tmp.close()
                atexit.register(lambda p=tmp.name: os.unlink(p) if os.path.exists(p) else None)
                st.session_state.video_tmp_path = tmp.name
                st.session_state.video_name     = uploaded.name
                st.session_state.video_worker   = VideoWorker(
                    tmp.name,
                    st.session_state.analyzer,
                    st.session_state.engine,
                    conf_threshold=st.session_state.get("conf_threshold", 0.45),
                )
                st.session_state.running = False
            # İşleme durumu göstergesi
            _w = st.session_state.get("video_worker")
            if _w:
                _ws    = _w.get_status()
                _tot   = _ws.get("total", 0)
                _cur   = _ws.get("frame", 0)
                _fps_v = _ws.get("fps", 0.0)
                if _ws.get("error"):
                    st.error(f"⚠️ {_ws['error']}")
                elif _ws.get("running"):
                    _pct = _cur / _tot if _tot > 0 else 0
                    st.progress(_pct)
                    st.caption(
                        f"⚙️ Analiz: Kare {_cur}/{_tot} "
                        f"({_pct*100:.0f}%) | {_fps_v:.0f} FPS"
                    )
                else:
                    st.caption(
                        f"🎞️ {st.session_state.video_name} "
                        f"— Hazır | {_fps_v:.0f} FPS | {_tot} kare"
                    )
            elif st.session_state.video_name:
                st.caption(f"📂 {st.session_state.video_name} — Henüz yüklenmedi")
            selected_window = "(video)"
        else:
            if st.button("🔄 Pencereleri Yenile", use_container_width=True):
                st.rerun(scope="fragment")
            _wl = list_capturable_windows()
            if not _wl:
                _wl = [(0, "(pencere bulunamadı)")]
            _wl_labels = [t[:80] for _, t in _wl]
            _sel_idx = st.selectbox(
                "Hedef Pencere:",
                options=range(len(_wl_labels)),
                format_func=lambda i: _wl_labels[i],
                key="selected_window_idx",
                label_visibility="collapsed",
            )
            selected_hwnd   = _wl[_sel_idx][0]
            selected_window = _wl_labels[_sel_idx]
            st.session_state["selected_hwnd"]   = selected_hwnd
            st.session_state["selected_wtitle"] = selected_window
            # Pencere worker durum göstergesi
            _ww_ui = st.session_state.get("window_worker")
            if _ww_ui:
                _ww_s = _ww_ui.get_status()
                if _ww_s.get("error"):
                    st.error(f"⚠️ {_ww_s['error']}")
                elif _ww_s.get("running"):
                    st.caption(
                        f"⚙️ Arka plan yakalama aktif — "
                        f"{_ww_s.get('frame', 0)} kare | {_ww_s.get('fps', 0):.0f} FPS hedef"
                    )

        btn_c1, btn_c2, btn_c3 = st.columns([2, 2, 3])
        with btn_c1:
            if st.button(
                "▶ Başlat" if not st.session_state.running else "⏸ Durdur",
                use_container_width=True,
                type="primary" if not st.session_state.running else "secondary",
            ):
                st.session_state.running = not st.session_state.running
                _btn_vw = st.session_state.get("video_worker")
                _btn_ww = st.session_state.get("window_worker")
                if st.session_state.running:
                    if source_mode == "🎬 Video Dosyası" and _btn_vw:
                        if not _btn_vw.is_alive():
                            if _btn_ww and _btn_ww.is_alive():
                                _btn_ww.stop()
                            with _btn_vw._lock:
                                _btn_vw.conf_threshold = st.session_state.get("conf_threshold", 0.45)
                            _btn_vw.start()
                    elif source_mode == "🖥️ Ekran Penceresi":
                        _hwnd_btn   = st.session_state.get("selected_hwnd", 0)
                        _wtitle_btn = st.session_state.get("selected_wtitle", "")
                        if _hwnd_btn and not (_btn_ww and _btn_ww.is_alive()):
                            if _btn_vw and _btn_vw.is_alive():
                                _btn_vw.stop()
                            new_ww = WindowWorker(
                                _hwnd_btn, _wtitle_btn,
                                st.session_state.analyzer,
                                st.session_state.engine,
                                conf_threshold=st.session_state.get("conf_threshold", 0.45),
                                capture_fps=5.0,
                            )
                            new_ww.start()
                            st.session_state.window_worker = new_ww
                else:
                    if _btn_vw and _btn_vw.is_alive():
                        _btn_vw.stop()
                    if _btn_ww and _btn_ww.is_alive():
                        _btn_ww.stop()
                    st.session_state.engine.reset()
                st.rerun(scope="fragment")
        with btn_c2:
            if st.button("🗑 Logu Temizle", use_container_width=True):
                st.session_state.log.clear()
                st.session_state.trend_data.clear()
        with btn_c3:
            if st.button("🎬 Sunum Modu (Look-ahead)", use_container_width=True,
                         help="5 saniyelik ileri analiz + koku dalgalanma — port 8502'de açılır"):
                _pres_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "presentation.py")
                _proc = st.session_state.get("pres_proc")
                if _proc is None or _proc.poll() is not None:
                    _proc = subprocess.Popen(
                        [sys.executable, "-m", "streamlit", "run", _pres_path,
                         "--server.port", "8502",
                         "--server.headless", "true",
                         "--browser.gatherUsageStats", "false"],
                        cwd=os.path.dirname(os.path.abspath(__file__)),
                    )
                    st.session_state.pres_proc = _proc
                    time.sleep(1.5)
                webbrowser.open("http://localhost:8502")
                st.toast("🎬 Sunum Modu http://localhost:8502 adresinde açılıyor…", icon="✅")

        if st.session_state.log:
            df_log = pd.DataFrame(
                st.session_state.log,
                columns=["Zaman", "Pencere", "Baskın Koku", "Güven %"],
            )
            st.download_button(
                "📥 Log (CSV)",
                df_log.to_csv(index=False).encode("utf-8"),
                file_name=f"dk_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv", use_container_width=True,
            )

        if st.session_state.running:
            st.markdown(
                f'<span class="badge-live">● CANLI</span>&nbsp;&nbsp;'
                f'<span class="fps-chip">⚡ {st.session_state.fps:.1f} FPS</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<span class="badge-stop">◼ Durduruldu</span>', unsafe_allow_html=True)

        st.markdown(
            '<div class="dk-card-title" style="margin-top:14px;">📸 Canlı Önizleme</div>',
            unsafe_allow_html=True,
        )
        _PH = 340  # sabit önizleme yüksekliği (px)
        if frame is not None:
            _prev = analyzer.draw_preview(frame, label=selected_window[:40])
            _prev = analyzer.draw_detections(_prev, detections, scent_map=OBJECT_SCENT_MAP)
            _prev = analyzer.draw_color_heatmap(_prev, colors)
            _buf = io.BytesIO()
            Image.fromarray(cv2.cvtColor(_prev, cv2.COLOR_BGR2RGB)).save(
                _buf, format="JPEG", quality=85
            )
            _b64 = base64.b64encode(_buf.getvalue()).decode()
            _cap = html.escape(selected_window[:55])
            components.html(
                f"""
                <div style="width:100%;height:{_PH}px;background:#0f172a;
                            border-radius:8px;overflow:hidden;position:relative;">
                  <img src="data:image/jpeg;base64,{_b64}"
                       style="width:100%;height:100%;object-fit:contain;display:block;" />
                  <div style="position:absolute;bottom:6px;left:8px;color:#cbd5e1;
                              font-size:0.70rem;background:rgba(0,0,0,0.65);
                              padding:2px 7px;border-radius:4px;max-width:92%;
                              overflow:hidden;white-space:nowrap;text-overflow:ellipsis;">
                    📌 {_cap}
                  </div>
                </div>""",
                height=_PH + 4, scrolling=False,
            )
        else:
            _msg = "⏳ Kare bekleniyor…" if st.session_state.running \
                else "ℹ️ Pencere seç veya video yükle, ardından ▶ Başlat'a tıkla."
            components.html(
                f"""
                <div style="width:100%;height:{_PH}px;background:#0f172a;
                            border-radius:8px;display:flex;
                            align-items:center;justify-content:center;">
                  <span style="color:#475569;font-size:0.85rem;">{_msg}</span>
                </div>""",
                height=_PH + 4, scrolling=False,
            )

    with sim_col:
        st.markdown('<div class="dk-card-title">🧪 Sanal Koku Ünitesi</div>', unsafe_allow_html=True)
        st.html(build_tube_html(scores, attenuation, purge_speed))

    st.markdown("---")
    wave_col, hw_col = st.columns([3, 2], gap="large")
    with wave_col:
        st.markdown(
            '<div class="dk-card-title">📈 Koku Yoğunluğu Dalgası (Son 60 Kare)</div>',
            unsafe_allow_html=True,
        )
        _tdata = st.session_state.trend_data or [{s: 0.0 for s in SCENTS}]
        df_t = pd.DataFrame(_tdata)
        fig  = go.Figure()
        for c in SCENTS:
            if c not in df_t.columns:
                df_t[c] = 0.0
            hx = SCENT_COLORS.get(c, "#94a3b8").lstrip("#")
            r2, g2, b2 = int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16)
            fig.add_trace(go.Scatter(
                y=df_t[c], name=c, mode="lines",
                line=dict(color=f"#{hx}", width=1.8),
                fill="tozeroy", fillcolor=f"rgba({r2},{g2},{b2},0.10)",
            ))
        fig.update_layout(
            height=200, margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=10)),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=True, gridcolor="#1e293b", range=[0, 1]),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    with hw_col:
        st.markdown('<div class="dk-card-title">🔌 8-Kanal Seri Çıkış</div>', unsafe_allow_html=True)
        cmd = st.session_state.engine.get_serial_command_8ch(
            scores, attenuation, purge_speed=purge_speed, intensity=1.0
        )
        att_rows = ""
        for s in SCENTS:
            att = float(attenuation.get(s, 1.0))
            col = SCENT_COLORS.get(s, "#94a3b8")
            icon = SCENT_ICONS.get(s, "")
            att_rows += (
                f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:6px;">'
                f'<span style="color:{col};font-size:0.70rem;min-width:80px;">{icon} {s}</span>'
                f'<div style="flex:1;background:#1e293b;border-radius:4px;height:5px;">'
                f'<div style="width:{int(att*100)}%;height:100%;background:{col};border-radius:4px;"></div>'
                f'</div>'
                f'<span style="color:#64748b;font-size:0.68rem;min-width:30px;text-align:right;">{att:.2f}</span>'
                f'</div>'
            )
        st.html(
            f"""
            <div style="background:#0f172a;padding:10px 12px;border-radius:8px;
                        border:1px solid #1e293b;">
              <div style="color:#38bdf8;font-size:0.72rem;letter-spacing:.05em;
                          margin-bottom:8px;text-transform:uppercase;">Seri Komut</div>
              <div style="background:#1e293b;border-radius:6px;padding:8px 12px;
                          color:#22c55e;font-size:0.82rem;word-break:break-all;
                          border:1px solid #334155;font-family:monospace;">{html.escape(cmd)}</div>
              <div style="color:#38bdf8;font-size:0.72rem;letter-spacing:.05em;
                          margin:12px 0 6px;text-transform:uppercase;">Attenuation</div>
              {att_rows}
            </div>"""
        )

    ins_col, log_col = st.columns([1, 1], gap="large")
    with ins_col:
        st.markdown('<div class="dk-card-title">🎯 Model İçgörüleri</div>', unsafe_allow_html=True)
        _parts = ['<div class="dk-fixed-panel">']
        m_motion  = float(metadata.get("motion_score", 0.0) or 0.0)
        m_alpha   = float(metadata.get("dynamic_alpha", 0.45) or 0.45)
        m_ms      = float(metadata.get("pdm_elapsed_ms", 0.0) or 0.0)
        m_ctx     = str(metadata.get("macro_context",    "UNKNOWN"))
        m_light   = str(metadata.get("lighting_profile", "DAYLIGHT"))
        m_dom     = str(metadata.get("dominant_scent",   "-"))
        ctx_icons = {"INDOOR": "🏠", "NATURE": "🌿", "WATER": "🌊",
                     "KITCHEN": "🍳", "CITY": "🏙️", "UNKNOWN": "❓"}
        lit_icons = {"DAYLIGHT": "☀️", "GOLDEN_HOUR": "🌅", "NIGHT_DARK": "🌙"}
        ctx_icon  = ctx_icons.get(m_ctx,   "❓")
        lit_icon  = lit_icons.get(m_light, "☀️")
        _parts.append(
            f'<div class="log-row">'
            f'Hareket: {m_motion:.3f} | Alpha: {m_alpha:.3f} | {m_ms:.1f} ms</div>'
        )
        _parts.append(
            f'<div class="log-row" style="color:#38bdf8;">'
            f'{ctx_icon} Sahne: <b>{html.escape(m_ctx)}</b> '
            f'{lit_icon} <b>{html.escape(m_light)}</b> '
            f'— Baskın: <b>{html.escape(m_dom)}</b></div>'
        )
        if purge_speed > 0.05:
            _parts.append(
                f'<div class="log-row" style="color:#38bdf8">⚡ Tahliye: {purge_speed:.2f}</div>'
            )
        mod = metadata.get("module_scores", {})
        if mod:
            for mn, ms_d in mod.items():
                top_s = max(ms_d, key=ms_d.get) if ms_d else "-"
                top_v = int(ms_d.get(top_s, 0) * 100) if ms_d else 0
                _parts.append(
                    f'<div class="log-row">{html.escape(mn)}: '
                    f'<b>{html.escape(top_s)}</b> %{top_v}</div>'
                )
        if detections:
            for det in detections[:5]:
                lbl  = html.escape(str(det.get("label", "")))
                conf = int(float(det.get("confidence", 0.0)) * 100)
                _parts.append(f'<div class="log-row">🔍 <b>{lbl}</b> — %{conf}</div>')
        else:
            _parts.append(
                '<div class="log-row" style="color:#475569">Nesne tespit edilmedi</div>'
            )
        _parts.append("</div>")
        st.markdown("".join(_parts), unsafe_allow_html=True)
    with log_col:
        st.markdown('<div class="dk-card-title">📋 Analiz Logu</div>', unsafe_allow_html=True)
        _log_slot = st.empty()

    _render_log(_log_slot)


# ─── Fragment: Birleşik Analiz Paneli ────────────────────────────────────────
@st.fragment(run_every=0.5)
def _unified_fragment():
    metadata    = st.session_state.last_metadata
    scores      = dict(st.session_state.last_scores)  # değer kopyası
    attenuation = metadata.get("attenuation", {s: 1.0 for s in SCENTS}) if metadata else {}

    # Veri yoksa boş fallback — bileşenler hiç unmount edilmez, sadece 0 gösterir.
    if not metadata:
        metadata = {}

    st.markdown(
        '<div class="dk-card-title">'
        '🔬 Birleşik Analiz Paneli — Model Çıktıları · Semantik Katmanlar · Donanım'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── 6-Stage Füzyon Pipeline (her zaman görünür) ──────────────────────────
    _N_S     = len(SCENTS)
    _FLAT_P  = 1.0 / _N_S
    _ms_p    = metadata.get("module_scores", {})
    _post_p  = metadata.get("v3_bayes_posterior") or {s: _FLAT_P for s in SCENTS}
    _kal_p   = metadata.get("v3_kalman_states", {})
    _sm_p    = metadata.get("v3_semantic_mask")  or {s: 1.0 for s in SCENTS}
    _pipe_rows = [
        ("A", "K-Means",  "🎨", _ms_p.get("kmeans") or {s: 0.0 for s in SCENTS}),
        ("B", "CLIP",     "🧠", _ms_p.get("ovl")    or {s: 0.0 for s in SCENTS}),
        ("C", "VLM Mask", "🔮", _sm_p),
        ("D", "Bayes",    "⚛️", _post_p),
        ("E", "Kalman",   "📡", {s: float((_kal_p.get(s) or {}).get("x", 0.0)) for s in SCENTS}),
        ("F", "PWM",      "🔌", {s: float(scores.get(s, 0.0)) for s in SCENTS}),
    ]
    _pcols = st.columns(6, gap="small")
    for _pi, (_pid, _pnm, _pic, _pd) in enumerate(_pipe_rows):
        with _pcols[_pi]:
            _tops = max(_pd, key=_pd.get) if any(v > 0 for v in _pd.values()) else SCENTS[0]
            _topv = float(_pd.get(_tops, 0.0))
            _cs   = SCENT_COLORS.get(_tops, "#94a3b8")
            _ic2  = SCENT_ICONS.get(_tops, "")
            _pp   = int(_topv * 100)
            st.html(
                f'<div style="background:#0f172a;border:1px solid {_cs}44;border-radius:8px;'
                f'padding:7px 5px;text-align:center;min-height:84px;">'
                f'<div style="color:#475569;font-size:0.57rem;font-weight:700;text-transform:uppercase;'
                f'letter-spacing:.04em;margin-bottom:3px;">{_pid} {_pic} {_pnm}</div>'
                f'<div style="color:{_cs};font-size:0.72rem;font-weight:700;'
                f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">'
                f'{_ic2} {html.escape(_tops)}</div>'
                f'<div style="color:#94a3b8;font-size:0.68rem;margin-top:2px;">%{_pp}</div>'
                f'<div style="background:#1e293b;border-radius:3px;height:4px;margin-top:5px;">'
                f'<div style="width:{_pp}%;height:100%;background:{_cs};border-radius:3px;"></div>'
                f'</div></div>'
            )
    st.markdown("---")

    col1, col2, col3 = st.columns([1, 1, 1], gap="medium")

    # ── Sol Sütun: Girdi, YOLO, CLIP, VLM Bağlam ─────────────────────────────
    with col1:
        st.markdown("##### 📷 Girdi & Modeller")

        ov = st.session_state.get("ov_frame")
        if ov:
            st.image(ov, caption="Son Kare (YOLO overlay)", use_container_width=True)

        st.markdown("**🔍 YOLO — En Güçlü 3 Nesne**")
        yolo_top3 = metadata.get("yolo_top3", [])
        if yolo_top3:
            for det in yolo_top3:
                lbl  = str(det.get("label", "?"))
                conf = int(det.get("confidence", 0.0) * 100)
                area = round(float(det.get("area_ratio", 0.0)) * 100, 1)
                st.markdown(
                    f'<div style="margin-bottom:7px;">'
                    f'<div style="display:flex;justify-content:space-between;">'
                    f'<b style="color:#e2e8f0;font-size:0.82rem;">{html.escape(lbl)}</b>'
                    f'<span style="color:#94a3b8;font-size:0.75rem;">%{conf} · alan %{area}</span>'
                    f'</div>'
                    f'<div style="background:#1e293b;border-radius:4px;height:6px;margin-top:3px;">'
                    f'<div style="width:{conf}%;height:100%;background:#34d399;border-radius:4px;"></div>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<span style="color:#475569;font-size:0.8rem;">Nesne tespit edilmedi</span>',
                unsafe_allow_html=True,
            )
        penalty_log = metadata.get("yolo_penalty_log", {})
        if penalty_log:
            st.markdown("**⚠️ Uygulanan Cezalar** (conf<0.55 & alan<%5 → ×0.20)")
            pen_html = "".join(
                f'<span style="display:inline-block;background:#7f1d1d33;'
                f'border:1px solid #ef444488;border-radius:4px;'
                f'padding:2px 7px;margin:2px;font-size:0.68rem;color:#fca5a5;">'
                f'{html.escape(lbl)}: ×{mult}</span>'
                for lbl, mult in penalty_log.items()
            )
            st.html(f'<div style="margin-bottom:6px;">{pen_html}</div>')

        st.markdown("---")
        st.markdown("**🧠 CLIP — En Güçlü 3 Sahne**")
        clip_top3 = metadata.get("clip_top3", [])
        has_clip  = bool(clip_top3) and any(v > 0 for _, v in clip_top3)
        if has_clip:
            for scent, prob in clip_top3:
                pct   = int(prob * 100)
                col_c = SCENT_COLORS.get(scent, "#94a3b8")
                icon  = SCENT_ICONS.get(scent, "")
                st.markdown(
                    f'<div style="margin-bottom:7px;">'
                    f'<div style="display:flex;justify-content:space-between;">'
                    f'<span style="color:{col_c};font-size:0.82rem;">{icon} <b>{html.escape(scent)}</b></span>'
                    f'<span style="color:#94a3b8;font-size:0.75rem;">%{pct}</span>'
                    f'</div>'
                    f'<div style="background:#1e293b;border-radius:4px;height:6px;margin-top:3px;">'
                    f'<div style="width:{pct}%;height:100%;background:{col_c};border-radius:4px;"></div>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<span style="color:#475569;font-size:0.8rem;">'
                'CLIP henüz çalışmadı (ilk 15 karede başlar)</span>',
                unsafe_allow_html=True,
            )

        # ── V3: VLM JSON Bağlam Verisi ───────────────────────────────────────
        st.markdown("---")
        st.markdown("**🔮 Katman B — VLM Sahne Bağlamı**")
        vlm_info_l = metadata.get("v3_vlm_status", {})
        last_ctx   = vlm_info_l.get("last_context", {})
        if last_ctx:
            env_v    = last_ctx.get("environment", "unknown")
            has_veg  = last_ctx.get("organic_vegetation", False)
            has_wat  = last_ctx.get("water_body", False)
            has_fir  = last_ctx.get("fire_smoke", False)
            conf_vlm = float(last_ctx.get("confidence", 0.0))
            st.html(
                f'<div style="background:#0f172a;border:1px solid #1e3a5f;'
                f'border-radius:8px;padding:10px 14px;">'
                f'<div style="color:#e2e8f0;font-size:0.82rem;font-weight:600;margin-bottom:6px;">'
                f'🌍 {html.escape(str(env_v))}</div>'
                f'<div style="display:flex;gap:10px;flex-wrap:wrap;">'
                f'<span style="font-size:0.72rem;color:{"#4ade80" if has_veg else "#64748b"};">'
                f'🌿 Bitki: {"✓" if has_veg else "✗"}</span>'
                f'<span style="font-size:0.72rem;color:{"#38bdf8" if has_wat else "#64748b"};">'
                f'💧 Su: {"✓" if has_wat else "✗"}</span>'
                f'<span style="font-size:0.72rem;color:{"#f97316" if has_fir else "#64748b"};">'
                f'🔥 Duman: {"✓" if has_fir else "✗"}</span>'
                f'<span style="font-size:0.72rem;color:#94a3b8;">'
                f'Güven: {conf_vlm:.2f}</span>'
                f'</div></div>'
            )
            with st.expander("Ham JSON Bağlam", expanded=False):
                st.json(last_ctx)
        else:
            st.html(
                '<div style="background:#0f172a;border:1px solid #1e3a5f;border-radius:8px;'
                'padding:10px 14px;color:#475569;font-size:0.8rem;">VLM analizi bekleniyor…</div>'
            )

    # ── Orta Sütun: Renk, PDM, Semantik Maske ────────────────────────────────
    with col2:
        st.markdown("##### 🎨 Renk & PDM & Semantik Maske")

        st.markdown("**🎨 K-Means — Baskın Renkler (HSV)**")
        roi_colors = metadata.get("roi_colors_bgr", [])
        if roi_colors:
            blocks = ""
            for bgr in roi_colors[:5]:
                b, g, r  = int(bgr[0]), int(bgr[1]), int(bgr[2])
                bgr_arr  = np.uint8([[[b, g, r]]])
                hsv_arr  = cv2.cvtColor(bgr_arr, cv2.COLOR_BGR2HSV)[0][0]
                h_deg    = int(hsv_arr[0]) * 2
                s_pct    = int(hsv_arr[1] / 255 * 100)
                v_pct    = int(hsv_arr[2] / 255 * 100)
                blocks  += (
                    f'<div style="display:inline-flex;flex-direction:column;align-items:center;'
                    f'margin:3px;gap:3px;vertical-align:top;">'
                    f'<div style="width:44px;height:44px;background:rgb({r},{g},{b});'
                    f'border-radius:6px;border:1px solid rgba(255,255,255,0.15);"></div>'
                    f'<span style="color:#64748b;font-size:0.62rem;text-align:center;line-height:1.4;">'
                    f'H:{h_deg}°<br>S:{s_pct}%<br>V:{v_pct}%</span>'
                    f'</div>'
                )
            st.html(f'<div style="display:flex;flex-wrap:wrap;gap:4px;margin-bottom:10px;">{blocks}</div>')
        else:
            st.markdown(
                '<span style="color:#475569;font-size:0.8rem;">Renk verisi yok</span>',
                unsafe_allow_html=True,
            )
        neutral_ct = metadata.get("kmeans_neutral_filtered", 0)
        if neutral_ct > 0:
            st.markdown(
                f'<span style="color:#64748b;font-size:0.75rem;">⛔ {neutral_ct} nötr renk filtrelendi (S/V<%15)</span>',
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown(
            "**⚖️ PDM 4-Yollu Matris Snapshotu**  "
            '<span style="color:#64748b;font-size:0.7rem;">Y=YOLO K=KMeans T=Doku C=CLIP</span>',
            unsafe_allow_html=True,
        )
        blended_4way = metadata.get("blended_4way") or {s: 0.0 for s in SCENTS}
        mod_scores   = metadata.get("module_scores", {})
        rows_html = ""
        sorted_4w = sorted(SCENTS, key=lambda s: blended_4way.get(s, 0), reverse=True)
        for s in sorted_4w:
            b4  = int(blended_4way.get(s, 0.0) * 100)
            yk  = int(mod_scores.get("yolo",    {}).get(s, 0.0) * 100)
            km  = int(mod_scores.get("kmeans",  {}).get(s, 0.0) * 100)
            tx  = int(mod_scores.get("texture", {}).get(s, 0.0) * 100)
            cl  = int(mod_scores.get("ovl",     {}).get(s, 0.0) * 100)
            col_s = SCENT_COLORS.get(s, "#94a3b8")
            icon  = SCENT_ICONS.get(s, "")
            rows_html += (
                f'<div style="margin-bottom:8px;">'
                f'<div style="display:flex;justify-content:space-between;margin-bottom:2px;">'
                f'<span style="color:{col_s};font-size:0.78rem;">{icon} {html.escape(s)}</span>'
                f'<b style="color:#e2e8f0;font-size:0.78rem;">%{b4}</b>'
                f'</div>'
                f'<div style="display:flex;gap:6px;font-size:0.65rem;color:#64748b;margin-bottom:3px;">'
                f'<span>Y:%{yk}</span><span>K:%{km}</span>'
                f'<span>T:%{tx}</span><span>C:%{cl}</span>'
                f'</div>'
                f'<div style="background:#1e293b;border-radius:3px;height:5px;">'
                f'<div style="width:{b4}%;height:100%;background:{col_s};border-radius:3px;"></div>'
                f'</div></div>'
            )
        st.html(f'<div>{rows_html}</div>')

        # ── V3: Semantik Maske Çarpanları (Veto Görselleştirmesi) ────────────
        st.markdown("---")
        st.markdown("**🎭 Katman B — Semantik Maske Çarpanları**")
        # Her zaman görünür — veri yoksa 1.0 fallback (normal çarpan)
        mask = metadata.get("v3_semantic_mask") or {s: 1.0 for s in SCENTS}
        mask_rows = ""
        for s in SCENTS:
            v        = float(mask.get(s, 1.0))
            col_s    = SCENT_COLORS.get(s, "#94a3b8")
            icon     = SCENT_ICONS.get(s, "")
            veto_col = "#ef4444" if v < 0.20 else ("#f59e0b" if v < 0.80 else "#22c55e")
            veto_lbl = "VETO" if v < 0.20 else ("Zayıf" if v < 0.80 else "Normal")
            pct_bar  = min(int(v * 50), 100)
            mask_rows += (
                f'<div style="display:flex;align-items:center;gap:5px;margin-bottom:5px;">'
                f'<span style="color:{col_s};font-size:0.73rem;min-width:72px;">{icon} {html.escape(s)}</span>'
                f'<div style="flex:1;background:#1e293b;border-radius:3px;height:7px;">'
                f'<div style="width:{pct_bar}%;height:100%;background:{veto_col};border-radius:3px;"></div>'
                f'</div>'
                f'<span style="color:{veto_col};font-size:0.68rem;min-width:42px;text-align:right;">'
                f'\u00d7{v:.3f}</span>'
                f'<span style="background:{veto_col}22;color:{veto_col};'
                f'border:1px solid {veto_col}55;border-radius:3px;'
                f'padding:1px 5px;font-size:0.60rem;min-width:44px;text-align:center;">'
                f'{veto_lbl}</span>'
                f'</div>'
            )
        st.html(f'<div>{mask_rows}</div>')

        # ── K-Means Tamer (her zaman görünür) ────────────────────────────────
        _TAMER_COLORS = {
            "Aksiyon":           "#ef4444",
            "Kurtar\u0131c\u0131":          "#22c55e",
            "Kurtar\u0131c\u0131+Emniyet":  "#f59e0b",
            "Kentsel":           "#3b82f6",
            "Normal":            "#64748b",
        }
        _base_w       = 0.15
        tamer_mode    = metadata.get("kmeans_tamer_mode", "Normal")
        tamer_weights = metadata.get("kmeans_dyn_weights") or {s: _base_w for s in SCENTS}
        _mc           = _TAMER_COLORS.get(tamer_mode, "#64748b")
        tamer_rows    = ""
        for s in SCENTS:
            w     = float(tamer_weights.get(s, _base_w))
            col_s = SCENT_COLORS.get(s, "#94a3b8")
            icon  = SCENT_ICONS.get(s, "")
            ratio = w / _base_w if _base_w > 0 else 1.0
            bar   = min(int(ratio * 50), 100)
            if w < 0.02:
                wc, wl = "#ef4444", "VETO"
            elif w > _base_w + 0.01:
                wc, wl = "#22c55e", "G\u00dcCL\u00dc"
            else:
                wc, wl = "#64748b", "Std"
            tamer_rows += (
                f'<div style="display:flex;align-items:center;gap:5px;margin-bottom:4px;">'
                f'<span style="color:{col_s};font-size:0.71rem;min-width:72px;">{icon} {html.escape(s)}</span>'
                f'<div style="flex:1;background:#1e293b;border-radius:3px;height:6px;">'
                f'<div style="width:{bar}%;height:100%;background:{wc};border-radius:3px;"></div>'
                f'</div>'
                f'<span style="color:{wc};font-size:0.67rem;min-width:38px;text-align:right;">'
                f'w={w:.3f}</span>'
                f'<span style="background:{wc}22;color:{wc};border:1px solid {wc}55;'
                f'border-radius:3px;padding:1px 4px;font-size:0.58rem;min-width:38px;text-align:center;">'
                f'{wl}</span>'
                f'</div>'
            )
        st.html(
            f'<div style="background:#0f172a;border:1px solid {_mc}44;border-radius:7px;'
            f'padding:8px 10px;margin-top:8px;">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:7px;">'
            f'<span style="color:#94a3b8;font-size:0.72rem;">\U0001f3af K-Means Tamer</span>'
            f'<span style="background:{_mc}33;color:{_mc};border:1px solid {_mc}66;'
            f'border-radius:4px;padding:2px 8px;font-size:0.68rem;font-weight:700;">'
            f'{html.escape(tamer_mode)}</span>'
            f'</div>'
            f'{tamer_rows}</div>'
        )

        # ── Bayes Posterior (her zaman görünür; expander kaldırıldı) ──────────
        st.markdown("---")
        st.markdown("**\u26db\ufe0f Katman C \u2014 Bayes Posterior**")
        _FLAT_B   = 1.0 / len(SCENTS)
        posterior = metadata.get("v3_bayes_posterior") or {s: _FLAT_B for s in SCENTS}
        mod_pdfs  = metadata.get("v3_module_pdfs", {})
        bay_rows  = ""
        for s in sorted(SCENTS, key=lambda x: posterior.get(x, 0.0), reverse=True):
            pv    = float(posterior.get(s, 0.0))
            col_s = SCENT_COLORS.get(s, "#94a3b8")
            icon  = SCENT_ICONS.get(s, "")
            pct   = int(pv * 100)
            y_v   = round(float(mod_pdfs.get("yolo",    {}).get(s, 0.0)) * 100)
            c_v   = round(float(mod_pdfs.get("clip",    {}).get(s, 0.0)) * 100)
            k_v   = round(float(mod_pdfs.get("kmeans",  {}).get(s, 0.0)) * 100)
            t_v   = round(float(mod_pdfs.get("texture", {}).get(s, 0.0)) * 100)
            bay_rows += (
                f'<div style="margin-bottom:6px;">'
                f'<div style="display:flex;justify-content:space-between;">'
                f'<span style="color:{col_s};font-size:0.78rem;">{icon} {html.escape(s)}</span>'
                f'<span style="color:#94a3b8;font-size:0.65rem;">'
                f'Y%{y_v} C%{c_v} K%{k_v} T%{t_v}</span>'
                f'</div>'
                f'<div style="background:#1e293b;border-radius:3px;height:5px;margin-top:3px;">'
                f'<div style="width:{pct}%;height:100%;background:{col_s};border-radius:3px;"></div>'
                f'</div></div>'
            )
        st.html(f'<div>{bay_rows}</div>')

    # ── Sağ Sütun: VLM LED, VCV, Kalman Grafik, Histerezis, PWM ─────────────
    with col3:
        st.markdown("##### 🔬 Füzyon & Stabilizasyon & Donanım")

        # ── V3: VLM Durum LED + Kokusuz Mod Rozeti ────────────────────────────
        vlm_info  = metadata.get("v3_vlm_status", {})
        status    = str(vlm_info.get("vlm_status", "disabled"))
        STATUS_COLORS = {"ok": "#22c55e", "init": "#f59e0b", "stale": "#ef4444",
                         "error": "#ef4444", "disabled": "#64748b"}
        st_col    = STATUS_COLORS.get(status, "#94a3b8")
        age       = vlm_info.get("last_valid_age_s", -1)
        ok_count  = vlm_info.get("successful_calls", 0)
        total_c   = vlm_info.get("total_calls", 0)
        age_str   = f"{age:.1f}s" if age >= 0 else "—"
        mdl_name  = vlm_info.get("model", "moondream")
        led_anim  = "animation:pulse 1.5s infinite;" if status == "ok" else ""
        _void_abstract = bool(
            metadata.get("is_abstract_or_void", False)
            or vlm_info.get("is_abstract_or_void", False)
            or st.session_state.get("void_mode_active", False)
        )
        _strict_ok = bool(
            vlm_info.get("strict_scent_allowed",
                st.session_state.get("strict_scent_allowed", True))
        )
        # Birleşik kokusuz mod: soyut VEYA belirgin koku kaynağı yok
        _any_void  = _void_abstract or not _strict_ok

        void_bg  = "#3f0000" if _any_void else "#0f172a"
        void_bdr = "#ef4444" if _any_void else "#1e3a5f"
        void_col = "#ef4444" if _any_void else "#22c55e"

        # Rozet 1: Soyut/Grafik Sahne (is_abstract_or_void)
        abs_lbl = "⛔ AKTİF" if _void_abstract else "✓ PASİF"
        abs_col = "#ef4444"  if _void_abstract else "#22c55e"
        # Rozet 2: Net Koku Kaynağı (strict_scent_allowed)
        sct_lbl = "⛔ YOK"   if not _strict_ok else "✓ VAR"
        sct_col = "#f59e0b"  if not _strict_ok else "#22c55e"

        st.html(
            f'<div style="background:{void_bg};border:1px solid {void_bdr};border-radius:8px;'
            f'padding:8px 12px;margin-bottom:6px;display:flex;align-items:center;gap:10px;">'
            f'<div style="width:10px;height:10px;border-radius:50%;background:{st_col};'
            f'box-shadow:0 0 6px {st_col};{led_anim}flex-shrink:0;"></div>'
            f'<div style="flex:1;">'
            f'<span style="color:#e2e8f0;font-size:0.80rem;font-weight:600;">VLM: {html.escape(mdl_name)}</span>'
            f'<span style="color:{st_col};font-size:0.72rem;margin-left:8px;">{status.upper()}</span>'
            f'</div>'
            f'<div style="color:#64748b;font-size:0.68rem;text-align:right;">'
            f'{ok_count}/{total_c} · {age_str} önce</div>'
            f'</div>'
            # ─ Rozet 1: Soyut/Void Sahne ─────────────────────────────────────
            f'<div style="background:{abs_col}15;border:1px solid {abs_col}44;'
            f'border-radius:6px;padding:5px 10px;margin-bottom:4px;'
            f'display:flex;align-items:center;justify-content:space-between;">'
            f'<span style="color:#94a3b8;font-size:0.72rem;font-weight:600;">'
            f'🚫 Soyut/Grafik Sahne</span>'
            f'<span style="background:{abs_col}33;color:{abs_col};'
            f'border:1px solid {abs_col}77;border-radius:4px;'
            f'padding:2px 8px;font-size:0.70rem;font-weight:700;">{abs_lbl}</span>'
            f'</div>'
            # ─ Rozet 2: Net Koku Kaynağı ────────────────────────────────────
            f'<div style="background:{sct_col}15;border:1px solid {sct_col}44;'
            f'border-radius:6px;padding:5px 10px;margin-bottom:6px;'
            f'display:flex;align-items:center;justify-content:space-between;">'
            f'<span style="color:#94a3b8;font-size:0.72rem;font-weight:600;">'
            f'👃 Koku Kaynağı '
            f'<span style="color:#475569;font-weight:400;font-size:0.67rem;">(Net Ortam)</span></span>'
            f'<span style="background:{sct_col}33;color:{sct_col};'
            f'border:1px solid {sct_col}77;border-radius:4px;'
            f'padding:2px 8px;font-size:0.70rem;font-weight:700;">{sct_lbl}</span>'
            f'</div>'
        )
        if _any_void:
            st.html(
                f'<div style="background:#1e0a00;border:1px solid #b45309;'
                f'border-radius:6px;padding:6px 12px;margin-bottom:8px;'
                f'display:flex;align-items:center;gap:8px;">'
                f'<span style="font-size:0.85rem;">&#x26A0;&#xFE0F;</span>'
                f'<span style="color:#fbbf24;font-size:0.73rem;font-weight:600;">'
                f'Sistem Durumu: Sadece Net Ortamlar (Kokusuz Alan)</span>'
                f'</div>'
            )

        # VCV Conflict Özeti
        urban_sig  = metadata.get("vcv_urban_sig",  0.5)
        nature_sig = metadata.get("vcv_nature_sig", 0.5)
        terr_sig   = metadata.get("vcv_terrestrial", 0.5)
        aquat_sig  = metadata.get("vcv_aquatic",     0.5)
        st.markdown(
            f'**🌀 VCV** — '
            f'<span style="color:#38bdf8;font-size:0.74rem;">'
            f'kentsel:{urban_sig:.2f} doğal:{nature_sig:.2f} '
            f'karasal:{terr_sig:.2f} sucul:{aquat_sig:.2f}</span>',
            unsafe_allow_html=True,
        )
        vcv_debug = metadata.get("vcv_debug", {})
        if vcv_debug:
            rows_vcv = ""
            for s in sorted(SCENTS, key=lambda x: vcv_debug.get(x, {}).get("conflict", 0), reverse=True):
                d        = vcv_debug.get(s, {})
                conflict = float(d.get("conflict", 0.0))
                decay    = float(d.get("decay",    1.0))
                eff_lam  = float(d.get("eff_lam",  2.5))
                col_s    = SCENT_COLORS.get(s, "#94a3b8")
                icon     = SCENT_ICONS.get(s, "")
                conf_pct = int(conflict * 100)
                conf_r   = min(255, int(conflict * 510))
                conf_g   = max(0,   255 - int(conflict * 400))
                conf_col = f"rgb({conf_r},{conf_g},60)"
                lam_col  = "#f97316" if eff_lam > 2.5 else "#64748b"
                rows_vcv += (
                    f'<div style="margin-bottom:5px;">'
                    f'<div style="display:flex;justify-content:space-between;">'
                    f'<span style="color:{col_s};font-size:0.73rem;">{icon} {html.escape(s)}</span>'
                    f'<span style="color:#94a3b8;font-size:0.68rem;">'
                    f'⚡{conflict:.2f}→×{decay:.2f} '
                    f'<span style="color:{lam_col}">λ{eff_lam:.1f}</span></span>'
                    f'</div>'
                    f'<div style="background:#1e293b;border-radius:3px;height:4px;margin-top:2px;">'
                    f'<div style="width:{conf_pct}%;height:100%;background:{conf_col};border-radius:3px;"></div>'
                    f'</div></div>'
                )
            st.html(f'<div>{rows_vcv}</div>')

        # ── V3: Kalman Filtre Banka Canlı Geçmiş Grafiği ──────────────────────
        st.markdown("---")
        st.markdown("**📡 Katman D — Kalman Çıkış Geçmişi (Son 50 Kare)**")
        kal_hist = list(st.session_state.kalman_history)
        if len(kal_hist) < 2:
            _zr = {s: 0.0 for s in SCENTS}
            kal_hist = [_zr, _zr]  # zero-pad → chart her zaman render edilir
        df_kal = pd.DataFrame(kal_hist)
        fig_k  = go.Figure()
        for s in SCENTS:
            if s not in df_kal.columns:
                df_kal[s] = 0.0
            hx = SCENT_COLORS.get(s, "#94a3b8").lstrip("#")
            r2, g2, b2 = int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16)
            fig_k.add_trace(go.Scatter(
                y=df_kal[s],
                name=SCENT_ICONS.get(s, "") + " " + s,
                mode="lines",
                line=dict(color=f"#{hx}", width=1.6),
                fill="tozeroy",
                fillcolor=f"rgba({r2},{g2},{b2},0.07)",
            ))
        fig_k.update_layout(
            height=185, margin=dict(l=0, r=0, t=4, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02,
                font=dict(size=8), itemwidth=40,
            ),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=True, gridcolor="#1e293b", range=[0, 1],
                       tickfont=dict(size=8)),
        )
        st.plotly_chart(fig_k, use_container_width=True, config={"displayModeBar": False})

        # Histerezis rozetleri
        st.markdown("**⚡ Histerezis Durumu**")
        hyst = metadata.get("hysteresis_state", {})
        hyst_parts = ""
        for s in SCENTS:
            active = bool(hyst.get(s, False))
            col_h  = SCENT_COLORS.get(s, "#94a3b8")
            icon   = SCENT_ICONS.get(s, "")
            bg     = f"{col_h}22" if active else "#1e293b"
            border = f"{col_h}88" if active else "#334155"
            label  = "AKTİF" if active else "pasif"
            hyst_parts += (
                f'<span style="display:inline-block;background:{bg};'
                f'border:1px solid {border};border-radius:4px;'
                f'padding:2px 7px;margin:2px;font-size:0.68rem;color:{col_h};">'
                f'{icon} {html.escape(s)}: {label}</span>'
            )
        st.html(f'<div style="margin-bottom:10px;">{hyst_parts}</div>')

        # Nihai PWM Değerleri
        st.markdown("**🔌 Nihai PWM Değerleri (max 4095)**")
        pwm_max  = 4095
        pwm_rows = ""
        for s in SCENTS:
            score = float(scores.get(s, 0.0))
            att   = float(attenuation.get(s, 1.0))
            pwm   = int(score * att * pwm_max)
            pct   = int(score * att * 100)
            col_s = SCENT_COLORS.get(s, "#94a3b8")
            icon  = SCENT_ICONS.get(s, "")
            pwm_rows += (
                f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:5px;">'
                f'<span style="color:{col_s};font-size:0.75rem;min-width:72px;">{icon} {html.escape(s)}</span>'
                f'<div style="flex:1;background:#1e293b;border-radius:3px;height:6px;">'
                f'<div style="width:{pct}%;height:100%;background:{col_s};border-radius:3px;"></div>'
                f'</div>'
                f'<span style="color:#94a3b8;font-size:0.72rem;min-width:60px;text-align:right;">'
                f'{pwm}/{pwm_max}</span>'
                f'</div>'
            )
        st.markdown(f'<div>{pwm_rows}</div>', unsafe_allow_html=True)

        urban_bonus = float(metadata.get("urban_is_duman_bonus", 0.0))
        if urban_bonus > 0:
            st.html(
                f'<div style="background:#431407;border:1px solid #f97316;border-radius:6px;'
                f'padding:6px 10px;margin-top:8px;">'
                f'<div style="color:#fb923c;font-size:0.76rem;font-weight:600;">🏙️ Urban Boost</div>'
                f'<div style="color:#94a3b8;font-size:0.70rem;">İs/Duman +{urban_bonus:.3f}</div>'
                f'</div>'
            )


# ─── Fragment: Debug Modu (eski — kullanılmıyor) ──────────────────────────────
def _debug_fragment():
    metadata    = st.session_state.last_metadata
    scores      = st.session_state.last_scores
    attenuation = metadata.get("attenuation", {s: 1.0 for s in SCENTS}) if metadata else {}

    if not metadata:
        st.info("Analiz başlatıldığında debug verileri burada görünecek.")
        return

    st.markdown(
        '<div class="dk-card-title">'
        '🔧 Canlı Debug Skor Panosu — Ham Model Çıktıları &amp; PDM Karar Akışı'
        '</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 1, 1], gap="medium")

    # ── Sütun 1: Girdi & Modeller ────────────────────────────────────────────
    with col1:
        st.markdown("##### 📷 Girdi & Modeller")

        ov = st.session_state.get("ov_frame")
        if ov:
            st.image(ov, caption="Son Kare (YOLO overlay)", use_container_width=True)

        st.markdown("**🔍 YOLO — En Güçlü 3 Nesne**")
        yolo_top3 = metadata.get("yolo_top3", [])
        if yolo_top3:
            for det in yolo_top3:
                lbl  = str(det.get("label", "?"))
                conf = int(det.get("confidence", 0.0) * 100)
                area = round(float(det.get("area_ratio", 0.0)) * 100, 1)
                st.markdown(
                    f'<div style="margin-bottom:7px;">'
                    f'<div style="display:flex;justify-content:space-between;">'
                    f'<b style="color:#e2e8f0;font-size:0.82rem;">{html.escape(lbl)}</b>'
                    f'<span style="color:#94a3b8;font-size:0.75rem;">%{conf} · alan %{area}</span>'
                    f'</div>'
                    f'<div style="background:#1e293b;border-radius:4px;height:6px;margin-top:3px;">'
                    f'<div style="width:{conf}%;height:100%;background:#34d399;border-radius:4px;"></div>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<span style="color:#475569;font-size:0.8rem;">Nesne tespit edilmedi</span>',
                unsafe_allow_html=True,
            )
        penalty_log = metadata.get("yolo_penalty_log", {})
        if penalty_log:
            st.markdown("**⚠️ Uygulanan Cezalar** (conf<0.55 & alan<%5 → ×0.20)")
            pen_html = "".join(
                f'<span style="display:inline-block;background:#7f1d1d33;'
                f'border:1px solid #ef444488;border-radius:4px;'
                f'padding:2px 7px;margin:2px;font-size:0.68rem;color:#fca5a5;">'
                f'{html.escape(lbl)}: ×{mult}</span>'
                for lbl, mult in penalty_log.items()
            )
            st.html(f'<div style="margin-bottom:6px;">{pen_html}</div>')

        st.markdown("---")
        st.markdown("**🧠 CLIP — En Güçlü 3 Sahne**")
        clip_top3 = metadata.get("clip_top3", [])
        has_clip  = bool(clip_top3) and any(v > 0 for _, v in clip_top3)
        if has_clip:
            for scent, prob in clip_top3:
                pct   = int(prob * 100)
                col_c = SCENT_COLORS.get(scent, "#94a3b8")
                icon  = SCENT_ICONS.get(scent, "")
                st.markdown(
                    f'<div style="margin-bottom:7px;">'
                    f'<div style="display:flex;justify-content:space-between;">'
                    f'<span style="color:{col_c};font-size:0.82rem;">{icon} <b>{html.escape(scent)}</b></span>'
                    f'<span style="color:#94a3b8;font-size:0.75rem;">%{pct}</span>'
                    f'</div>'
                    f'<div style="background:#1e293b;border-radius:4px;height:6px;margin-top:3px;">'
                    f'<div style="width:{pct}%;height:100%;background:{col_c};border-radius:4px;"></div>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<span style="color:#475569;font-size:0.8rem;">'
                'CLIP henüz çalışmadı (ilk 15 karede başlar)</span>',
                unsafe_allow_html=True,
            )

    # ── Sütun 2: Renk & PDM Karar ────────────────────────────────────────────
    with col2:
        st.markdown("##### 🎨 Renk & PDM Karar")

        st.markdown("**🎨 K-Means — Baskın Renkler (HSV)**")
        roi_colors = metadata.get("roi_colors_bgr", [])
        if roi_colors:
            blocks = ""
            for bgr in roi_colors[:5]:
                b, g, r  = int(bgr[0]), int(bgr[1]), int(bgr[2])
                bgr_arr  = np.uint8([[[b, g, r]]])
                hsv_arr  = cv2.cvtColor(bgr_arr, cv2.COLOR_BGR2HSV)[0][0]
                h_deg    = int(hsv_arr[0]) * 2
                s_pct    = int(hsv_arr[1] / 255 * 100)
                v_pct    = int(hsv_arr[2] / 255 * 100)
                blocks  += (
                    f'<div style="display:inline-flex;flex-direction:column;align-items:center;'
                    f'margin:3px;gap:3px;vertical-align:top;">'
                    f'<div style="width:44px;height:44px;background:rgb({r},{g},{b});'
                    f'border-radius:6px;border:1px solid rgba(255,255,255,0.15);"></div>'
                    f'<span style="color:#64748b;font-size:0.62rem;text-align:center;line-height:1.4;">'
                    f'H:{h_deg}°<br>S:{s_pct}%<br>V:{v_pct}%</span>'
                    f'</div>'
                )
            st.html(f'<div style="display:flex;flex-wrap:wrap;gap:4px;margin-bottom:10px;">{blocks}</div>')
        else:
            st.markdown(
                '<span style="color:#475569;font-size:0.8rem;">Renk verisi yok</span>',
                unsafe_allow_html=True,
            )
        neutral_ct = metadata.get("kmeans_neutral_filtered", 0)
        if neutral_ct > 0:
            st.markdown(
                f'<span style="color:#64748b;font-size:0.75rem;">⛔ {neutral_ct} nötr renk filtrelendi (S/V<%15)</span>',
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown(
            "**⚖️ PDM 4-Yollu Matris Snapshotu**  "
            '<span style="color:#64748b;font-size:0.7rem;">Y=YOLO K=KMeans T=Doku C=CLIP</span>',
            unsafe_allow_html=True,
        )
        blended_4way = metadata.get("blended_4way", {})
        mod_scores   = metadata.get("module_scores", {})
        if blended_4way:
            rows_html = ""
            sorted_4w = sorted(SCENTS, key=lambda s: blended_4way.get(s, 0), reverse=True)
            for s in sorted_4w:
                b4  = int(blended_4way.get(s, 0.0) * 100)
                yk  = int(mod_scores.get("yolo",    {}).get(s, 0.0) * 100)
                km  = int(mod_scores.get("kmeans",  {}).get(s, 0.0) * 100)
                tx  = int(mod_scores.get("texture", {}).get(s, 0.0) * 100)
                cl  = int(mod_scores.get("ovl",     {}).get(s, 0.0) * 100)
                col_s = SCENT_COLORS.get(s, "#94a3b8")
                icon  = SCENT_ICONS.get(s, "")
                rows_html += (
                    f'<div style="margin-bottom:8px;">'
                    f'<div style="display:flex;justify-content:space-between;margin-bottom:2px;">'
                    f'<span style="color:{col_s};font-size:0.78rem;">{icon} {html.escape(s)}</span>'
                    f'<b style="color:#e2e8f0;font-size:0.78rem;">%{b4}</b>'
                    f'</div>'
                    f'<div style="display:flex;gap:6px;font-size:0.65rem;color:#64748b;margin-bottom:3px;">'
                    f'<span>Y:%{yk}</span><span>K:%{km}</span>'
                    f'<span>T:%{tx}</span><span>C:%{cl}</span>'
                    f'</div>'
                    f'<div style="background:#1e293b;border-radius:3px;height:5px;">'
                    f'<div style="width:{b4}%;height:100%;background:{col_s};border-radius:3px;"></div>'
                    f'</div></div>'
                )
            st.html(f'<div>{rows_html}</div>')
        else:
            st.markdown(
                '<span style="color:#475569;font-size:0.8rem;">Veri bekleniyor…</span>',
                unsafe_allow_html=True,
            )

    # ── Sütun 3: VCV & Nihai Çıktı ───────────────────────────────────────────
    with col3:
        st.markdown("##### 🔬 VCV & Nihai Çıktı")

        urban_sig   = metadata.get("vcv_urban_sig",   0.5)
        nature_sig  = metadata.get("vcv_nature_sig",  0.5)
        terr_sig    = metadata.get("vcv_terrestrial", 0.5)
        aquat_sig   = metadata.get("vcv_aquatic",     0.5)
        st.markdown(
            f'**🌀 VCV — Conflict & Sönümleme**  '
            f'<span style="color:#38bdf8;font-size:0.75rem;">'
            f'kentsel:{urban_sig:.2f} | doğal:{nature_sig:.2f}</span>  '
            f'<span style="color:#86efac;font-size:0.72rem;">'
            f'🏔️karasal:{terr_sig:.2f} | 🌊sucul:{aquat_sig:.2f}</span>',
            unsafe_allow_html=True,
        )
        vcv_debug = metadata.get("vcv_debug", {})
        if vcv_debug:
            rows_vcv = ""
            for s in sorted(SCENTS, key=lambda x: vcv_debug.get(x, {}).get("conflict", 0), reverse=True):
                d        = vcv_debug.get(s, {})
                conflict = float(d.get("conflict", 0.0))
                decay    = float(d.get("decay",    1.0))
                eff_lam  = float(d.get("eff_lam",  2.5))
                col_s    = SCENT_COLORS.get(s, "#94a3b8")
                icon     = SCENT_ICONS.get(s, "")
                conf_pct = int(conflict * 100)
                conf_r   = min(255, int(conflict * 510))
                conf_g   = max(0,   255 - int(conflict * 400))
                conf_col = f"rgb({conf_r},{conf_g},60)"
                hardened = eff_lam > 2.5
                lam_color = "#f97316" if hardened else "#64748b"
                rows_vcv += (
                    f'<div style="margin-bottom:6px;">'
                    f'<div style="display:flex;justify-content:space-between;">'
                    f'<span style="color:{col_s};font-size:0.75rem;">{icon} {html.escape(s)}</span>'
                    f'<span style="color:#94a3b8;font-size:0.70rem;">'
                    f'⚡{conflict:.2f} → ×{decay:.2f} '
                    f'<span style="color:{lam_color}">λ{eff_lam:.1f}</span></span>'
                    f'</div>'
                    f'<div style="background:#1e293b;border-radius:3px;height:4px;margin-top:2px;">'
                    f'<div style="width:{conf_pct}%;height:100%;background:{conf_col};border-radius:3px;"></div>'
                    f'</div></div>'
                )
            st.html(f'<div>{rows_vcv}</div>')
        else:
            st.markdown(
                '<span style="color:#475569;font-size:0.8rem;">VCV verisi bekleniyor…</span>',
                unsafe_allow_html=True,
            )

        urban_bonus = float(metadata.get("urban_is_duman_bonus", 0.0))
        if urban_bonus > 0:
            st.html(
                f'<div style="background:#431407;border:1px solid #f97316;border-radius:6px;'
                f'padding:7px 10px;margin:8px 0;">'
                f'<div style="color:#fb923c;font-size:0.78rem;font-weight:600;">🏙️ Urban Boost Aktif</div>'
                f'<div style="color:#94a3b8;font-size:0.72rem;margin-top:4px;">'
                f'İs/Duman\'a <b style="color:#fdba74;">+{urban_bonus:.3f}</b> taban bonus eklendi '
                f'<span style="color:#64748b;">(urban×0.15 | eşik: 0.20)</span></div>'
                f'</div>'
            )

        st.markdown("---")
        st.markdown("**⚡ Histerezis Durumu** (etkinleştir: 0.20 / kapat: 0.08)")
        hyst = metadata.get("hysteresis_state", {})
        if hyst:
            hyst_parts = ""
            for s in SCENTS:
                active = bool(hyst.get(s, False))
                col_h  = SCENT_COLORS.get(s, "#94a3b8")
                icon   = SCENT_ICONS.get(s, "")
                bg     = f"{col_h}22" if active else "#1e293b"
                border = f"{col_h}88" if active else "#334155"
                label  = "AKTİF" if active else "pasif"
                hyst_parts += (
                    f'<span style="display:inline-block;background:{bg};'
                    f'border:1px solid {border};border-radius:4px;'
                    f'padding:2px 7px;margin:2px;font-size:0.68rem;color:{col_h};">'
                    f'{icon} {html.escape(s)}: {label}</span>'
                )
            st.html(f'<div style="margin-bottom:10px;">{hyst_parts}</div>')

        st.markdown("---")
        st.markdown("**🔌 Nihai PWM Kanal Değerleri (max 4095)**")
        pwm_max  = 4095
        pwm_rows = ""
        for s in SCENTS:
            score = float(scores.get(s, 0.0))
            att   = float(attenuation.get(s, 1.0))
            pwm   = int(score * att * pwm_max)
            pct   = int(score * att * 100)
            col_s = SCENT_COLORS.get(s, "#94a3b8")
            icon  = SCENT_ICONS.get(s, "")
            pwm_rows += (
                f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:5px;">'
                f'<span style="color:{col_s};font-size:0.75rem;min-width:72px;">{icon} {html.escape(s)}</span>'
                f'<div style="flex:1;background:#1e293b;border-radius:3px;height:6px;">'
                f'<div style="width:{pct}%;height:100%;background:{col_s};border-radius:3px;"></div>'
                f'</div>'
                f'<span style="color:#94a3b8;font-size:0.72rem;min-width:60px;text-align:right;">'
                f'{pwm} / {pwm_max}</span>'
                f'</div>'
            )
        st.markdown(f'<div>{pwm_rows}</div>', unsafe_allow_html=True)


# ─── Fragment: V3 Semantik Katmanlar ─────────────────────────────────────────
@st.fragment(run_every=1.0)
def _v3_fragment():
    metadata = st.session_state.last_metadata
    if not metadata:
        st.info("Analiz başlatıldığında V3 katman verileri burada görünecek.")
        return

    st.markdown(
        '<div class="dk-card-title">'
        '🧬 V3 Semantik Katmanlar — VLM Maske · Bayes Posterior · Kalman Durumu'
        '</div>',
        unsafe_allow_html=True,
    )

    vlm_col, kal_col = st.columns([1, 1], gap="large")

    # ── Katman B: VLM Semantik Maske ──────────────────────────────────────
    with vlm_col:
        st.markdown("##### 🔮 Katman B — VLM Semantik Maske")

        vlm_info = metadata.get("v3_vlm_status", {})
        status   = str(vlm_info.get("vlm_status", "disabled"))
        STATUS_COLORS = {"ok": "#22c55e", "init": "#f59e0b", "stale": "#ef4444",
                         "error": "#ef4444", "disabled": "#64748b"}
        status_col = STATUS_COLORS.get(status, "#94a3b8")
        age        = vlm_info.get("last_valid_age_s", -1)
        model      = vlm_info.get("model", "—")
        ok_count   = vlm_info.get("successful_calls", 0)
        total_c    = vlm_info.get("total_calls", 0)
        age_str    = f"{age:.1f}s önce" if age >= 0 else "henüz çalışmadı"

        st.html(
            f'<div style="background:#0f172a;border:1px solid #1e3a5f;border-radius:8px;'
            f'padding:10px 14px;margin-bottom:10px;">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;">'
            f'<span style="color:#e2e8f0;font-size:0.82rem;font-weight:600;">{model}</span>'
            f'<span style="background:{status_col}33;color:{status_col};border:1px solid {status_col}66;'
            f'border-radius:4px;padding:2px 8px;font-size:0.70rem;font-weight:600;">{status.upper()}</span>'
            f'</div>'
            f'<div style="color:#64748b;font-size:0.72rem;margin-top:6px;">'
            f'Son güncelleme: {age_str} &nbsp;|&nbsp; Başarılı: {ok_count}/{total_c}'
            f'</div></div>'
        )

        last_ctx = vlm_info.get("last_context", {})
        if last_ctx:
            env     = last_ctx.get("environment", "unknown")
            has_veg = last_ctx.get("organic_vegetation", "?")
            has_wat = last_ctx.get("water_body", "?")
            has_fir = last_ctx.get("fire_smoke", "?")
            conf    = float(last_ctx.get("confidence", 0.0))
            st.html(
                f'<div style="background:#0f172a;border:1px solid #1e3a5f;border-radius:8px;'
                f'padding:10px 14px;margin-bottom:10px;">'
                f'<div style="color:#38bdf8;font-size:0.75rem;font-weight:600;'
                f'text-transform:uppercase;letter-spacing:.05em;margin-bottom:6px;">Son Sahne Analizi</div>'
                f'<div style="color:#e2e8f0;font-size:0.80rem;">🌍 {html.escape(env)}</div>'
                f'<div style="color:#94a3b8;font-size:0.73rem;margin-top:4px;">'
                f'🌿 Bitki: {"✓" if has_veg else "✗"} &nbsp;'
                f'💧 Su: {"✓" if has_wat else "✗"} &nbsp;'
                f'🔥 Duman: {"✓" if has_fir else "✗"} &nbsp;'
                f'Güven: {conf:.2f}'
                f'</div></div>'
            )

        st.markdown("**🎭 Semantik Maske Değerleri**")
        mask = metadata.get("v3_semantic_mask", {})
        if mask:
            mask_rows = ""
            for s in SCENTS:
                v     = float(mask.get(s, 1.0))
                col_s = SCENT_COLORS.get(s, "#94a3b8")
                icon  = SCENT_ICONS.get(s, "")
                pct   = int(v * 100)
                veto_col = "#ef4444" if v < 0.20 else ("#f59e0b" if v < 0.60 else "#22c55e")
                mask_rows += (
                    f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:5px;">'
                    f'<span style="color:{col_s};font-size:0.75rem;min-width:80px;">{icon} {html.escape(s)}</span>'
                    f'<div style="flex:1;background:#1e293b;border-radius:3px;height:6px;">'
                    f'<div style="width:{pct}%;height:100%;background:{veto_col};border-radius:3px;"></div>'
                    f'</div>'
                    f'<span style="color:{veto_col};font-size:0.72rem;min-width:40px;text-align:right;">'
                    f'×{v:.2f}</span>'
                    f'</div>'
                )
            st.html(f'<div>{mask_rows}</div>')

        # ── K-Means Tamer: Mod ve Dinamik Ağırlıklar ─────────────────────────
        st.markdown("---")
        st.markdown("**🎯 K-Means Tamer — Dinamik Ağırlıklandırma**")
        tamer_mode2    = metadata.get("kmeans_tamer_mode", "Normal")
        tamer_weights2 = metadata.get("kmeans_dyn_weights", {})
        if tamer_weights2:
            _TAMER_COLORS2 = {
                "Aksiyon":           "#ef4444",
                "Kurtarıcı":         "#22c55e",
                "Kurtarıcı+Emniyet": "#f59e0b",
                "Kentsel":           "#3b82f6",
                "Normal":            "#64748b",
            }
            _base_w2  = 0.15
            _mc2      = _TAMER_COLORS2.get(tamer_mode2, "#64748b")
            tamer_rows2 = ""
            for s in SCENTS:
                w     = float(tamer_weights2.get(s, _base_w2))
                col_s = SCENT_COLORS.get(s, "#94a3b8")
                icon  = SCENT_ICONS.get(s, "")
                ratio = w / _base_w2 if _base_w2 > 0 else 1.0
                bar   = min(int(ratio * 50), 100)
                if w < 0.02:
                    wc, wl = "#ef4444", "VETO"
                elif w > _base_w2 + 0.01:
                    wc, wl = "#22c55e", "GÜÇLÜ"
                else:
                    wc, wl = "#64748b", "Std"
                tamer_rows2 += (
                    f'<div style="display:flex;align-items:center;gap:5px;margin-bottom:4px;">'
                    f'<span style="color:{col_s};font-size:0.73rem;min-width:80px;">{icon} {html.escape(s)}</span>'
                    f'<div style="flex:1;background:#1e293b;border-radius:3px;height:6px;">'
                    f'<div style="width:{bar}%;height:100%;background:{wc};border-radius:3px;"></div>'
                    f'</div>'
                    f'<span style="color:{wc};font-size:0.68rem;min-width:40px;text-align:right;">'
                    f'w={w:.3f}</span>'
                    f'<span style="background:{wc}22;color:{wc};border:1px solid {wc}55;'
                    f'border-radius:3px;padding:1px 4px;font-size:0.58rem;min-width:38px;text-align:center;">'
                    f'{wl}</span>'
                    f'</div>'
                )
            st.html(
                f'<div style="background:#0f172a;border:1px solid {_mc2}44;border-radius:7px;'
                f'padding:8px 10px;">'
                f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:7px;">'
                f'<span style="color:#94a3b8;font-size:0.72rem;">Aktif Mod</span>'
                f'<span style="background:{_mc2}33;color:{_mc2};border:1px solid {_mc2}66;'
                f'border-radius:4px;padding:2px 8px;font-size:0.70rem;font-weight:700;">'
                f'{html.escape(tamer_mode2)}</span>'
                f'</div>'
                f'{tamer_rows2}</div>'
            )
        else:
            st.html('<div style="color:#475569;font-size:0.80rem;">Tamer veri bekleniyor…</div>')

    # ── Katman D: Kalman Filtre Durumları ─────────────────────────────────
    with kal_col:
        st.markdown("##### 📡 Katman D — Kalman Filtre Banka Durumu")

        kal_states = metadata.get("v3_kalman_states", {})
        if kal_states:
            kal_rows = ""
            for s in SCENTS:
                state = kal_states.get(s, {})
                x_val = float(state.get("x", 0.0))
                p_val = float(state.get("P", 1.0))
                q_val = float(state.get("Q", 0.01))
                r_val = float(state.get("R", 0.10))
                k_val = float(state.get("K", 0.5))
                col_s = SCENT_COLORS.get(s, "#94a3b8")
                icon  = SCENT_ICONS.get(s, "")
                x_pct = int(x_val * 100)
                k_pct = int(k_val * 100)
                kal_rows += (
                    f'<div style="margin-bottom:8px;">'
                    f'<div style="display:flex;justify-content:space-between;">'
                    f'<span style="color:{col_s};font-size:0.78rem;font-weight:600;">{icon} {html.escape(s)}</span>'
                    f'<span style="color:#94a3b8;font-size:0.68rem;">'
                    f'x={x_val:.3f} P={p_val:.3f} K={k_val:.3f}</span>'
                    f'</div>'
                    f'<div style="background:#1e293b;border-radius:3px;height:5px;margin:3px 0;">'
                    f'<div style="width:{x_pct}%;height:100%;background:{col_s};border-radius:3px;"></div>'
                    f'</div>'
                    f'<div style="color:#475569;font-size:0.65rem;">'
                    f'Q={q_val} R={r_val} · Kazanç: %{k_pct}</div>'
                    f'</div>'
                )
            st.html(f'<div>{kal_rows}</div>')
        else:
            st.markdown(
                '<span style="color:#475569;font-size:0.8rem;">Kalman verisi bekleniyor…</span>',
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown("##### ⚛️ Katman C — Bayes Posterior")
        posterior = metadata.get("v3_bayes_posterior", {})
        mod_pdfs  = metadata.get("v3_module_pdfs", {})
        if posterior:
            bay_rows = ""
            for s in sorted(SCENTS, key=lambda x: posterior.get(x, 0.0), reverse=True):
                pv    = float(posterior.get(s, 0.0))
                col_s = SCENT_COLORS.get(s, "#94a3b8")
                icon  = SCENT_ICONS.get(s, "")
                pct   = int(pv * 100)
                y_v   = round(float(mod_pdfs.get("yolo",    {}).get(s, 0.0)) * 100)
                c_v   = round(float(mod_pdfs.get("clip",    {}).get(s, 0.0)) * 100)
                k_v   = round(float(mod_pdfs.get("kmeans",  {}).get(s, 0.0)) * 100)
                t_v   = round(float(mod_pdfs.get("texture", {}).get(s, 0.0)) * 100)
                bay_rows += (
                    f'<div style="margin-bottom:7px;">'
                    f'<div style="display:flex;justify-content:space-between;">'
                    f'<span style="color:{col_s};font-size:0.78rem;">{icon} {html.escape(s)}</span>'
                    f'<span style="color:#94a3b8;font-size:0.65rem;">'
                    f'Y%{y_v} C%{c_v} K%{k_v} T%{t_v}</span>'
                    f'</div>'
                    f'<div style="background:#1e293b;border-radius:3px;height:5px;margin-top:3px;">'
                    f'<div style="width:{pct}%;height:100%;background:{col_s};border-radius:3px;"></div>'
                    f'</div></div>'
                )
            st.html(f'<div>{bay_rows}</div>')
        else:
            st.markdown(
                '<span style="color:#475569;font-size:0.8rem;">Posterior bekleniyor…</span>',
                unsafe_allow_html=True,
            )


# ─── Sekmelere yerleştir ──────────────────────────────────────────────────────
with tab_main:
    _main_fragment()

with tab_unified:
    _unified_fragment()
