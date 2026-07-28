"""
presentation.py — DijitalKoku Sunum Modu (Look-ahead)
Streamlit tabanlı sunum penceresi.

Çalıştırma:
    streamlit run presentation.py --server.port 8502
"""

import atexit
import sys
import os
import time
import math
import threading
import socket
import http.server
import urllib.parse
import tempfile

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pdm_engine import ProbabilisticDecisionMatrix, SCENTS
from analyzer import Analyzer

# ── Sayfa yapılandırması ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="DijitalKoku — Sunum Modu",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Sabitler ────────────────────────────────────────────────────────────────
LOOKAHEAD_SEC = 5.0
PULSE_PERIOD  = 7.5
PULSE_AMP     = 0.15

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
    "Çam":"\U0001f332", "Deniz":"\U0001f30a", "Vanilya":"\U0001f366",
    "İs/Duman":"\U0001f525", "Nane":"\U0001f33f", "Toprak":"\U0001f30d", "Limon":"\U0001f34b",
}

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;background:#0a0e1a;color:#e2e8f0;}
.pres-hdr{background:linear-gradient(135deg,#0d1b2a,#1a2744,#0d2137);
  border:1px solid #1e3a5f;border-radius:16px;padding:16px 24px;margin-bottom:14px;}
.pres-hdr h1{margin:0;font-size:1.6rem;font-weight:700;
  background:linear-gradient(90deg,#38bdf8,#818cf8);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.pres-hdr p{margin:4px 0 0;color:#64748b;font-size:0.82rem;}
.dk-panel{background:#0f1929;border:1px solid #1e3a5f;
  border-radius:12px;padding:14px 16px;margin-bottom:10px;}
.dk-ptitle{font-size:0.74rem;font-weight:600;letter-spacing:.1em;
  color:#38bdf8;text-transform:uppercase;margin-bottom:10px;}
.fut-panel{background:#0f1929;border:1px solid #312e81;
  border-radius:12px;padding:14px 16px;margin-bottom:10px;}
.fut-ptitle{font-size:0.74rem;font-weight:600;letter-spacing:.1em;
  color:#818cf8;text-transform:uppercase;margin-bottom:10px;}
.dbg-box{background:#060d14;border:1px solid #1e3a5f;border-radius:8px;
  padding:10px 14px;font-family:'Consolas',monospace;font-size:0.76rem;line-height:1.7;}
.s-row{margin-bottom:7px;}
.s-bg{background:#172033;border-radius:4px;height:13px;overflow:hidden;}
div[data-testid="stButton"]>button{border-radius:10px!important;font-weight:600!important;}
</style>
""", unsafe_allow_html=True)


# ── OlfactoryPulse ───────────────────────────────────────────────────────────
class OlfactoryPulse:
    def __init__(self):
        self._t0   = time.time()
        self.active = True

    def apply(self, scores: dict) -> dict:
        if not self.active:
            return dict(scores)
        t     = time.time() - self._t0
        delta = PULSE_AMP * math.sin(2 * math.pi * t / PULSE_PERIOD)
        return {s: max(0.0, min(1.0, v + delta * (1.0 - v))) for s, v in scores.items()}

    @property
    def phase_pct(self) -> float:
        t = time.time() - self._t0
        return (math.sin(2 * math.pi * t / PULSE_PERIOD) + 1.0) / 2.0 * 100.0


# ── Kaynak çözümleyici (YouTube / web URL desteği) ──────────────────────────
def _resolve_source(source: str) -> str:
    """
    Yerel dosya veya RTSP → doğrudan döndür.
    YouTube / web URL → yt-dlp ile direkt stream URL'ye çevir.
    """
    _YT_HOSTS = ("youtube.com", "youtu.be", "www.youtube.com")
    is_web = source.startswith(("http://", "https://", "rtmp://"))
    is_yt  = any(h in source for h in _YT_HOSTS)

    if is_web and is_yt:
        try:
            import yt_dlp, concurrent.futures
            ydl_opts = {
                "format":       "best[ext=mp4][height<=720]/best[height<=720]/best",
                "quiet":        True,
                "no_warnings":  True,
                "noplaylist":   True,   # playlist URL'lerini yok say — sadece tek video
                "socket_timeout": 10,   # ağ soketi zaman aşımı
            }

            def _fetch():
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(source, download=False)
                    return (info.get("url")
                            or (info.get("formats", [{}])[-1].get("url", "")))

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(_fetch)
                url = fut.result(timeout=25)   # 25sn genel zaman aşımı

            if not url:
                raise ValueError("yt-dlp stream URL bulunamadı")
            return url
        except concurrent.futures.TimeoutError:
            raise RuntimeError(
                "YouTube URL çözümlemesi zaman aşımına uğradı (25sn).\n"
                "Lütfen tekrar deneyin veya yerel dosya kullanın."
            )
        except ImportError:
            raise RuntimeError(
                "YouTube URL için yt-dlp gerekli.\n"
                "Çalıştır: pip install yt-dlp"
            )
        except Exception as e:
            raise RuntimeError(f"YouTube URL çözümlenemedi: {e}")
    return source


# ── Video HTTP Sunucusu (tarayıcı native oynatım + JS zaman takibi) ──────────
class _VideoHTTPServer:
    """
    Minimal HTTP sunucusu — Streamlit ana thread'inden tamamen bağımsız:
      /video  → Range isteklerini destekleyen video akışı (seek için gerekli)
      /time   → JS 'timeupdate' event'inden currentTime alır
    """
    def __init__(self, video_path: str | None):
        self._path   = video_path
        self._time   = 0.0
        self._lock   = threading.Lock()
        self._server = None
        self._port   = 0
        self._ready  = threading.Event()
        threading.Thread(target=self._serve, daemon=True).start()
        self._ready.wait(timeout=5.0)

    def _serve(self):
        srv = self

        class _H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass   # sunucu log'larını sustur

            def _cors(self):
                self.send_header("Access-Control-Allow-Origin",  "*")
                self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Range")

            def do_OPTIONS(self):
                self.send_response(200)
                self._cors()
                self.end_headers()

            def do_GET(self):
                parsed = urllib.parse.urlparse(self.path)
                qs     = urllib.parse.parse_qs(parsed.query)
                if parsed.path == "/time":
                    t = float(qs.get("t", ["0"])[0])
                    with srv._lock:
                        srv._time = t
                    self.send_response(204)
                    self._cors()
                    self.end_headers()
                elif parsed.path == "/video":
                    if srv._path is None:
                        self.send_response(404)
                        self.end_headers()
                    else:
                        self._stream()
                else:
                    self.send_response(404)
                    self.end_headers()

            def _stream(self):
                path = srv._path
                size = os.path.getsize(path)
                rng  = self.headers.get("Range", "")
                if rng:
                    parts  = rng.replace("bytes=", "").split("-")
                    start  = int(parts[0]) if parts[0] else 0
                    end    = int(parts[1]) if len(parts) > 1 and parts[1] else size - 1
                    end    = min(end, size - 1)
                    length = end - start + 1
                    self.send_response(206)
                    self.send_header("Content-Type",   "video/mp4")
                    self.send_header("Content-Range",  f"bytes {start}-{end}/{size}")
                    self.send_header("Content-Length", str(length))
                    self.send_header("Accept-Ranges",  "bytes")
                    self._cors()
                    self.end_headers()
                    with open(path, "rb") as f:
                        f.seek(start)
                        rem = length
                        while rem > 0:
                            chunk = f.read(min(65536, rem))
                            if not chunk:
                                break
                            try:
                                self.wfile.write(chunk)
                            except (BrokenPipeError, ConnectionResetError):
                                break
                            rem -= len(chunk)
                else:
                    self.send_response(200)
                    self.send_header("Content-Type",   "video/mp4")
                    self.send_header("Content-Length", str(size))
                    self.send_header("Accept-Ranges",  "bytes")
                    self._cors()
                    self.end_headers()
                    with open(path, "rb") as f:
                        while True:
                            chunk = f.read(65536)
                            if not chunk:
                                break
                            try:
                                self.wfile.write(chunk)
                            except (BrokenPipeError, ConnectionResetError):
                                break

        self._server = http.server.HTTPServer(("127.0.0.1", 0), _H)
        self._port   = self._server.server_address[1]
        self._ready.set()
        self._server.serve_forever()

    @property
    def current_time(self) -> float:
        with self._lock:
            return self._time

    @property
    def port(self) -> int:
        return self._port

    @property
    def video_url(self) -> str:
        return f"http://127.0.0.1:{self._port}/video"

    @property
    def time_url(self) -> str:
        return f"http://127.0.0.1:{self._port}/time"

    def stop(self):
        if self._server:
            try:
                self._server.shutdown()
            except Exception:
                pass


# ── Asenkron Look-ahead Analiz Motoru ───────────────────────────────────────
class RingBufferPlayer:
    """
    Video oynatımı tamamen tarayıcıya devredilmiştir (_VideoHTTPServer).
    Bu sınıf yalnızca arka plan analiz thread'ini yönetir:
      ANALYSIS : cap_analysis → video_server.current_time + LOOKAHEAD_SEC seek
                 Her ANALYSIS_INTERVAL iterasyonda bir GPU analizi yapar.
                 Ana thread veya Streamlit rerun'larından tamamen bağımsızdır.
    """
    ANALYSIS_INTERVAL = 15

    def __init__(self, source: str, az: Analyzer):
        resolved    = _resolve_source(source)
        self.source = resolved
        self.az     = az

        # Yerel dosya: HTTP sunucu video + zaman takibi sağlar
        # Uzak URL : HTTP sunucu yalnızca zaman takibi sağlar (video doğrudan oynatılır)
        _local = not resolved.startswith(("http://", "https://", "rtsp://"))
        self.video_server = _VideoHTTPServer(resolved if _local else None)
        self.video_html_src = self.video_server.video_url if _local else resolved

        # Analiz için ayrı VideoCapture (look-ahead seek)
        self.cap_analysis = cv2.VideoCapture(resolved)
        if not self.cap_analysis.isOpened():
            raise RuntimeError(
                f"Video açılamadı: {source}\n"
                "Desteklenen: yerel dosya (.mp4, .avi…) veya YouTube URL"
            )

        _raw_fps       = self.cap_analysis.get(cv2.CAP_PROP_FPS)
        self.fps       = _raw_fps if math.isfinite(_raw_fps) and _raw_fps > 0 else 30.0
        _raw_total     = int(self.cap_analysis.get(cv2.CAP_PROP_FRAME_COUNT))
        self.total_f   = _raw_total if _raw_total > 0 else -1
        self.total_sec = self.total_f / self.fps if self.total_f > 0 else 0.0

        self._running        = False
        self.future_engine   = None
        self._engine_loading = False
        self._engine_lock    = threading.Lock()
        self._results_lock   = threading.Lock()

        self.future_scores: dict = {s: 0.0 for s in SCENTS}
        self.future_meta  : dict = {}
        self.buffer_ready        = False
        self.status_msg          = "Hazır"

    # ── Yaşam döngüsü ────────────────────────────────────────────────
    def start(self):
        if self._running:
            return
        self._running = True
        threading.Thread(target=self._analysis_loop, daemon=True).start()

    def stop(self):
        self._running = False
        try:
            self.cap_analysis.release()
        except Exception:
            pass
        self.video_server.stop()

    # ── ANALYSIS: video_server.current_time + LOOKAHEAD_SEC → GPU ────
    def _analysis_loop(self):
        with self._engine_lock:
            if not self._engine_loading:
                self._engine_loading = True
                self.status_msg = "CLIP/YOLO modeli yükleniyor…"
                try:
                    self.future_engine = ProbabilisticDecisionMatrix()
                    self.status_msg    = "Model hazır — analiz başlıyor"
                except Exception as e:
                    self.status_msg      = f"Model hatası: {e}"
                    self._engine_loading = False
                    return
                self._engine_loading = False

        ana_cnt  = 0
        last_cur = 0.0

        while self._running:
            ana_cnt += 1
            if ana_cnt % self.ANALYSIS_INTERVAL != 0:
                time.sleep(0.01)
                continue

            cur  = self.current_sec
            la_f = int((cur + LOOKAHEAD_SEC) * self.fps)
            if self.total_f > 0:
                la_f = min(la_f, self.total_f - 1)

            if abs(cur - last_cur) > 3.0 and self.future_engine:
                try:
                    self.future_engine.reset()
                except Exception:
                    pass
            last_cur = cur

            self.cap_analysis.set(cv2.CAP_PROP_POS_FRAMES, la_f)
            ret, frame = self.cap_analysis.read()
            if not ret:
                time.sleep(0.05)
                continue

            try:
                import torch
                _small = cv2.resize(frame, (160, 120), interpolation=cv2.INTER_NEAREST)
                dets, colors, _motion, _scene_cut, _ = self.az.analyze(
                    frame,
                    prev_frame=None,
                    conf_threshold=0.35,
                    kmeans_frame=_small,
                )
                del _small
                scores, meta = self.future_engine.decide(
                    frame, dets, colors,
                    motion_score=_motion,
                    is_scene_cut=_scene_cut,
                )
                del frame

                with self._results_lock:
                    self.future_scores = scores
                    self.future_meta   = meta
                    self.buffer_ready  = True
                self.status_msg = f"+{LOOKAHEAD_SEC:.1f}sn → {la_f/max(1.0,self.fps):.0f}s Hazır"

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                self.status_msg = f"Analiz hatası: {e}"
                time.sleep(0.1)

    # ── API ──────────────────────────────────────────────────────────
    def consume_future(self) -> tuple[dict, dict]:
        with self._results_lock:
            return dict(self.future_scores), dict(self.future_meta)

    @property
    def current_sec(self) -> float:
        return self.video_server.current_time

    @property
    def progress(self) -> float:
        return min(1.0, self.current_sec / max(1.0, self.total_sec))

    @property
    def la_progress(self) -> float:
        return min(1.0, (self.current_sec + LOOKAHEAD_SEC) / max(1.0, self.total_sec))


# ── Yardımcı: koku barı HTML ─────────────────────────────────────────────────
def _bars_html(scores: dict, dim: bool = False) -> str:
    rows = []
    for scent in SCENTS:
        val   = scores.get(scent, 0.0)
        color = SCENT_COLORS.get(scent, "#94a3b8")
        icon  = SCENT_ICONS.get(scent, "")
        pct   = int(val * 100)
        op    = "0.50" if dim else "1.00"
        lc    = "#4b5563" if dim else "#94a3b8"
        vc    = "#374151" if dim else color
        rows.append(
            f'<div class="s-row">'
            f'<div style="display:flex;justify-content:space-between;margin-bottom:3px;">'
            f'<span style="font-size:.81rem;color:{lc};">{icon} {scent}</span>'
            f'<span style="font-size:.77rem;font-weight:600;color:{vc};">{pct}%</span>'
            f'</div>'
            f'<div class="s-bg"><div style="width:{pct}%;height:100%;background:{color};'
            f'opacity:{op};border-radius:4px;transition:width .5s ease;"></div></div>'
            f'</div>'
        )
    return "".join(rows)


# ── Session State ─────────────────────────────────────────────────────────────
_defs: dict = {
    "pres_buf":           None,
    "pres_playing":       False,
    "pres_analyzer":      None,
    "pres_pulse":         None,
    "pres_future_scores": {s: 0.0 for s in SCENTS},
    "pres_future_meta":   {},
    "pres_status_msg":    "Hazır",
    "pres_progress":      0.0,
    "pres_la_progress":   0.0,
    "pres_tmp_path":      None,
    "pres_current_sec":   0.0,
    "pres_total_sec":     0.0,
}
for _k, _v in _defs.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

if st.session_state.pres_analyzer is None:
    st.session_state.pres_analyzer = Analyzer()
if st.session_state.pres_pulse is None:
    st.session_state.pres_pulse    = OlfactoryPulse()


# ── Başlık ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="pres-hdr">
  <h1>\U0001f3ac DijitalKoku — Sunum Modu</h1>
  <p>Look-ahead Buffer &nbsp;·&nbsp; Olfactory Pulsing &nbsp;·&nbsp; Real-time Scent Preview</p>
</div>""", unsafe_allow_html=True)


# ── Kontrol Paneli ────────────────────────────────────────────────────────────
with st.container():
    cc1, cc2, cc3 = st.columns([4, 2, 2])

    with cc1:
        url_in = st.text_input(
            "Video yolu / URL",
            placeholder="C:/video.mp4  veya  rtsp://...  veya  https://...",
            key="pres_url", label_visibility="collapsed")
        up_file = st.file_uploader(
            "veya dosya yükle", type=["mp4","avi","mov","mkv","webm"],
            key="pres_upload")

    with cc2:
        if st.button("▶ Aç / Başlat", use_container_width=True, type="primary",
                     key="pres_open"):
            if st.session_state.pres_buf:
                st.session_state.pres_buf.stop()
                st.session_state.pres_buf = None
            src = None
            if up_file:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tmp.write(up_file.read())
                tmp.close()
                old_tmp = st.session_state.pres_tmp_path
                if old_tmp:
                    try: os.unlink(old_tmp)
                    except: pass
                st.session_state.pres_tmp_path = tmp.name
                atexit.register(
                    lambda p=tmp.name: os.unlink(p) if os.path.exists(p) else None
                )
                src = tmp.name
            elif url_in.strip():
                src = url_in.strip()
            if src:
                with st.spinner("Video açılıyor…"):
                    try:
                        buf = RingBufferPlayer(
                            src,
                            st.session_state.pres_analyzer,
                        )
                        buf.start()
                        st.session_state.pres_buf           = buf
                        st.session_state.pres_playing       = True
                        st.session_state.pres_pulse         = OlfactoryPulse()
                        st.session_state.pres_future_scores = {s: 0.0 for s in SCENTS}
                        st.session_state.pres_progress      = 0.0
                        st.session_state.pres_la_progress   = 0.0
                        st.session_state.pres_status_msg    = "Başlatıldı…"
                        st.toast("\u25b6 Sunum Modu başladı", icon="\U0001f3ac")
                    except Exception as e:
                        st.error(f"Video açılamadı: {e}")
            else:
                st.warning("Dosya se\u00e7in veya bir yol / URL girin.")

        if st.button("⏸ Duraklat / Devam", use_container_width=True, key="pres_pause"):
            st.session_state.pres_playing = not st.session_state.pres_playing

        if st.button("\u23f9 Durdur", use_container_width=True, key="pres_stop"):
            if st.session_state.pres_buf:
                st.session_state.pres_buf.stop()
                st.session_state.pres_buf = None
            st.session_state.pres_playing = False
            st.rerun()

    with cc3:
        pulse_on = st.toggle("\U0001f30a Olfaktif Dalgalanma", value=True,
                             key="pres_pulse_on")
        if st.session_state.pres_pulse:
            st.session_state.pres_pulse.active = pulse_on
        st.caption("Koku k\u00f6rl\u00fc\u011f\u00fcn\u00fc \u00f6nlemek i\u00e7in sin\u00fcs dalgas\u0131")

st.divider()

# ── HTML5 Oynatıcı HTML şablonu ──────────────────────────────────────────────
def _player_html(video_url: str, time_url: str) -> str:
    la = int(LOOKAHEAD_SEC)
    return f"""<!DOCTYPE html>
<html style="background:#0a0e1a;margin:0;padding:0;">
<body style="margin:0;padding:0;background:#0a0e1a;">
<div style="border-radius:10px;overflow:hidden;">
  <video id="dk_vp" controls autoplay muted playsinline
         style="width:100%;display:block;background:#000;"
         crossorigin="anonymous">
    <source src="{video_url}" type="video/mp4">
  </video>
  <div style="background:#0f172a;padding:5px 12px;display:flex;
              justify-content:space-between;align-items:center;
              border-top:1px solid #1e3a5f;">
    <span id="dk_time" style="font-family:monospace;color:#94a3b8;
          font-size:13px;font-weight:600;">00:00 / 00:00</span>
    <span style="color:#38bdf8;font-size:11px;">Look-ahead +{la}s aktif</span>
  </div>
</div>
<script>
const vp=document.getElementById('dk_vp');
const tl=document.getElementById('dk_time');
const TU='{time_url}';
let _lr=-1;
function fmt(s){{s=Math.floor(s||0);return String(Math.floor(s/60)).padStart(2,'0')+':'+String(s%60).padStart(2,'0');}}
vp.addEventListener('timeupdate',function(){{const t=vp.currentTime;tl.textContent=fmt(t)+' / '+fmt(vp.duration||0);if(Math.abs(t-_lr)>0.45){{_lr=t;fetch(TU+'?t='+t.toFixed(2)).catch(()=>{{}});}}}});
vp.addEventListener('seeked',function(){{_lr=-1;fetch(TU+'?t='+vp.currentTime.toFixed(2)).catch(()=>{{}});}});
</script>
</body></html>"""


# ── Video + Koku kolonları (fragment dışında — video iframe reset'lenmez) ─────────
_vid_col, _scent_col = st.columns([55, 45])

_buf_outer = st.session_state.pres_buf
with _vid_col:
    if _buf_outer:
        components.html(
            _player_html(_buf_outer.video_html_src, _buf_outer.video_server.time_url),
            height=430, scrolling=False,
        )
    else:
        st.markdown(
            '<div style="height:375px;background:#0f172a;border-radius:10px;'
            'display:flex;align-items:center;justify-content:center;'
            'color:#334155;font-size:.95rem;border:1px solid #1e3a5f;">'
            '\U0001f4c2 Video y\u00fcklemek i\u00e7in ▶ A\u00e7 / Ba\u015flat butonuna bas\u0131n</div>',
            unsafe_allow_html=True)
    _debug_slot = st.empty()    # fragment her 0.5s'de buraya yazar (replace)

with _scent_col:
    _scent_slot = st.empty()    # fragment koku barlarını buraya yazar (replace)


# ── Canlı Panel (Fragment — 0.5 sn'de bir yenilenir; video burada YOK) ────────
@st.fragment(run_every=0.5)
def _live_panel():
    buf   = st.session_state.pres_buf
    pulse = st.session_state.pres_pulse

    if buf:
        _fs, _fm = buf.consume_future()
        st.session_state.pres_future_scores = _fs
        st.session_state.pres_future_meta   = _fm
        st.session_state.pres_progress      = buf.progress
        st.session_state.pres_la_progress   = buf.la_progress
        st.session_state.pres_status_msg    = buf.status_msg
        st.session_state.pres_current_sec   = buf.current_sec
        st.session_state.pres_total_sec     = buf.total_sec

    future_s    = st.session_state.pres_future_scores
    pulsed_s    = pulse.apply(future_s) if pulse else future_s
    meta        = st.session_state.pres_future_meta
    pct_d       = st.session_state.pres_progress    * 100
    pct_la      = st.session_state.pres_la_progress * 100
    status      = st.session_state.pres_status_msg
    current_sec = st.session_state.pres_current_sec
    total_sec   = max(1.0, st.session_state.pres_total_sec)

    # ── Sol kolon: debug kutusu — slot'u replace et (video dokunulmaz) ────────
    with _debug_slot.container():
        playing = st.session_state.pres_playing and buf is not None

        ctx   = meta.get("macro_context",    "\u2014")
        lit   = meta.get("lighting_profile", "\u2014")
        top_k = max(future_s, key=future_s.get) if any(v > 0 for v in future_s.values()) else "\u2014"
        top_v = future_s.get(top_k, 0.0)
        mode  = "Aktif (Dalgalanma Modu)" if (pulse and pulse.active) else "Pasif"
        phase = f"{pulse.phase_pct:.0f}%" if pulse else "\u2014"

        # Skor detay satırları (top-3 koku)
        score_detail = meta.get("score_detail", {})
        sky_r        = meta.get("sky_blue_ratio", 0.0)
        yc_adj       = meta.get("yolo_clip_adj", {})
        _detail_html = ""
        if score_detail and any(v > 0 for v in future_s.values()):
            top3 = sorted(SCENTS, key=lambda _s: future_s.get(_s, 0.0), reverse=True)[:3]
            _rows = []
            for _s in top3:
                _d   = score_detail.get(_s, {})
                _cp  = _d.get("clip_pct",   0)
                _yp  = _d.get("yolo_pct",   0)
                _kp  = _d.get("kmeans_pct", 0)
                _gd  = _d.get("grid_delta", 0)
                _ap  = _d.get("adj_pct",    0)
                _extras = ""
                if abs(_gd) >= 2:
                    _extras += f" Grid:{_gd:+.0f}%"
                if abs(_ap) >= 2:
                    _extras += f" Düz:{_ap:+.0f}%"
                _rows.append(
                    f"{_s}: CLIP {_cp}% · YOLO {_yp}% · K-Means {_kp}%{_extras}"
                )
            _sky_str = f" · Gökyüzü Mavi: %{sky_r*100:.0f}" if sky_r > 0.15 else ""
            _detail_html = (
                f'<br><span style="color:#64748b;font-size:.70rem;">'
                + " &nbsp;|&nbsp; ".join(_rows)
                + _sky_str
                + "</span>"
            )

        st.markdown(
            f'<div class="dbg-box">'
            f'<span style="color:#38bdf8;">[Analiz Tamponu: {status}]</span><br>'
            f'<span style="color:#a3e635;">[Durum: Sahne={ctx} \u00b7 I\u015f\u0131k={lit}'
            f' \u00b7 Gelecek={top_k} %{top_v*100:.0f}]</span><br>'
            f'<span style="color:#fb923c;">[\u0130leri Analiz: +{LOOKAHEAD_SEC:.0f}sn'
            f' \u00b7 G\u00f6r\u00fcnt\u00fc=%{pct_d:.0f} \u00b7 Analiz=%{pct_la:.0f}]</span><br>'
            f'<span style="color:#f0abfc;">[Burun Adaptasyonu: {mode} \u00b7 Sin\u00fcsFaz\u0131={phase}]</span>'
            + _detail_html +
            f'</div>',
            unsafe_allow_html=True)

    # ── Sağ: Koku Barları + Gelecek + Sinüs ───────────────────────────────
    with _scent_slot.container():
        st.markdown(
            '<div class="dk-panel"><div class="dk-ptitle">'
            '\U0001f4e1 Mevcut Koku Yo\u011funlu\u011fu (Dalgal\u0131)</div>'
            + _bars_html(pulsed_s) + '</div>',
            unsafe_allow_html=True)

        st.markdown(
            '<div class="fut-panel"><div class="fut-ptitle">'
            '\U0001f52e Gelecek Sahne Tahmini (+5 sn)</div>'
            + _bars_html(future_s, dim=True) + '</div>',
            unsafe_allow_html=True)

        # Sinüs dalgası grafik (Plotly)
        t_now = time.time()
        xs    = list(range(60))
        ys    = [
            PULSE_AMP * math.sin(2 * math.pi * (i / 60 + t_now / PULSE_PERIOD))
            for i in xs
        ]
        col_wave = "#f0abfc" if (pulse and pulse.active) else "#374151"
        fig = go.Figure(go.Scatter(
            x=xs, y=ys, mode="lines",
            line=dict(color=col_wave, width=2),
            fill="tozeroy",
            fillcolor="rgba(240,171,252,0.08)" if (pulse and pulse.active) else "rgba(0,0,0,0)",
            showlegend=False,
        ))
        fig.update_layout(
            height=72, margin=dict(l=0, r=0, t=2, b=2),
            paper_bgcolor="#0f1929", plot_bgcolor="#0f1929",
            xaxis=dict(visible=False), yaxis=dict(visible=False, range=[-0.22, 0.22]),
        )
        st.markdown(
            '<div style="background:#0f1929;border:1px solid #1e3a5f;'
            'border-radius:10px;padding:8px 12px;">'
            '<div style="font-size:.72rem;font-weight:600;letter-spacing:.1em;'
            'color:#f0abfc;text-transform:uppercase;margin-bottom:2px;">'
            '\u301c Olfaktif Dalgalanma</div>',
            unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True,
                        config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)


_live_panel()
