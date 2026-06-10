import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import yt_dlp
import threading
import time
import torch
import sqlite3
import json
import os
import datetime
import pandas as pd

st.set_page_config(
    page_title="EOTRNet · Traffic Safety AI",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

INDONESIAN_DAYS = {
    "Monday": "Senin", "Tuesday": "Selasa", "Wednesday": "Rabu",
    "Thursday": "Kamis", "Friday": "Jumat", "Saturday": "Sabtu", "Sunday": "Minggu"
}

INDONESIAN_MONTHS = {
    1: "Januari", 2: "Februari", 3: "Maret", 4: "April", 5: "Mei", 6: "Juni",
    7: "Juli", 8: "Agustus", 9: "September", 10: "Oktober", 11: "November", 12: "Desember"
}

def format_indonesian_date(ts):
    day_name = INDONESIAN_DAYS[ts.strftime("%A")]
    day = ts.day
    month_name = INDONESIAN_MONTHS[ts.month]
    year = ts.year
    return f"{day_name}, {day} {month_name} {year}"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@400;500&display=swap');

/* ── Root variables ── */
:root {
    --bg:       #0a0d14;
    --surface:  #111520;
    --border:   #1e2535;
    --accent:   #00e5ff;
    --accent2:  #ff3e6c;
    --ok:       #00e676;
    --warn:     #ff9100;
    --text:     #e8eaf0;
    --muted:    #5a6480;
    --font:     'Syne', sans-serif;
    --mono:     'DM Mono', monospace;
}

/* ── Global resets ── */
html, body, [class*="css"] {
    font-family: var(--font) !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

/* ── Main container ── */
.main .block-container {
    padding: 1.5rem 2.5rem 3rem;
    max-width: 1400px;
}

/* ── App header ── */
.app-header {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 1.2rem 1.8rem;
    background: linear-gradient(135deg, #0d1220 0%, #111827 100%);
    border: 1px solid var(--border);
    border-left: 4px solid var(--accent);
    border-radius: 12px;
    margin-bottom: 1.5rem;
}
.app-header .logo { font-size: 2rem; }
.app-header h1 {
    margin: 0;
    font-size: 1.6rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    color: var(--text);
}
.app-header h1 span { color: var(--accent); }
.app-header .subtitle {
    margin: 0;
    font-size: 0.75rem;
    color: var(--muted);
    font-family: var(--mono) !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .block-container { padding: 1.5rem 1rem; }

.sidebar-section {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 1rem;
}
.sidebar-section h3 {
    margin: 0 0 0.75rem;
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--muted);
    font-family: var(--mono) !important;
}

/* ── Status badge ── */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.3rem 0.75rem;
    border-radius: 100px;
    font-family: var(--mono) !important;
    font-size: 0.72rem;
    font-weight: 500;
    margin-bottom: 1rem;
}
.badge-gpu  { background: rgba(0,230,118,0.1);  border: 1px solid rgba(0,230,118,0.3);  color: var(--ok);   }
.badge-cpu  { background: rgba(255,145,0,0.1);   border: 1px solid rgba(255,145,0,0.3);  color: var(--warn); }
.badge-live { background: rgba(0,229,255,0.1);   border: 1px solid rgba(0,229,255,0.3);  color: var(--accent); }

/* ── Metric cards ── */
.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
}
.metric-card.ok::before   { background: var(--ok);   }
.metric-card.alert::before { background: var(--accent2); }
.metric-card .value {
    font-size: 2.8rem;
    font-weight: 800;
    letter-spacing: -0.04em;
    line-height: 1;
    margin-bottom: 0.2rem;
}
.metric-card.ok    .value { color: var(--ok);    }
.metric-card.alert .value { color: var(--accent2); }
.metric-card .label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--muted);
    font-family: var(--mono) !important;
}
.metric-card .delta {
    font-size: 0.75rem;
    color: var(--muted);
    margin-top: 0.3rem;
    font-family: var(--mono) !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] [role="tablist"] {
    border-bottom: 1px solid var(--border) !important;
    gap: 0.25rem;
}
[data-testid="stTabs"] [role="tab"] {
    background: transparent !important;
    border: 1px solid transparent !important;
    border-radius: 8px 8px 0 0 !important;
    color: var(--muted) !important;
    font-family: var(--mono) !important;
    font-size: 0.8rem !important;
    padding: 0.5rem 1.2rem !important;
    transition: all 0.2s;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    background: var(--surface) !important;
    border-color: var(--border) !important;
    border-bottom-color: var(--surface) !important;
    color: var(--accent) !important;
}

/* ── Feed container ── */
.feed-wrapper {
    background: #000;
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
    position: relative;
}
.feed-topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.6rem 1rem;
    background: rgba(0,0,0,0.7);
    border-bottom: 1px solid var(--border);
}
.feed-topbar .area-label {
    font-family: var(--mono) !important;
    font-size: 0.75rem;
    color: var(--accent);
    font-weight: 500;
}
.feed-topbar .rec-dot {
    width: 8px; height: 8px;
    background: var(--accent2);
    border-radius: 50%;
    display: inline-block;
    margin-right: 0.4rem;
    animation: pulse 1.2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.3; }
}

/* ── Buttons ── */
.stButton > button {
    font-family: var(--mono) !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    border-radius: 8px !important;
    transition: all 0.2s !important;
}
.stButton > button[kind="primary"] {
    background: var(--accent) !important;
    color: #000 !important;
    border: none !important;
}
.stButton > button[kind="primary"]:hover {
    background: #00b8d4 !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 15px rgba(0,229,255,0.3) !important;
}
.stButton > button[kind="secondary"] {
    background: transparent !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
}
.stButton > button[kind="secondary"]:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
}

/* ── Inputs ── */
.stTextInput input, .stSelectbox select, [data-baseweb="select"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: var(--mono) !important;
    font-size: 0.82rem !important;
}
.stTextInput input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(0,229,255,0.15) !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    overflow: hidden;
}
[data-testid="stDataFrame"] table { font-family: var(--mono) !important; }

/* ── Alerts ── */
.stAlert {
    border-radius: 8px !important;
    font-family: var(--mono) !important;
    font-size: 0.8rem !important;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; margin: 1rem 0 !important; }

/* ── Expander ── */
[data-testid="stExpander"] {
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    background: var(--surface) !important;
}

/* ── Canvas label ── */
.canvas-hint {
    font-family: var(--mono) !important;
    font-size: 0.72rem;
    color: var(--muted);
    text-align: center;
    padding: 0.4rem;
    border-top: 1px solid var(--border);
}

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 3rem 2rem;
    color: var(--muted);
}
.empty-state .icon { font-size: 3rem; margin-bottom: 0.75rem; }
.empty-state h3    { font-size: 1rem; color: var(--text); margin-bottom: 0.4rem; }
.empty-state p     { font-size: 0.8rem; font-family: var(--mono) !important; }

/* ── Section header ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 1rem;
}
.section-header h3 {
    margin: 0;
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--muted);
    font-family: var(--mono) !important;
}
.section-header .line {
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* ── Violation rate badge ── */
.vrate {
    display: inline-block;
    padding: 0.15rem 0.5rem;
    border-radius: 4px;
    font-family: var(--mono) !important;
    font-size: 0.7rem;
    font-weight: 600;
}
.vrate-high { background: rgba(255,62,108,0.15); color: var(--accent2); }
.vrate-low  { background: rgba(0,230,118,0.15);  color: var(--ok); }

/* Hide Streamlit branding */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }

/* ── Settings tooltip (?) badge ── */
span[title] {
    cursor: help;
}
span[title]:hover::after {
    content: attr(title);
    position: fixed;
    background: #1a2035;
    border: 1px solid var(--accent);
    border-radius: 8px;
    padding: 0.5rem 0.75rem;
    font-size: 0.72rem;
    font-family: var(--mono) !important;
    color: var(--text);
    max-width: 240px;
    white-space: normal;
    z-index: 9999;
    box-shadow: 0 4px 20px rgba(0,0,0,0.6);
    line-height: 1.5;
    margin-left: 8px;
}
</style>
""", unsafe_allow_html=True)

DEVICE              = "cuda" if torch.cuda.is_available() else "cpu"
INFER_EVERY         = 3
DISPLAY_EVERY       = 3
DISPLAY_WIDTH       = 640
INFER_SIZE          = 360
DB_PATH             = "Data/TrafficSafetyDatabase.db"
CLIP_DIR            = "Data/Sample Video Clip"
MIN_CROSSING_DEPTH  = 20  

os.makedirs(CLIP_DIR, exist_ok=True)

def init_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS areas (
            name TEXT PRIMARY KEY,
            source TEXT,
            lines TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS surveillance_logs (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            cctv_location    TEXT,
            log_timestamp    DATETIME DEFAULT CURRENT_TIMESTAMP,
            correct_count    INTEGER,
            violator_count   INTEGER,
            video_filepath   TEXT
        )
    """)
    conn.commit()
    return conn

db_conn = init_db()

def get_youtube_stream_url(youtube_url: str):
    ydl_opts = {
        "format": "best[ext=mp4][height<=720]/best[height<=720]/best",
        "noplaylist": True, "quiet": True,
        "no_warnings": True, "live_from_start": False,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            url  = info.get("url") or info.get("manifest_url")
            if url:
                return url
            for fmt in reversed(info.get("requested_formats", [])):
                u = fmt.get("url")
                if u:
                    return u
        return None
    except Exception as e:
        st.error(f"YouTube error: {e}")
        return None

def open_capture(source: str):
    if source.isdigit():
        return cv2.VideoCapture(int(source))
    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap

def capture_screenshot(source: str):
    if "youtube.com" in source or "youtu.be" in source:
        source = get_youtube_stream_url(source)
        if not source:
            return None
    cap = open_capture(source)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return None

def draw_ui_elements(img, p1, p2, inverted=False):
    start, end = (p2, p1) if inverted else (p1, p2)
    dx, dy = end[0] - start[0], end[1] - start[1]
    dist   = np.sqrt(dx**2 + dy**2)
    if dist == 0:
        return img
    nx, ny  = -dy / dist, dx / dist
    mx, my  = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
    offset  = 50
    cor_end = (int(mx - nx * offset), int(my - ny * offset))
    wrg_end = (int(mx + nx * offset), int(my + ny * offset))
    cv2.line(img, p1, p2, (0, 255, 255), 4)
    cv2.arrowedLine(img, (int(mx + nx * offset), int(my + ny * offset)), cor_end, (255, 0, 0), 4, tipLength=0.3)
    cv2.arrowedLine(img, (int(mx - nx * offset), int(my - ny * offset)), wrg_end, (0, 255, 0), 4, tipLength=0.3)
    cv2.putText(img, "WRONG", cor_end, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.putText(img, "CORRECT",   wrg_end, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return img


def render_calibration_preview(image: Image.Image, lines: list) -> Image.Image:
    if not lines:
        return image
    img_np = np.array(image.convert("RGB"))
    h, w   = img_np.shape[:2]
    for idx, ln in enumerate(lines):
        p1 = (int(ln["p1"][0] * w), int(ln["p1"][1] * h))
        p2 = (int(ln["p2"][0] * w), int(ln["p2"][1] * h))
        img_np = draw_ui_elements(img_np, p1, p2, ln["inv"])
        mid    = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
        cv2.putText(img_np, f"Lane {idx+1}", (mid[0]+5, mid[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return Image.fromarray(img_np)

class ConfirmedLineZone:
    """
    Wraps sv.LineZone with a depth-confirmation buffer.

    A vehicle crossing the line is only counted once its centre has
    travelled at least `min_depth` pixels *past* the line on the new
    side.  This prevents false positives caused by vehicles that just
    graze the line or briefly oscillate across it.

    How it works
    ────────────
    For each tracked vehicle we store the signed perpendicular distance
    from the line on the last frame.  When the sign flips (= a crossing
    candidate), we record the candidate direction and start accumulating
    depth.  Only when the accumulated depth ≥ min_depth do we actually
    increment the in / out counter — and we lock that ID so it cannot
    be re-counted until it returns to the other side.
    """

    def __init__(self, sv_zone: sv.LineZone, min_depth: int = 20):
        self._zone      = sv_zone
        self.min_depth  = max(1, min_depth)


        self.in_count   = 0
        self.out_count  = 0


        self._state: dict = {}


        sx, sy = sv_zone.vector.start.x, sv_zone.vector.start.y
        ex, ey = sv_zone.vector.end.x,   sv_zone.vector.end.y
        dx, dy = ex - sx, ey - sy
        length = np.sqrt(dx * dx + dy * dy) or 1.0
        self._nx   = -dy / length
        self._ny   =  dx / length
        self._sx   = sx
        self._sy   = sy

    def _signed_distance(self, cx: float, cy: float) -> float:
        """Positive = 'in' side, negative = 'out' side."""
        return (cx - self._sx) * self._nx + (cy - self._sy) * self._ny

    def trigger(self, detections: sv.Detections):
        if detections.tracker_id is None or len(detections) == 0:
            return

        boxes = detections.xyxy         
        ids   = detections.tracker_id    

        for box, tid in zip(boxes, ids):
            cx = (box[0] + box[2]) / 2.0
            cy = (box[1] + box[3]) / 2.0
            dist = self._signed_distance(cx, cy)
            side = 1 if dist >= 0 else -1 

            if tid not in self._state:
                self._state[tid] = {"last_side": side, "candidate": None, "depth": 0.0}
                continue

            st = self._state[tid]

            if side == st["last_side"]:

                if st["candidate"] is not None:
                    st["depth"] += abs(dist)
                    if st["depth"] >= self.min_depth:
                        if st["candidate"] == "in":
                            self.in_count += 1
                        else:
                            self.out_count += 1
                        st["candidate"] = None
                        st["depth"]     = 0.0
            else:
          
                st["candidate"] = "in" if side == 1 else "out"
                st["depth"]     = abs(dist) 
                st["last_side"] = side


def build_pixel_zones(lines, f_w, f_h, min_depth: int = MIN_CROSSING_DEPTH):
    zones, coords = [], []
    for ln in lines:
        r1 = (int(ln["p1"][0] * f_w), int(ln["p1"][1] * f_h))
        r2 = (int(ln["p2"][0] * f_w), int(ln["p2"][1] * f_h))
        s, e = (r2, r1) if ln["inv"] else (r1, r2)
        sv_zone = sv.LineZone(
            start=sv.Point(*s), end=sv.Point(*e),
            triggering_anchors=[sv.Position.CENTER],
        )
        zones.append(ConfirmedLineZone(sv_zone, min_depth=min_depth))
        coords.append((r1, r2, ln["inv"]))
    return zones, coords


def get_360p_dims(w, h):
    th = 360
    tw = int(th * (w / h))
    return (tw if tw % 2 == 0 else tw + 1), th


def prepare_annotated_frame(frame, last_dets, box_ann, lbl_ann, coords, counts):
    img = frame.copy()
    if last_dets is not None and last_dets.tracker_id is not None and len(last_dets) > 0:
        img = box_ann.annotate(img, last_dets)
        img = lbl_ann.annotate(img, last_dets,
                               labels=[f"ID {tid}" for tid in last_dets.tracker_id])
    for r1, r2, inv in coords:
        img = draw_ui_elements(img, r1, r2, inv)
    h, w = img.shape[:2]
    cv2.putText(img, f"CORRECT: {counts['in']}",   (w - 250, 50),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    cv2.putText(img, f"VIOLATORS: {counts['out']}", (w - 250, 95),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    return img

def stream_worker(source: str, lines: list, model, ss: dict, area_name: str, settings: dict = None):
    if settings is None:
        settings = {
            "infer_every":       INFER_EVERY,
            "display_every":     DISPLAY_EVERY,
            "display_width":     DISPLAY_WIDTH,
            "infer_size":        INFER_SIZE,
            "device":            DEVICE,
            "min_crossing_depth": MIN_CROSSING_DEPTH,
        }

    _infer_every        = settings["infer_every"]
    _display_every      = settings["display_every"]
    _display_width      = settings["display_width"]
    _infer_size         = settings["infer_size"]
    _device             = settings["device"]
    _min_crossing_depth = settings.get("min_crossing_depth", MIN_CROSSING_DEPTH)

    cap = open_capture(source)
    if not cap.isOpened():
        ss["error"]   = "Could not open the video source."
        ss["running"] = False
        return

    tracker   = sv.ByteTrack()
    box_ann   = sv.BoxAnnotator()
    lbl_ann   = sv.LabelAnnotator()

    ret, first_frame = cap.read()
    if not ret:
        ss["error"]   = "Could not read the first frame."
        ss["running"] = False
        cap.release()
        return

    f_h, f_w       = first_frame.shape[:2]
    save_w, save_h = get_360p_dims(f_w, f_h)
    disp_h         = int(f_h * _display_width / f_w)
    zones, coords  = build_pixel_zones(lines, f_w, f_h, min_depth=_min_crossing_depth)

    local_conn = sqlite3.connect(DB_PATH, timeout=10)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = f"{CLIP_DIR}/{area_name.replace(' ', '_')}_{ts}.mp4"   
    
    out = cv2.VideoWriter(video_filename, fourcc, fps, (save_w, save_h))

    prev_in, prev_out = 0, 0

    frame_idx = 0
    last_dets = None
    frame     = first_frame

    while ss["running"]:
        annotated = prepare_annotated_frame(frame, last_dets, box_ann, lbl_ann, coords, ss["counts"])
        out.write(cv2.resize(annotated, (save_w, save_h)))

        if frame_idx % _infer_every == 0:
            res       = model(frame, imgsz=_infer_size, device=_device, verbose=False)[0]
            last_dets = sv.Detections.from_ultralytics(res)
            last_dets = tracker.update_with_detections(last_dets)
            if last_dets.tracker_id is not None and len(last_dets) > 0:
                for zone in zones:
                    zone.trigger(last_dets)

        ss["counts"] = {
            "in":  sum(z.in_count  for z in zones),
            "out": sum(z.out_count for z in zones),
        }

        if frame_idx % _display_every == 0:
            disp = annotated.copy()
            disp = cv2.resize(disp, (_display_width, disp_h))
            ss["frame"] = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)

        frame_idx += 1
        ret, frame = cap.read()
        if not ret:
            break

    out.release()

    h264_filename = video_filename.replace(".mp4", "_h264.mp4")
    os.system(f'ffmpeg -i "{video_filename}" -vcodec libx264 -acodec aac "{h264_filename}" -y -loglevel quiet')

    final_filename = h264_filename if os.path.exists(h264_filename) and os.path.getsize(h264_filename) > 0 else video_filename

    chunk_in  = ss["counts"]["in"]  - prev_in
    chunk_out = ss["counts"]["out"] - prev_out

    try:
        local_conn.execute(
            "INSERT INTO surveillance_logs (cctv_location, correct_count, violator_count, video_filepath) "
            "VALUES (?, ?, ?, ?)",
            (area_name, chunk_in, chunk_out, final_filename),
        )
        local_conn.commit()
        if final_filename == h264_filename and os.path.exists(video_filename):
            os.remove(video_filename)
    except Exception as e:
        print(f"Error logging final chunk: {e}")

    cap.release()
    local_conn.close()
    ss["running"] = False

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "Application Model", "model.pt")
model = YOLO(MODEL_PATH)

if "area_registry" not in st.session_state:
    st.session_state["area_registry"] = {}
    cur = db_conn.cursor()
    cur.execute("SELECT name, source, lines FROM areas")
    for row in cur.fetchall():
        st.session_state["area_registry"][row[0]] = {
            "source": row[1],
            "lines":  json.loads(row[2]),
            "image":  None,
            "counts": {"in": 0, "out": 0},
        }

if "stream_state" not in st.session_state:
    st.session_state["stream_state"] = {
        "running": False, "frame": None,
        "counts": {"in": 0, "out": 0}, "error": None,
    }

if "confirm_delete_area" not in st.session_state:
    st.session_state["confirm_delete_area"] = False

_INFER_SIZE_OPTIONS    = [160, 320, 480, 640]
_DISPLAY_WIDTH_OPTIONS = [320, 480, 640, 800, 960, 1280]

def _nearest(value, options):
    """Return the option closest to value."""
    return min(options, key=lambda x: abs(x - value))

if "tracking_settings" not in st.session_state:
    st.session_state["tracking_settings"] = {
        "infer_every":        INFER_EVERY,
        "display_every":      DISPLAY_EVERY,
        "display_width":      _nearest(DISPLAY_WIDTH, _DISPLAY_WIDTH_OPTIONS),
        "infer_size":         _nearest(INFER_SIZE,    _INFER_SIZE_OPTIONS),
        "device":             DEVICE,
        "min_crossing_depth": MIN_CROSSING_DEPTH,
    }

ss = st.session_state["stream_state"]

st.markdown("""
<div class="app-header">
    <div class="logo">🚦</div>
    <div>
        <h1><span>EOTR</span>Net · Traffic Safety AI</h1>
        <p class="subtitle">Smart Traffic Security &amp; Surveillance System</p>
    </div>
</div>
""", unsafe_allow_html=True)
with st.sidebar:
    # Device badge
    if DEVICE == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        st.markdown(f'<div class="status-badge badge-gpu">⬤ &nbsp;GPU · {gpu_name}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-badge badge-cpu">⬤ &nbsp;CPU mode — no GPU detected</div>', unsafe_allow_html=True)

    if ss["running"]:
        st.markdown(f'<div class="status-badge badge-live">⬤ &nbsp;LIVE · {st.session_state.get("active_area_name", "")}</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section"><h3>Area Registry</h3>', unsafe_allow_html=True)
    new_area_name = st.text_input("", placeholder="e.g. Ancol - Tj.Priok", label_visibility="collapsed")
    if st.button("＋  Register New Area", use_container_width=True):
        n = new_area_name.strip()
        if n and n not in st.session_state["area_registry"]:
            st.session_state["area_registry"][n] = {
                "lines": [], "source": "0", "image": None, "counts": {"in": 0, "out": 0}
            }
            db_conn.execute("INSERT INTO areas (name, source, lines) VALUES (?, ?, ?)", (n, "0", "[]"))
            db_conn.commit()
            st.success(f"Registered: {n}")
        elif not n:
            st.warning("Enter an area name.")
        else:
            st.warning("Area already exists.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section"><h3>Active Area</h3>', unsafe_allow_html=True)
    areas_list = list(st.session_state["area_registry"].keys())
    active_area = st.selectbox("", options=areas_list, label_visibility="collapsed",
                               key="active_area_select")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    def cfg_label(text, tooltip):
        return (
            f'{text} <span title="{tooltip}" style="cursor:help;color:var(--accent);'
            f'font-size:0.75rem;border:1px solid var(--accent);border-radius:50%;'
            f'padding:0 4px;font-weight:700;margin-left:4px">?</span>'
        )
    with st.expander("⚙️  Tracking Settings", expanded=False):
        ts = st.session_state["tracking_settings"]

        st.markdown(cfg_label("Device", "Which hardware accelerator runs the YOLO model. 'cuda' uses GPU (much faster), 'cpu' uses the processor only. Auto-detected at startup."), unsafe_allow_html=True)
        device_options = ["cpu"]
        if torch.cuda.is_available():
            device_options = ["cuda"] + device_options
        new_device = st.selectbox(
            "", options=device_options,
            index=device_options.index(ts["device"]) if ts["device"] in device_options else 0,
            key="cfg_device", label_visibility="collapsed"
        )

        st.markdown(cfg_label("Infer Every (frames)", "Run the detection model once every N frames. Higher = faster playback but may miss fast-moving vehicles. Lower = more accurate but higher CPU/GPU load. Recommended: 2–5."), unsafe_allow_html=True)
        new_infer_every = st.slider("", min_value=1, max_value=10, value=ts["infer_every"],
                                    key="cfg_infer_every", label_visibility="collapsed")

        st.markdown(cfg_label("Display Every (frames)", "Refresh the on-screen video feed every N frames. Does not affect detection accuracy. Higher = less UI lag. Recommended: 2–5."), unsafe_allow_html=True)
        new_display_every = st.slider("", min_value=1, max_value=10, value=ts["display_every"],
                                      key="cfg_display_every", label_visibility="collapsed")

        st.markdown(cfg_label("Display Width (px)", "Width (in pixels) of the video feed shown in the browser. Larger looks better but uses more memory and bandwidth. Recommended: 480–960."), unsafe_allow_html=True)
        new_display_width = st.select_slider("", options=_DISPLAY_WIDTH_OPTIONS,
                                             value=_nearest(ts["display_width"], _DISPLAY_WIDTH_OPTIONS),
                                             key="cfg_display_width", label_visibility="collapsed")

        st.markdown(cfg_label("Inference Size (px)", "The image resolution fed into the YOLO model for detection. Smaller = faster but less accurate on small/distant vehicles. Larger = more accurate but heavier. Recommended: 320–640."), unsafe_allow_html=True)
        new_infer_size = st.select_slider("", options=_INFER_SIZE_OPTIONS,
                                          value=_nearest(ts["infer_size"], _INFER_SIZE_OPTIONS),
                                          key="cfg_infer_size", label_visibility="collapsed")

        st.markdown(cfg_label(
            "Crossing Depth (px)",
            "How many pixels past the line a vehicle must travel before it is counted as a confirmed crossing. "
            "Higher = less sensitive, avoids flagging vehicles that merely graze or briefly oscillate across the line. "
            "Lower = more sensitive. Recommended: 15–40 px. Takes effect on next stream start."
        ), unsafe_allow_html=True)
        new_min_crossing_depth = st.slider(
            "", min_value=5, max_value=100,
            value=ts.get("min_crossing_depth", MIN_CROSSING_DEPTH),
            step=5,
            key="cfg_min_crossing_depth", label_visibility="collapsed"
        )

        st.markdown("<br>", unsafe_allow_html=True)
        col_apply, col_reset = st.columns(2)

        with col_apply:
            if st.button("✅  Apply", use_container_width=True, type="primary", key="cfg_apply"):
                st.session_state["tracking_settings"] = {
                    "device":             new_device,
                    "infer_every":        new_infer_every,
                    "display_every":      new_display_every,
                    "display_width":      new_display_width,
                    "infer_size":         new_infer_size,
                    "min_crossing_depth": new_min_crossing_depth,
                }
                if ss["running"]:
                    ss["display_width_live"] = new_display_width
                    ss["display_every_live"] = new_display_every
                st.success("Settings applied!")

        with col_reset:
            if st.button("↺  Reset", use_container_width=True, key="cfg_reset"):
                st.session_state["tracking_settings"] = {
                    "infer_every":        INFER_EVERY,
                    "display_every":      DISPLAY_EVERY,
                    "display_width":      _nearest(DISPLAY_WIDTH, _DISPLAY_WIDTH_OPTIONS),
                    "infer_size":         _nearest(INFER_SIZE,    _INFER_SIZE_OPTIONS),
                    "device":             DEVICE,
                    "min_crossing_depth": MIN_CROSSING_DEPTH,
                }
                st.rerun()

        if ss["running"]:
            st.caption("⚠️ Inference settings (Device, Infer Every, Inference Size, Crossing Depth) take effect on next stream start.")

    st.markdown("---")

    if active_area:
        if not st.session_state["confirm_delete_area"]:
            if st.button("🗑️  Delete Active Area", use_container_width=True):
                st.session_state["confirm_delete_area"] = True
                st.rerun()
        else:
            st.markdown(
                f'<div style="background:rgba(255,62,108,0.1);border:1px solid rgba(255,62,108,0.4);'
                f'border-radius:10px;padding:0.85rem 1rem;margin-bottom:0.75rem;">'
                f'<p style="margin:0 0 0.4rem;font-size:0.8rem;color:#ff3e6c;font-weight:700;">⚠️ Confirm Delete</p>'
                f'<p style="margin:0;font-size:0.75rem;color:var(--text);font-family:var(--mono);">'
                f'Delete <b>{active_area}</b>?<br>All logs and video evidence for this area will be permanently removed.</p>'
                f'</div>',
                unsafe_allow_html=True,
            )
            col_yes, col_no = st.columns(2)
            with col_yes:
                if st.button("✅ Yes, Delete", use_container_width=True, type="primary"):
                    video_rows = db_conn.execute(
                        "SELECT video_filepath FROM surveillance_logs WHERE cctv_location = ?",
                        (active_area,)
                    ).fetchall()
                    deleted_files = 0
                    for (vpath,) in video_rows:
                        if vpath and os.path.exists(vpath):
                            try:
                                os.remove(vpath)
                                deleted_files += 1
                            except Exception:
                                pass
                    db_conn.execute("DELETE FROM surveillance_logs WHERE cctv_location = ?", (active_area,))
                    db_conn.execute("DELETE FROM areas WHERE name = ?", (active_area,))
                    db_conn.commit()
                    del st.session_state["area_registry"][active_area]
                    st.session_state["confirm_delete_area"] = False
                    st.success(f"Deleted '{active_area}' — {deleted_files} video file(s) removed.")
                    st.rerun()
            with col_no:
                if st.button("✖ Cancel", use_container_width=True):
                    st.session_state["confirm_delete_area"] = False
                    st.rerun()

if not active_area:
    st.markdown("""
    <div class="empty-state">
        <div class="icon">📍</div>
        <h3>No Areas Registered</h3>
        <p>Use the sidebar to register your first monitoring area.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

curr_data = st.session_state["area_registry"][active_area]
st.session_state["active_area_name"] = active_area

tab1, tab2, tab3, tab4 = st.tabs([
    "🎯  Calibration",
    "📡  Live Monitor",
    "📊  Analytics",
    "📂  Tracking Logs",
])

with tab1:
    col_c, col_s = st.columns([3, 1], gap="large")

    with col_c:
        st.markdown('<div class="section-header"><h3>Area Image</h3><div class="line"></div></div>', unsafe_allow_html=True)

        uploaded = st.file_uploader(f"Upload reference image for **{active_area}**", type=["jpg", "png", "jpeg"])

        if st.button("📸  Grab Frame from Source", use_container_width=False):
            with st.spinner("Connecting to source…"):
                img = capture_screenshot(curr_data["source"])
            if img:
                curr_data["image"] = img
                st.success("Frame captured!")
            else:
                st.error("Could not capture frame — check the source URL.")

        if uploaded:
            curr_data["image"] = Image.open(uploaded).convert("RGB")

        image = curr_data.get("image")

        if image:
            display_image = render_calibration_preview(image, curr_data["lines"])
            max_w  = 800
            scale  = min(1.0, max_w / image.width)
            canv_w = int(image.width  * scale)
            canv_h = int(image.height * scale)

            st.markdown('<div class="section-header"><h3>Draw Lane Lines</h3><div class="line"></div></div>', unsafe_allow_html=True)
            canvas_result = st_canvas(
                stroke_width=3,
                stroke_color="#00FFFF",
                background_image=display_image.resize((canv_w, canv_h)),
                height=canv_h,
                width=canv_w,
                drawing_mode="line",
                key=f"canv_{active_area}",
            )
            st.markdown('<p class="canvas-hint">Draw cyan lines across each lane · Each line defines one wrong-way detection zone</p>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="empty-state">
                <div class="icon">🖼️</div>
                <h3>No image yet</h3>
                <p>Upload an image or grab a frame from the source.</p>
            </div>
            """, unsafe_allow_html=True)

    with col_s:
        st.markdown('<div class="section-header"><h3>Source Config</h3><div class="line"></div></div>', unsafe_allow_html=True)
        new_source = st.text_input(
            "Video Source",
            value=curr_data["source"],
            placeholder="YouTube URL / RTSP / 0",
            help="Enter a YouTube URL, RTSP stream URL, or a webcam index (0, 1, …)",
        )
        if new_source != curr_data["source"]:
            curr_data["source"] = new_source
            db_conn.execute("UPDATE areas SET source = ? WHERE name = ?", (new_source, active_area))
            db_conn.commit()

        st.markdown("---")

        if st.button("✅  Register Lanes", use_container_width=True, type="primary"):
            if image and canvas_result and canvas_result.json_data:
                added = 0
                for obj in canvas_result.json_data.get("objects", []):
                    if obj["type"] == "line":
                        p1 = ((obj["left"] + obj["x1"]) / canv_w, (obj["top"]  + obj["y1"]) / canv_h)
                        p2 = ((obj["left"] + obj["x2"]) / canv_w, (obj["top"]  + obj["y2"]) / canv_h)
                        curr_data["lines"].append({"p1": p1, "p2": p2, "inv": False})
                        added += 1
                if added:
                    db_conn.execute("UPDATE areas SET lines = ? WHERE name = ?",
                                    (json.dumps(curr_data["lines"]), active_area))
                    db_conn.commit()
                    st.success(f"Registered {added} lane(s).")
                    st.rerun()
                else:
                    st.warning("No lines drawn yet.")
            else:
                st.warning("Upload an image and draw lines first.")

        if curr_data["lines"]:
            st.markdown("---")
            st.markdown('<div class="section-header"><h3>Registered Lanes</h3><div class="line"></div></div>', unsafe_allow_html=True)
            for idx, line in enumerate(curr_data["lines"]):
                with st.expander(f"Lane {idx + 1}", expanded=False):
                    toggled = st.toggle("Flip Direction Logic",
                                        key=f"inv_{active_area}_{idx}",
                                        value=line["inv"])
                    if toggled != line["inv"]:
                        line["inv"] = toggled
                        db_conn.execute("UPDATE areas SET lines = ? WHERE name = ?",
                                        (json.dumps(curr_data["lines"]), active_area))
                        db_conn.commit()
                        st.rerun()
                    if st.button("🗑️ Delete Lane", key=f"del_{active_area}_{idx}", use_container_width=True):
                        curr_data["lines"].pop(idx)
                        db_conn.execute("UPDATE areas SET lines = ? WHERE name = ?",
                                        (json.dumps(curr_data["lines"]), active_area))
                        db_conn.commit()
                        st.rerun()

with tab2:
    if not curr_data["lines"]:
        st.markdown("""
        <div class="empty-state">
            <div class="icon">🎯</div>
            <h3>No lanes configured</h3>
            <p>Go to the Calibration tab and draw lane lines first.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        col_feed, col_panel = st.columns([3, 1], gap="large")

        with col_panel:
            st.markdown('<div class="section-header"><h3>Session Counts</h3><div class="line"></div></div>', unsafe_allow_html=True)

            c_in  = ss["counts"]["in"]
            c_out = ss["counts"]["out"]
            total = c_in + c_out
            v_rate = f"{(c_out / total * 100):.1f}%" if total > 0 else "0.0%"

            st.markdown(f"""
            <div class="metric-card ok" style="margin-bottom:0.75rem">
                <div class="value">{c_in}</div>
                <div class="label">Correct Way</div>
                <div class="delta">Vehicles travelling correctly</div>
            </div>
            <div class="metric-card alert" style="margin-bottom:0.75rem">
                <div class="value">{c_out}</div>
                <div class="label">Violators</div>
                <div class="delta">Wrong-way crossings detected</div>
            </div>
            """, unsafe_allow_html=True)

            rate_cls = "vrate-high" if (c_out / total > 0.1 if total else False) else "vrate-low"
            st.markdown(f'<p style="text-align:center;margin-top:0.5rem">Violation rate &nbsp;<span class="vrate {rate_cls}">{v_rate}</span></p>', unsafe_allow_html=True)

            st.markdown("---")
            st.markdown('<div class="section-header"><h3>Controls</h3><div class="line"></div></div>', unsafe_allow_html=True)

            start_clicked = st.button(
                "🚀  Start Stream",
                use_container_width=True,
                type="primary",
                disabled=ss["running"],
            )
            stop_clicked = st.button(
                "⏹  Stop Stream",
                use_container_width=True,
                disabled=not ss["running"],
            )

            if ss["running"]:
                st.markdown('<div class="status-badge badge-live" style="margin-top:0.75rem;justify-content:center">⬤ &nbsp;LIVE</div>', unsafe_allow_html=True)

        with col_feed:
            feed_placeholder  = st.empty()
            status_placeholder = st.empty()

            if ss["frame"] is not None:
                feed_placeholder.image(ss["frame"], channels="RGB", use_column_width=True)
            else:
                feed_placeholder.markdown("""
                <div class="empty-state" style="background:var(--surface);border:1px solid var(--border);border-radius:12px;min-height:360px;display:flex;flex-direction:column;align-items:center;justify-content:center">
                    <div class="icon">📡</div>
                    <h3>No feed active</h3>
                    <p>Press Start Stream to begin monitoring.</p>
                </div>
                """, unsafe_allow_html=True)

        if ss.get("error"):
            st.error(ss["error"])
            ss["error"] = None

        if stop_clicked:
            ss["running"] = False
            status_placeholder.warning("Stopping stream… finalising video chunk.")
            time.sleep(3)
            st.rerun()

        if start_clicked:
            source = curr_data["source"].strip()
            db_conn.execute("UPDATE areas SET source = ? WHERE name = ?", (source, active_area))
            db_conn.commit()

            if "youtube.com" in source or "youtu.be" in source:
                with st.spinner("Resolving YouTube stream URL…"):
                    resolved = get_youtube_stream_url(source)
                if not resolved:
                    st.error("Could not resolve YouTube URL.")
                    st.stop()
                source = resolved

            ss["running"] = True
            ss["frame"]   = None
            ss["counts"]  = {"in": 0, "out": 0}
            ss["error"]   = None
            curr_data["counts"] = {"in": 0, "out": 0}

            threading.Thread(
                target=stream_worker,
                args=(source, curr_data["lines"], model, ss, active_area),
                kwargs={"settings": dict(st.session_state["tracking_settings"])},
                daemon=True,
            ).start()
            status_placeholder.success("Stream started!")


        if ss["running"]:

            c_in  = ss["counts"]["in"]
            c_out = ss["counts"]["out"]
            total = c_in + c_out
            v_rate = f"{(c_out / total * 100):.1f}%" if total > 0 else "0.0%"
            rate_cls = "vrate-high" if (c_out / total > 0.1 if total else False) else "vrate-low"

            with col_panel:
                st.markdown(f"""
                <div class="metric-card ok" style="margin-bottom:0.75rem">
                    <div class="value">{c_in}</div>
                    <div class="label">Correct Way</div>
                </div>
                <div class="metric-card alert" style="margin-bottom:0.75rem">
                    <div class="value">{c_out}</div>
                    <div class="label">Violators</div>
                </div>
                <p style="text-align:center">Violation rate &nbsp;<span class="vrate {rate_cls}">{v_rate}</span></p>
                """, unsafe_allow_html=True)

            if ss["frame"] is not None:
                feed_placeholder.image(ss["frame"], channels="RGB", use_column_width=True)

            time.sleep(0.15)
            st.rerun()

with tab3:
    st.markdown('<div class="section-header"><h3>Analytics Dashboard</h3><div class="line"></div></div>', unsafe_allow_html=True)

    df_all = pd.read_sql_query(
        "SELECT * FROM surveillance_logs WHERE cctv_location = ? ORDER BY log_timestamp DESC",
        db_conn, params=(active_area,)
    )

    if df_all.empty:
        st.markdown("""
        <div class="empty-state">
            <div class="icon">📊</div>
            <h3>No data yet</h3>
            <p>Start monitoring to collect data for analytics.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        df_all["log_timestamp"] = pd.to_datetime(df_all["log_timestamp"]) + pd.Timedelta(hours=7)
        df_all["RawDate"]  = df_all["log_timestamp"].dt.date
        df_all["Hour"]  = df_all["log_timestamp"].dt.hour
        df_all["total"] = df_all["correct_count"] + df_all["violator_count"]
        df_all["vrate"] = (df_all["violator_count"] / df_all["total"].replace(0, 1) * 100).round(1)

        total_correct  = int(df_all["correct_count"].sum())
        total_violators = int(df_all["violator_count"].sum())
        total_all      = total_correct + total_violators
        avg_vrate      = f"{(total_violators / total_all * 100):.1f}%" if total_all else "—"
        peak_hour_row  = df_all.groupby("Hour")["violator_count"].sum()
        peak_hour      = int(peak_hour_row.idxmax()) if not peak_hour_row.empty else "—"

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Correct",    f"{total_correct:,}")
        k2.metric("Total Violators",  f"{total_violators:,}")
        k3.metric("Average Violation Rate", avg_vrate)
        k4.metric("Peak Violation Hour", f"{peak_hour}:00" if isinstance(peak_hour, int) else peak_hour)

        st.markdown("---")

        chart_col1, chart_col2 = st.columns(2)


        with chart_col1:
            st.markdown("**Daily Counts**")
            daily = df_all.groupby("RawDate")[["correct_count", "violator_count"]].sum().reset_index()

            daily["Date"] = pd.to_datetime(daily["RawDate"]).apply(format_indonesian_date)
            daily.columns = ["RawDate", "Correct", "Violators", "Date"]
            st.bar_chart(daily.set_index("Date")[["Correct", "Violators"]], color=["#00e676", "#ff3e6c"], use_container_width=True)


        with chart_col2:
            st.markdown("**Violations by Hour of Day**")
            hourly = df_all.groupby("Hour")["violator_count"].sum().reindex(range(24), fill_value=0).reset_index()
            hourly.columns = ["Hour", "Violators"]
            st.bar_chart(hourly.set_index("Hour"), color="#ff3e6c", use_container_width=True)

with tab4:
    st.markdown(f'<div class="section-header"><h3>Evidence Logs · {active_area}</h3><div class="line"></div></div>', unsafe_allow_html=True)

    df = pd.read_sql_query(
        "SELECT id, log_timestamp, correct_count, violator_count, video_filepath "
        "FROM surveillance_logs WHERE cctv_location = ? ORDER BY log_timestamp DESC",
        db_conn, params=(active_area,)
    )

    if df.empty:
        st.markdown("""
        <div class="empty-state">
            <div class="icon">📼</div>
            <h3>No recordings yet</h3>
            <p>Logs and video evidence will appear here once monitoring has run for at least 1 minute.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        df["log_timestamp"] = pd.to_datetime(df["log_timestamp"]) + pd.Timedelta(hours=7)

        df["Date"]  = df["log_timestamp"].apply(format_indonesian_date)
        df["Time"]  = df["log_timestamp"].dt.strftime("%H:%M:%S")
        df["total"] = df["correct_count"] + df["violator_count"]
        df["Rate"]  = (df["violator_count"] / df["total"].replace(0, 1) * 100).round(1).astype(str) + "%"

        display_df = df[["Date", "Time", "correct_count", "violator_count", "Rate"]].copy()
        display_df.columns = ["Date", "Time (WIB)", "✅ Correct", "🚨 Violators", "Violation Rate"]
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown('<div class="section-header"><h3>Evidence Playback</h3><div class="line"></div></div>', unsafe_allow_html=True)

        df["label"] = df["Date"] + "  ·  " + df["Time"] + "  —  " + df["violator_count"].astype(str) + " violator(s)"
        selected_label = st.selectbox("Select recording:", df["label"].tolist(), label_visibility="visible")

        if selected_label:
            row = df[df["label"] == selected_label].iloc[0]
            abs_path = os.path.abspath(row["video_filepath"])

            col_v, col_meta = st.columns([2, 1], gap="large")
            with col_v:
                if os.path.exists(abs_path):
                    st.video(abs_path)
                else:
                    st.error(f"Video not found: `{abs_path}`")

            with col_meta:
                st.markdown("**Recording details**")
                st.markdown(f"""
                | Field | Value |
                |---|---|
                | Date | {row['Date']} |
                | Time (WIB) | {row['Time']} |
                | ✅ Correct | {row['correct_count']} |
                | 🚨 Violators | {row['violator_count']} |
                | Rate | {row['Rate']} |
                """)
                st.caption(f"`{row['video_filepath']}`")

                if st.button("❌  Delete This Entry", type="primary", use_container_width=True):
                    if os.path.exists(abs_path):
                        os.remove(abs_path)
                    db_conn.execute("DELETE FROM surveillance_logs WHERE id = ?", (int(row["id"]),))
                    db_conn.commit()
                    st.success("Entry and video deleted.")
                    st.rerun()
