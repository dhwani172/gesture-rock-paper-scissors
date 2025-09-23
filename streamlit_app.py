import sys
import os
# Make "src/" importable on Streamlit Cloud and locally
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import av
import cv2
import time
import yaml
import numpy as np
import streamlit as st
from pathlib import Path
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

# ──────────────────────────────────────────────────────────────────────────────
# Page setup + CSS injection
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Gesture RPS", layout="wide")

def inject_css(path_str: str):
    """Inject local CSS file into Streamlit."""
    p = Path(path_str)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

def inject_theme(theme: str):
    """Inject CSS vars for light/dark theme at runtime."""
    if theme == "dark":
        st.markdown(
            """
            <style>
            :root {
              --bg: #0d1117;
              --fg: #e6edf3;
              --muted: #95a1b2;
              --accent: #58a6ff;
              --accent-2: #7ee787;
              --accent-warm: #c297ff;

              --card: rgba(255,255,255,0.06);
              --border: rgba(240,246,252,0.12);
              --shadow: 0 16px 40px rgba(0,0,0,.45);

              --glass: rgba(255,255,255,.06);
              --glass-border: rgba(255,255,255,.12);
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <style>
            :root {
              --bg: #ffffff;
              --fg: #3B6255;
              --muted: #8BA49A;
              --accent: #3B6255;
              --accent-2: #D2C49E;
              --accent-warm: #739882;

              --card: rgba(255, 255, 255, 0.92);
              --border: rgba(59, 98, 85, 0.18);
              --shadow: 0 10px 30px rgba(59, 98, 85, 0.12);

              --glass: rgba(255,255,255,.72);
              --glass-border: rgba(59,98,85,.18);
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

# Inject base CSS
inject_css("src/gesture_rps/web/static/styles.css")

# OpenCV perf hints
cv2.setUseOptimized(True)
try:
    cv2.setNumThreads(2)
except Exception:
    pass

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar: Theme switcher + Tips
# ──────────────────────────────────────────────────────────────────────────────
if "theme" not in st.session_state:
    st.session_state.theme = "light"

with st.sidebar:
    st.selectbox(
        "Theme",
        ["light", "dark"],
        key="theme",
        help="Switch between light and dark themes.",
    )
inject_theme(st.session_state.theme)

with st.sidebar.expander("📘 Tips", True):
    st.markdown(
        """
        ### How to Play
        - Allow webcam access.  
        - Keep one hand in view.  
        - On countdown end, show ✊, ✋, or ✌.  
        - AI locks its move near the last second.  
        - Results and score update after each round.  
        """
    )

# ──────────────────────────────────────────────────────────────────────────────
# Landing screen
# ──────────────────────────────────────────────────────────────────────────────
if "started" not in st.session_state:
    st.session_state.started = False

if not st.session_state.started:
    st.markdown(
        """
        <div class="container page float-in">
          <div class="hero">
            <h1 class="logo shimmer"><span>Gesture RPS</span></h1>
            <div class="status">Webcam gesture game</div>
          </div>
          <div class="card glass raise appear" style="padding:22px;text-align:center;">
            <p style="margin:0 0 14px 0;color:var(--muted)">
              Play Rock–Paper–Scissors with your hand gestures.
            </p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns([1,1,1])
    with c2:
        if st.button("Start Game", use_container_width=True):
            st.session_state.started = True
            st.rerun()
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# Imports from your package + config
# ──────────────────────────────────────────────────────────────────────────────
from src.gesture_rps.gesture_detector import GestureDetector
from src.gesture_rps.ui_overlay import draw_hand_skeleton
from src.gesture_rps.game_logic import adjudicate
from src.gesture_rps.ai_policy import RandomPolicy, AdaptiveFrequencyPolicy, MarkovPolicy

with open("src/gesture_rps/config.yaml", "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

policy_map = {
    "random": RandomPolicy,
    "adaptive_freq": AdaptiveFrequencyPolicy,
    "markov": MarkovPolicy,
}

@st.cache_resource
def get_detector(cfg):
    return GestureDetector(cfg)

detector = get_detector(CFG)

# ──────────────────────────────────────────────────────────────────────────────
# App state
# ──────────────────────────────────────────────────────────────────────────────
if "game_state" not in st.session_state:
    st.session_state.game_state = {
        "phase": "idle",
        "countdown_end": 0.0,
        "ai_locked": False,
        "locked_move": None,
        "result_started": 0.0,
        "show_player": "invalid",
        "show_ai": "rock",
        "show_result": "—",
        "wins": 0,
        "losses": 0,
        "ties": 0,
    }
gs = st.session_state.game_state

# ──────────────────────────────────────────────────────────────────────────────
# Layout
# ──────────────────────────────────────────────────────────────────────────────
left, right = st.columns([3, 2], gap="large")

with right:
    st.subheader("Difficulty")
    difficulty = st.selectbox(
        "AI Strategy",
        ["random", "adaptive_freq", "markov"],
        index=0,
        help="Choose how the AI picks its moves.",
    )

    info = {
        "random": ("Random",
            "AI plays uniformly at random. Great for testing camera & gestures."),
        "adaptive_freq": ("Adaptive Frequency",
            "AI tracks your recent moves and plays what is most likely to beat your habits."),
        "markov": ("Markov (Predictive)",
            "AI models transition patterns between your moves to anticipate your next gesture."),
    }
    title, desc = info[difficulty]
    st.markdown(
        f"""
        <div class="card glass raise appear" style="padding:18px">
          <h4 style="margin:.25rem 0">{title}</h4>
          <div style="opacity:.9">{desc}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with left:
    st.markdown("### Gesture Rock–Paper–Scissors")

# ──────────────────────────────────────────────────────────────────────────────
# Video transformer (black canvas + skeleton only)
# ──────────────────────────────────────────────────────────────────────────────
class RPSVideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.policy = policy_map.get(difficulty, RandomPolicy)()
        self.stride = 3
        self.downscale = 0.4
        self.frame_idx = 0
        self.last_pts = None
        self.last_label = "invalid"

    def _detect(self, bgr):
        if self.downscale != 1.0:
            small = cv2.resize(bgr, None, fx=self.downscale, fy=self.downscale,
                               interpolation=cv2.INTER_AREA)
            scale = 1.0 / self.downscale
        else:
            small = bgr
            scale = 1.0

        pts = detector.get_hand_points_for_overlay(small)
        if pts:
            pts = [(int(x*scale), int(y*scale)) for (x, y) in pts]
        self.last_pts = pts

        label, conf, _ = detector._predict_single(small)
        self.last_label = label

    def recv(self, frame: av.VideoFrame):
        orig = frame.to_ndarray(format="bgr24")
        h, w = orig.shape[:2]
        img = np.zeros((h, w, 3), dtype=np.uint8)
        now = time.perf_counter()

        if self.frame_idx % self.stride == 0:
            self._detect(orig)
        self.frame_idx += 1

        if self.last_pts:
            draw_hand_skeleton(img, self.last_pts, None)
            wrist = self.last_pts[0]
            cv2.circle(img, wrist, 16, (0, 255, 255), 2)

        if gs["phase"] == "countdown":
            if not gs["ai_locked"]:
                secs_left = int(gs["countdown_end"] - now) + 1
                if secs_left <= max(1, int(CFG["fairness"]["lock_at_count"])):
                    pass  # TODO: Add logic here for when the AI should lock its move
