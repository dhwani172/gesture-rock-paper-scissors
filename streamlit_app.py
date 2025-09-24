# ──────────────────────────────────────────────────────────────────────────────
# Path setup + page config (must be first Streamlit command)
# ──────────────────────────────────────────────────────────────────────────────
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))  # so "gesture_rps" works both locally & on Cloud

import streamlit as st  # noqa: E402

st.set_page_config(
    page_title="Gesture RPS",
    page_icon="✌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Deps
# ──────────────────────────────────────────────────────────────────────────────
import av  # noqa: E402
import cv2  # noqa: E402
import time  # noqa: E402
import yaml  # noqa: E402
import numpy as np  # noqa: E402
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode  # noqa: E402

# ──────────────────────────────────────────────────────────────────────────────
# CSS (force a light base so Cloud themes don’t invert)
# ──────────────────────────────────────────────────────────────────────────────
def local_css(file_name: str):
    try:
        with open(file_name, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception:
        pass

st.markdown(
    """
    <style>
      :root { --bg:#fff !important; --fg:#3B6255 !important; --muted:#8BA49A !important; }
      .stApp { background:var(--bg)!important; color:var(--fg)!important; }
    </style>
    """,
    unsafe_allow_html=True,
)

css_path = Path("src/gesture_rps/web/static/styles.css")
if css_path.exists():
    local_css(css_path)

# OpenCV perf tweaks
cv2.setUseOptimized(True)
try:
    cv2.setNumThreads(2)
except Exception:
    pass

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
            <h1 class="logo"><span>Gesture RPS</span></h1>
            <div class="status">Webcam gesture game</div>
          </div>
          <div class="card" style="padding:22px;text-align:center;">
            <p style="margin:0 0 14px 0;color:var(--muted)">
              Play Rock–Paper–Scissors with your hand gestures.
            </p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns([1, 1, 1])
    with c2:
        if st.button("Start Game", use_container_width=True):
            st.session_state.started = True
            st.rerun()
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# Local package imports + config (try both import styles)
# ──────────────────────────────────────────────────────────────────────────────
try:
    from gesture_rps.gesture_detector import GestureDetector  # noqa: E402
    from gesture_rps.ui_overlay import draw_hand_skeleton  # noqa: E402
    from gesture_rps.game_logic import adjudicate  # noqa: E402
    from gesture_rps.ai_policy import RandomPolicy, AdaptiveFrequencyPolicy, MarkovPolicy  # noqa: E402
except Exception:
    from src.gesture_rps.gesture_detector import GestureDetector  # noqa: E402
    from src.gesture_rps.ui_overlay import draw_hand_skeleton  # noqa: E402
    from src.gesture_rps.game_logic import adjudicate  # noqa: E402
    from src.gesture_rps.ai_policy import RandomPolicy, AdaptiveFrequencyPolicy, MarkovPolicy  # noqa: E402

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
# Sidebar: connection presets + tips
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("Connection")
    preset = st.selectbox(
        "Webcam preset",
        ["Minimal (most compatible)", "Balanced (720p)", "HD (may fail on Cloud)"],
        index=0,
        help="If the camera won't start, try 'Minimal'.",
    )
    st.caption("If your network blocks P2P, TURN helps. We use public OpenRelay TURN.")

with st.sidebar.expander("📘 Tips", True):
    st.markdown(
        """
        ### How to Play
        - Allow webcam access (browser prompt).  
        - Keep one hand in view.  
        - On countdown end, show ✊, ✋, or ✌.  
        - AI locks its move near the last second.  
        - Results and score update after each round.  
        """
    )

# Build constraints from preset
if preset.startswith("Minimal"):
    MEDIA_CONSTRAINTS = {"video": True, "audio": False}
elif preset.startswith("Balanced"):
    MEDIA_CONSTRAINTS = {
        "video": {
            "width": {"ideal": 1280},
            "height": {"ideal": 720},
            "frameRate": {"ideal": 24, "max": 30},
            "facingMode": "user",
        },
        "audio": False,
    }
else:  # HD
    MEDIA_CONSTRAINTS = {
        "video": {
            "width": {"ideal": 1920},
            "height": {"ideal": 1080},
            "frameRate": {"ideal": 30, "max": 30},
            "facingMode": "user",
        },
        "audio": False,
    }

# Robust ICE configuration: Google STUN + OpenRelay TURN
RTC_CONFIG = {
    "iceServers": [
        {"urls": [
            "stun:stun.l.google.com:19302",
            "stun:stun1.l.google.com:19302",
            "stun:stun2.l.google.com:19302",
        ]},
        {
            "urls": [
                "turn:openrelay.metered.ca:80",
                "turn:openrelay.metered.ca:443",
                "turns:openrelay.metered.ca:443?transport=tcp",
            ],
            "username": "openrelayproject",
            "credential": "openrelayproject",
        },
    ]
}

# ──────────────────────────────────────────────────────────────────────────────
# Layout (main)
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
        <div class="card raise appear" style="padding:18px">
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
        self.stride = 3          # detect every N frames
        self.downscale = 0.4     # detect at 40% size
        self.frame_idx = 0
        self.last_pts = None
        self.last_label = "invalid"

    def _detect(self, bgr):
        if self.downscale != 1.0:
            small = cv2.resize(bgr, None, fx=self.downscale, fy=self.downscale, interpolation=cv2.INTER_AREA)
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
                    gs["ai_locked"] = True
                    gs["locked_move"] = self.policy.choose([])

            if now >= gs["countdown_end"]:
                label = self.last_label
                ai = gs["locked_move"] or self.policy.choose([])
                outcome = adjudicate(label, ai)
                gs.update({
                    "phase": "result",
                    "result_started": now,
                    "show_player": label,
                    "show_ai": ai,
                    "show_result": outcome,
                })
                if outcome == "win": gs["wins"] += 1
                elif outcome == "lose": gs["losses"] += 1
                else: gs["ties"] += 1

        elif gs["phase"] == "result":
            if (now - gs["result_started"]) >= float(CFG["ui"].get("result_anim_secs", 1.2)):
                gs.update({"phase": "idle", "ai_locked": False, "locked_move": None})

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ──────────────────────────────────────────────────────────────────────────────
# Player + Start Round + Connection status
# ──────────────────────────────────────────────────────────────────────────────
with left:
    webrtc_ctx = webrtc_streamer(
        key="gesture-rps",
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=RPSVideoTransformer,
        media_stream_constraints=MEDIA_CONSTRAINTS,    # preset-based
        rtc_configuration=RTC_CONFIG,                  # STUN + TURN
        async_processing=True,
        video_html_attrs={
            "autoPlay": True,
            "muted": True,
            "playsinline": True,
            "style": {
                "width": "100%",
                "height": "360px",
                "borderRadius": "16px",
                "backgroundColor": "#000"
            },
        },
    )

    # Show WebRTC state to help diagnose
    st.caption(f"WebRTC status: {'playing' if (webrtc_ctx and webrtc_ctx.state.playing) else 'not started'}")

    if st.button("Start Round", use_container_width=True, disabled=not (webrtc_ctx and webrtc_ctx.state.playing)):
        gs.update({
            "phase": "countdown",
            "countdown_end": time.perf_counter() + 3.0,
            "ai_locked": False,
            "locked_move": None,
            "show_result": "—",
        })
        status_box = st.empty()
        for n in [3, 2, 1]:
            status_box.markdown(
                f"""
                <div class="card appear" style="padding:14px">
                  <div class="label">Countdown</div>
                  <div style="font-size:28px;font-weight:800">{n}</div>
                </div>
                """, unsafe_allow_html=True)
            time.sleep(1)
        status_box.markdown(
            """
            <div class="card appear" style="padding:14px">
              <div class="label">Countdown</div>
              <div style="font-size:24px;font-weight:700">Shoot!</div>
            </div>
            """, unsafe_allow_html=True)

        t0 = time.time()
        while time.time() - t0 < 2.0 and gs["phase"] != "result":
            time.sleep(0.05)

        if gs["phase"] == "result":
            color = {"win": "v win", "lose": "v lose", "tie": "v tie"}.get(gs["show_result"], "v")
            status_box.markdown(
                f"""
                <div class="card appear" style="padding:14px">
                  <div class="label">Result</div>
                  <div class="{color}" style="font-size:26px">
                    {gs["show_result"].upper()}
                  </div>
                  <div class="last">You: <strong>{gs["show_player"]}</strong>
                    <span class="arrow">→</span> AI: <strong>{gs["show_ai"]}</strong>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ──────────────────────────────────────────────────────────────────────────────
# Scoreboard (right, under difficulty)
# ──────────────────────────────────────────────────────────────────────────────
with right:
    st.markdown('<div class="scoreboard card appear" style="padding:14px;margin-top:12px;">', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"<div class='metric'><div class='k'>Wins</div><div class='v win'>{gs['wins']}</div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric'><div class='k'>Losses</div><div class='v lose'>{gs['losses']}</div></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='metric'><div class='k'>Ties</div><div class='v tie'>{gs['ties']}</div></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='metric'><div class='k'>Last</div><div class='v'>{gs['show_result']}</div></div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
