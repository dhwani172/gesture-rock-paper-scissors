import os
import io
import base64
import time
import uuid
from collections import deque
from typing import List, Optional, Dict, Tuple

import cv2
import numpy as np
import yaml
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from gesture_rps.gesture_detector import GestureDetector
from gesture_rps.ai_policy import RandomPolicy, AdaptiveFrequencyPolicy, MarkovPolicy
from gesture_rps.game_logic import adjudicate


# ---------------- Config helpers ----------------
def _load_config() -> dict:
    here = os.path.dirname(__file__)
    cfg_path = os.path.join(here, "config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _ensure_outputs_dir(out_dir_name: str) -> str:
    # Project root is two levels up from this file ( .../src/gesture_rps/web_server.py )
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    out_dir = os.path.join(root, out_dir_name)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _get_csv_logger_path(out_dir: str) -> str:
    csv_path = os.path.join(out_dir, "round_log.csv")
    if not os.path.exists(csv_path):
        import csv

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["ts", "player", "ai", "result", "conf", "difficulty", "proof"])
    return csv_path


def _log_round(csv_path: str, player: str, ai: str, result: str, conf: float, difficulty: str, proof: str):
    import csv

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([f"{time.time():.3f}", player, ai, result, f"{conf:.3f}", difficulty, (proof or "")])


def _make_policy(name: str):
    name = (name or "").lower()
    if name == "adaptive_freq":
        return AdaptiveFrequencyPolicy(), "adaptive_freq"
    if name == "markov":
        return MarkovPolicy(), "markov"
    return RandomPolicy(), "random"


# ---------------- Session storage ----------------
class SessionState:
    def __init__(self, difficulty: str):
        policy, name = _make_policy(difficulty)
        self.policy = policy
        self.difficulty = name
        self.history: deque[str] = deque(maxlen=100)


_sessions: Dict[str, SessionState] = {}
_SID_COOKIE = "sid"


def _get_or_create_session(req: Request, resp: Response, default_difficulty: str) -> Tuple[str, SessionState]:
    sid = req.cookies.get(_SID_COOKIE)
    if not sid or sid not in _sessions:
        sid = uuid.uuid4().hex
        _sessions[sid] = SessionState(default_difficulty)
        resp.set_cookie(_SID_COOKIE, sid, httponly=False, samesite="lax")
    return sid, _sessions[sid]


# ---------------- Request/Response models ----------------
class RoundRequest(BaseModel):
    images: List[str]
    difficulty: Optional[str] = None


class RoundResponse(BaseModel):
    player: str
    conf: float
    reason: str
    ai: str
    result: str
    difficulty: str

class OverlayReq(BaseModel):
    image: str  # base64 data URL or raw base64


# ---------------- App init ----------------
cfg = _load_config()
detector = GestureDetector(cfg)
app = FastAPI(title="Gesture RPS Web")


# Serve static UI from package directory
_static_dir = os.path.join(os.path.dirname(__file__), "web", "static")



@app.get("/api/config")
def api_config():
    ui_cfg = cfg.get("ui", {})
    cap_cfg = cfg.get("capture", {})
    fairness = cfg.get("fairness", {})
    return {
        "burst_frames": int(cap_cfg.get("burst_frames", 11)),
        "window_ms": int(cap_cfg.get("window_ms", 450)),
        "lock_at_count": int(fairness.get("lock_at_count", 2)),
        "difficulties": ["random", "adaptive_freq", "markov"],
        "default_difficulty": str(cfg.get("ai", {}).get("policy", "adaptive_freq")),
    }


def _decode_b64_images(images: List[str]) -> List[np.ndarray]:
    frames = []
    for s in images:
        if "," in s and s.strip().startswith("data:"):
            s = s.split(",", 1)[1]
        try:
            raw = base64.b64decode(s)
            arr = np.frombuffer(raw, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                frames.append(img)
        except Exception:
            continue
    return frames


@app.post("/api/round", response_model=RoundResponse)
def api_round(req: Request, rr: RoundRequest):
    resp = Response(media_type="application/json")
    sid, sess = _get_or_create_session(req, resp, str(cfg.get("ai", {}).get("policy", "adaptive_freq")))

    # Update difficulty/policy per request if provided
    if rr.difficulty:
        policy, name = _make_policy(rr.difficulty)
        sess.policy = policy
        sess.difficulty = name

    frames = _decode_b64_images(rr.images or [])
    if not frames:
        raise HTTPException(status_code=400, detail="No valid frames provided")

    player_label, conf, reason = detector.predict_gesture(frames)
    ai_label = "rock"
    result = "invalid"

    if not reason and player_label in {"rock", "paper", "scissors"}:
        ai_label = sess.policy.choose(sess.history)
        result = adjudicate(player_label, ai_label)
        # learn from the new transition
        prev_len = len(sess.history)
        sess.history.append(player_label)
        if prev_len >= 1:
            try:
                sess.policy.learn(sess.history)
            except Exception:
                pass
        # optional logging
        out_dir = _ensure_outputs_dir(str(cfg.get("logging", {}).get("out_dir", "outputs")))
        csv_path = _get_csv_logger_path(out_dir)
        _log_round(csv_path, player_label, ai_label, result, float(conf), sess.difficulty, proof="")
    else:
        ai_label = "rock"
        result = "invalid"

    payload = {
        "player": player_label,
        "conf": float(conf),
        "reason": reason,
        "ai": ai_label,
        "result": result,
        "difficulty": sess.difficulty,
    }
    return JSONResponse(content=payload)


# Simple health check
@app.get("/health")
def health():
    return {"ok": True}

# Mount static last, and serve index at root
app.mount("/static", StaticFiles(directory=_static_dir), name="static")

@app.get("/")
def index_page():
    return FileResponse(os.path.join(_static_dir, "index.html"))


@app.post("/api/overlay_points")
def overlay_points(req: OverlayReq):
    # decode
    s = req.image
    if "," in s and s.strip().startswith("data:"):
        s = s.split(",", 1)[1]
    try:
        raw = base64.b64decode(s)
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        raise HTTPException(status_code=400, detail="Bad image")
    if img is None:
        raise HTTPException(status_code=400, detail="Bad image")

    # get points (pixel coords in img)
    pts = detector.get_hand_points_for_overlay(img)
    if not pts:
        return {"points": []}
    h, w = img.shape[:2]
    norm = [[float(x)/max(1.0,w), float(y)/max(1.0,h)] for (x,y) in pts]
    return {"points": norm}
