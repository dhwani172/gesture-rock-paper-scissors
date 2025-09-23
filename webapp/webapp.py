import gradio as gr
import numpy as np
import cv2
import yaml
import time

# import your modules
from src.gesture_rps.gesture_detector import GestureDetector
from src.gesture_rps.ui_overlay import (
    draw_moves, draw_result_animated, draw_countdown, draw_hand_skeleton
)
from src.gesture_rps.game_logic import adjudicate
from src.gesture_rps.ai_policy import RandomPolicy

# load config once
with open("src/gesture_rps/config.yaml", "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

detector = GestureDetector(CFG)
policy = RandomPolicy()

state = {
    "phase": "idle",           # idle | countdown | result
    "countdown_end": 0.0,
    "ai_locked": False,
    "locked_move": None,
    "result_started": 0.0,
    "show_player": "invalid",
    "show_ai": "rock",
    "show_result": "invalid",
}

def process(frame):
    """Gradio streams webcam frames here (numpy RGB). Return an annotated RGB frame."""
    if frame is None:
        return None
    # Gradio provides RGB; OpenCV expects BGR
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # live skeleton overlay
    pts = detector.get_hand_points_for_overlay(bgr)
    if pts:
        draw_hand_skeleton(bgr, pts, None)

    # simple state machine (tap the canvas to start)
    now = time.perf_counter()
    if state["phase"] == "idle":
        # show hint-ish countdown banner
        draw_countdown(bgr, 3, False, CFG["ui"].get("font_path"), CFG["ui"].get("use_ttf", True))
    elif state["phase"] == "countdown":
        secs_left = int(state["countdown_end"] - now) + 1
        if not state["ai_locked"] and secs_left <= CFG["fairness"]["lock_at_count"]:
            state["ai_locked"] = True
            state["locked_move"] = policy.choose([])
        draw_countdown(bgr, max(secs_left, 0), state["ai_locked"],
                       CFG["ui"].get("font_path"), CFG["ui"].get("use_ttf", True))
        if now >= state["countdown_end"]:
            # single-shot classify current frame (simpler than burst for web demo)
            player, conf, _reason = detector._predict_single(bgr)
            ai = state["locked_move"] or policy.choose([])
            outcome = adjudicate(player, ai)

            state.update({
                "phase": "result",
                "result_started": now,
                "show_player": player,
                "show_ai": ai,
                "show_result": outcome,
            })
    elif state["phase"] == "result":
        t_norm = (now - state["result_started"]) / CFG["ui"].get("result_anim_secs", 1.2)
        draw_moves(bgr, state["show_player"], state["show_ai"], CFG["ui"].get("font_path"), CFG["ui"].get("use_ttf", True))
        draw_result_animated(bgr, state["show_result"], max(0.0, min(1.0, t_norm)),
                             CFG["ui"].get("font_path"), CFG["ui"].get("use_ttf", True))
        if t_norm >= 1.0:
            state.update({"phase": "idle", "ai_locked": False, "locked_move": None})

    # return RGB for browser
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def start_round():
    state.update({
        "phase": "countdown",
        "countdown_end": time.perf_counter() + 3.0,
        "ai_locked": False,
        "locked_move": None,
    })
    return "Round started!"

with gr.Blocks() as demo:
    gr.Markdown("## Gesture RPS — webcam demo (press **Start**)")
    with gr.Row():
        cam = gr.Image(sources=["webcam"], streaming=True, label="Webcam")
        out = gr.Image(label="Game")
    btn = gr.Button("Start / Restart")
    btn.click(fn=lambda: start_round(), outputs=[])
    cam.stream(process, inputs=cam, outputs=out)

if __name__ == "__main__":
    demo.launch()
