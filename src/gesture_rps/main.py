import os
import csv
import time
from collections import deque
from hashlib import sha256

import cv2
import numpy as np
import yaml
from loguru import logger

from gesture_rps.ui_overlay import (
    draw_countdown,
    draw_moves,
    draw_result_animated,
    draw_hint,
    draw_warning,
    draw_scoreboard,
    draw_hand_skeleton,
    draw_countdown_morph,
    apply_bloom,
    apply_gradient_backdrop,
    apply_vignette,
    draw_glow_trail,
    draw_result_spotlight,
    draw_start_screen,
    Theme,
)
from gesture_rps.gesture_detector import GestureDetector
from gesture_rps.ai_policy import RandomPolicy, AdaptiveFrequencyPolicy, MarkovPolicy
from gesture_rps.game_logic import adjudicate


def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def open_camera(width, height, index=0):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)  # CAP_DSHOW is stable on Windows
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise RuntimeError("Camera could not be opened. Check Privacy settings and device index.")
    return cap

def capture_burst(cap, n):
    frames = []
    for _ in range(max(1, n)):
        ok, frame = cap.read()
        if ok:
            frames.append(frame.copy())
    return frames

def ensure_outputs_dir(out_dir_name: str) -> str:
    # Project root is two levels up from this file ( .../src/gesture_rps/main.py )
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    out_dir = os.path.join(root, out_dir_name)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def get_logger_csv(out_dir: str) -> str:
    csv_path = os.path.join(out_dir, "round_log.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["ts", "player", "ai", "result", "conf", "difficulty", "proof"])
    return csv_path

def save_clip(out_dir: str, frames, label: str, reason: str = "", prefix: str = "clip"):
    try:
        ts = int(time.time() * 1000)
        clip_root = os.path.join(out_dir, "clips")
        os.makedirs(clip_root, exist_ok=True)
        safe_reason = (reason or "").replace(" ", "_").replace(":", "-")[:40]
        name = f"{prefix}_{ts}_{label}{('_' + safe_reason) if safe_reason else ''}"
        clip_dir = os.path.join(clip_root, name)
        os.makedirs(clip_dir, exist_ok=True)
        for i, f in enumerate(frames or []):
            cv2.imwrite(os.path.join(clip_dir, f"{i:02d}.jpg"), f)
    except Exception:
        pass

def log_round(csv_path, player, ai, result, conf, difficulty, proof):
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([f"{time.time():.3f}", player, ai, result, f"{conf:.3f}", difficulty, (proof or "")])

def make_policy(name: str):
    name = (name or "").lower()
    if name == "adaptive_freq":
        return AdaptiveFrequencyPolicy(), "adaptive_freq"
    if name == "markov":
        return MarkovPolicy(), "markov"
    return RandomPolicy(), "random"

def main():
    cfg = load_config()
    width = int(cfg["video"]["width"])
    height = int(cfg["video"]["height"])
    burst_frames = int(cfg["capture"]["burst_frames"])
    lock_at = int(cfg["fairness"]["lock_at_count"])
    out_dir = ensure_outputs_dir(cfg["logging"]["out_dir"])
    csv_log = get_logger_csv(out_dir)
    save_clips = bool(cfg["logging"].get("save_last_clip", False))

    # UI config
    ui_cfg = cfg.get("ui", {})
    font_path = ui_cfg.get("font_path", "")
    use_ttf = bool(ui_cfg.get("use_ttf", False))
    result_anim_secs = float(ui_cfg.get("result_anim_secs", 1.6))
    # themes
    themes_cfg = ui_cfg.get("themes", {})
    theme_names = list(themes_cfg.keys()) or ["neon"]
    active_idx = max(0, theme_names.index(ui_cfg.get("active_theme", "neon"))) if theme_names else 0
    def make_theme(idx):
        name = theme_names[idx % len(theme_names)]
        return name, Theme(themes_cfg.get(name, {}))
    theme_name, theme = make_theme(active_idx)
    # bloom
    bloom_cfg = ui_cfg.get("bloom", {})
    bloom_enabled = bool(bloom_cfg.get("enabled", True))
    bloom_threshold = int(bloom_cfg.get("threshold", 220))
    bloom_intensity = float(bloom_cfg.get("intensity", 0.55))
    bloom_ksize = int(bloom_cfg.get("blur_ksize", 17))
    morph_digits = bool(ui_cfg.get("morph_digits", True))

    # Visuals
    vis_cfg   = ui_cfg.get("visuals", {})
    grad_cfg  = ui_cfg.get("gradient", {})
    vig_cfg   = ui_cfg.get("vignette", {})
    glass_cfg = ui_cfg.get("glass", {})
    trail_cfg = ui_cfg.get("glow_trail", {})
    spot_cfg  = ui_cfg.get("spotlight", {})

    use_grad  = bool(vis_cfg.get("gradient_backdrop", True))
    use_vig   = bool(vis_cfg.get("vignette", True))
    use_glass = bool(vis_cfg.get("glass_panels", True))
    use_trail = bool(vis_cfg.get("glow_trail", True))
    use_spot  = bool(vis_cfg.get("result_spotlight", True))

    grad_phase = 0.0

    cap = open_camera(width, height)
    try:
        cv2.setUseOptimized(True)
        # Let OpenCV choose best threading; override if needed
        cv2.setNumThreads(0)
    except Exception:
        pass
    try:
        cv2.namedWindow("Gesture RPS", cv2.WINDOW_AUTOSIZE)
    except Exception:
        pass
    detector = GestureDetector(cfg)

    policy, difficulty = make_policy(cfg["ai"]["policy"])
    history = deque(maxlen=int(cfg["ai"]["history"]))
    fps_accum = deque(maxlen=20)

    wins = losses = ties = rounds = 0
    overlay_msg = ""
    overlay_until = 0.0

    # Hand trail for wrist (landmark 0)
    hand_trail = deque(maxlen=int(ui_cfg.get("glow_trail", {}).get("max_len", 24)))

    # States
    START_MENU, IDLE, COUNTDOWN, RESULT = 0, 1, 2, 3
    state = START_MENU

    # COUNTDOWN vars
    countdown_end = 0.0
    ai_locked = False
    locked_move = None
    proof = ""

    # RESULT vars
    result_started = 0.0
    show_player, show_ai, show_result, show_conf = "invalid", "rock", "invalid", 0.0

    try:
        while True:
            t0 = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                continue

            # Create a clean display canvas (hide background/face)
            display = np.zeros_like(frame)
            # Animated gradient underlay
            if use_grad:
                grad_phase += float(grad_cfg.get("speed", 0.08)) * 0.033
                display = apply_gradient_backdrop(
                    display,
                    grad_phase,
                    tuple(grad_cfg.get("color_a", [10,20,60])),
                    tuple(grad_cfg.get("color_b", [60,20,100])),
                    float(grad_cfg.get("alpha", 0.25)),
                )
            key = -1  # default until we read from waitKey in each state

            # Live hand skeleton (per-frame) drawn on the clean canvas
            pts = detector.get_hand_points_for_overlay(frame)
            if pts:
                hand_trail.append(pts[0])  # wrist
                draw_hand_skeleton(display, pts, hand_trail)
                if use_trail:
                    draw_glow_trail(display, hand_trail, trail_cfg, color=(0,200,255))
            else:
                hand_trail.clear()

            # Note: handle key presses after showing a frame to keep the window responsive

            # -------- START MENU --------
            if state == START_MENU:
                # simple start panel
                draw_start_screen(display, difficulty, font_path, use_ttf, theme)
                if cfg["ui"].get("show_scoreboard", True):
                    draw_scoreboard(display, 0, 0, 0, 0, difficulty, font_path, use_ttf, theme, use_glass, glass_cfg)
                shown = display
                if bloom_enabled:
                    shown = apply_bloom(shown, bloom_threshold, bloom_intensity, bloom_ksize)
                if use_vig:
                    shown = apply_vignette(shown, float(vig_cfg.get("strength", 0.45)), float(vig_cfg.get("softness", 0.35)))
                cv2.imshow("Gesture RPS", shown)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                if key in (ord('1'), ord('2'), ord('3')):
                    sel = {ord('1'): 'random', ord('2'): 'adaptive_freq', ord('3'): 'markov'}[key]
                    policy, difficulty = make_policy(sel)
                if key == ord('t'):
                    active_idx = (active_idx + 1) % max(1, len(theme_names))
                    theme_name, theme = make_theme(active_idx)
                if key == ord('b'):
                    bloom_enabled = not bloom_enabled
                if key == ord(' '):
                    state = IDLE

            # -------- IDLE --------
            elif state == IDLE:
                fps = (sum(fps_accum) / len(fps_accum)) if fps_accum else 0.0
                draw_hint(display, fps=fps, font_path=font_path, use_ttf=use_ttf, theme=theme, use_glass=use_glass, glass_cfg=glass_cfg)
                if cfg["ui"].get("show_scoreboard", True):
                    draw_scoreboard(display, wins, losses, ties, rounds, difficulty, font_path, use_ttf, theme, use_glass, glass_cfg)
                if overlay_msg and time.perf_counter() < overlay_until:
                    draw_warning(display, overlay_msg, theme)
                shown = display
                if bloom_enabled:
                    shown = apply_bloom(shown, bloom_threshold, bloom_intensity, bloom_ksize)
                if use_vig:
                    shown = apply_vignette(shown, float(vig_cfg.get("strength", 0.45)), float(vig_cfg.get("softness", 0.35)))
                cv2.imshow("Gesture RPS", shown)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key in (ord("1"), ord("2"), ord("3")):
                    sel = {ord("1"): "random", ord("2"): "adaptive_freq", ord("3"): "markov"}[key]
                    policy, difficulty = make_policy(sel)
                    overlay_msg = f"Difficulty: {difficulty}"
                    overlay_until = time.perf_counter() + 1.0
                if key == ord('t'):
                    active_idx = (active_idx + 1) % max(1, len(theme_names))
                    theme_name, theme = make_theme(active_idx)
                if key == ord('b'):
                    bloom_enabled = not bloom_enabled
                if key == ord(" "):
                    cv2.waitKey(1)  # consume key buffer
                    countdown_end = time.perf_counter() + 3.0
                    ai_locked, locked_move, proof = False, None, ""
                    state = COUNTDOWN

            # -------- COUNTDOWN --------
            elif state == COUNTDOWN:
                secs_left = int(countdown_end - time.perf_counter()) + 1
                if not ai_locked and secs_left <= lock_at:
                    ai_locked = True
                    locked_move = policy.choose(history)
                    payload = f"{time.time():.6f}|{locked_move}"
                    proof = sha256(payload.encode("utf-8")).hexdigest()[:10]

                if ui_cfg.get("morph_digits", True):
                    now = time.perf_counter()
                    total_rem = max(0.0, countdown_end - now)
                    secs_left_int = max(0, int(total_rem) + 1)
                    frac_in_second = 1.0 - (total_rem - int(total_rem))
                    current_txt = str(secs_left_int) if secs_left_int > 0 else "1"
                    next_txt = str(secs_left_int - 1) if secs_left_int > 1 else "Shoot!"
                    draw_countdown_morph(display, current_txt, next_txt, frac_in_second, font_path, use_ttf, theme)
                else:
                    draw_countdown(display, max(secs_left, 0), ai_locked, font_path, use_ttf, theme)
                shown = display
                if bloom_enabled:
                    shown = apply_bloom(shown, bloom_threshold, bloom_intensity, bloom_ksize)
                if use_vig:
                    shown = apply_vignette(shown, float(vig_cfg.get("strength", 0.45)), float(vig_cfg.get("softness", 0.35)))
                cv2.imshow("Gesture RPS", shown)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord('t'):
                    active_idx = (active_idx + 1) % max(1, len(theme_names))
                    theme_name, theme = make_theme(active_idx)
                if key == ord('b'):
                    bloom_enabled = not bloom_enabled

                if time.perf_counter() >= countdown_end:
                    frames = capture_burst(cap, burst_frames)
                    if not frames:
                        state = IDLE
                        continue

                    player_label, conf, reason = detector.predict_gesture(frames)
                    logger.info(f"Player: {player_label} (conf={conf:.2f}) reason='{reason}'")

                    if player_label == "invalid":
                        if save_clips:
                            save_clip(out_dir, frames, player_label, reason, prefix="invalid")
                        show_until = time.perf_counter() + 1.2
                        while time.perf_counter() < show_until:
                            ok2, frame2 = cap.read()
                            if not ok2:
                                continue
                            display2 = np.zeros_like(frame2)
                            pts2 = detector.get_hand_points_for_overlay(frame2)
                            if pts2:
                                hand_trail.append(pts2[0])
                                draw_hand_skeleton(display2, pts2, hand_trail)
                            draw_warning(display2, f"Redo: {reason}")
                            cv2.imshow("Gesture RPS", display2)
                            if cv2.waitKey(1) & 0xFF == ord("q"):
                                cap.release()
                                cv2.destroyAllWindows()
                                return
                        state = IDLE
                        continue

                    ai_label = locked_move if locked_move in {"rock", "paper", "scissors"} else policy.choose(history)
                    outcome = adjudicate(player_label, ai_label)

                    # Update history first, then let the policy learn (for Markov)
                    prev_len = len(history)
                    history.append(player_label)
                    if prev_len >= 1:
                        # now history[-2] -> history[-1] is a valid transition
                        try:
                            policy.learn(history)
                        except Exception:
                            pass

                    # Prepare RESULT state
                    show_player, show_ai, show_result, show_conf = player_label, ai_label, outcome, conf
                    result_started = time.perf_counter()
                    rounds += 1
                    if outcome == "win":
                        wins += 1
                    elif outcome == "lose":
                        losses += 1
                    elif outcome == "tie":
                        ties += 1

                    # Save clip and log round
                    if save_clips:
                        save_clip(out_dir, frames, player_label, "", prefix="round")
                    log_round(csv_log, player_label, ai_label, outcome, conf, difficulty, proof)

                    state = RESULT

            # -------- RESULT --------
            elif state == RESULT:
                # Animated result banner
                t_norm = (time.perf_counter() - result_started) / max(0.1, float(result_anim_secs))
                draw_moves(display, show_player, show_ai, font_path, use_ttf, theme)
                draw_result_animated(display, show_result, min(max(t_norm, 0.0), 1.0), font_path, use_ttf, theme)
                if use_spot:
                    display = draw_result_spotlight(display, min(max(t_norm, 0.0), 1.0), float(spot_cfg.get("width", 0.55)), float(spot_cfg.get("intensity", 0.35)))
                if cfg["ui"].get("show_scoreboard", True):
                    draw_scoreboard(display, wins, losses, ties, rounds, difficulty, font_path, use_ttf, theme, use_glass, glass_cfg)
                shown = display
                if bloom_enabled:
                    shown = apply_bloom(shown, bloom_threshold, bloom_intensity, bloom_ksize)
                if use_vig:
                    shown = apply_vignette(shown, float(vig_cfg.get("strength", 0.45)), float(vig_cfg.get("softness", 0.35)))
                cv2.imshow("Gesture RPS", shown)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord('t'):
                    active_idx = (active_idx + 1) % max(1, len(theme_names))
                    theme_name, theme = make_theme(active_idx)
                if key == ord('b'):
                    bloom_enabled = not bloom_enabled
                if t_norm >= 1.0:
                    state = IDLE

            # FPS estimate
            dt = time.perf_counter() - t0
            if dt > 0:
                fps_accum.append(1.0 / dt)

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
