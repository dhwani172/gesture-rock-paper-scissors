import cv2
import numpy as np
import math
from typing import List, Tuple, Deque, Optional

from gesture_rps.utils.facy_text import (
    draw_ttf_center, draw_ttf, draw_ttf_right, add_translucent_bar
)

FONT = cv2.FONT_HERSHEY_SIMPLEX

# Theme helper
class Theme:
    def __init__(self, d):
        self.panel_bg = tuple(d.get("panel_bg", [30, 30, 30]))
        self.text     = tuple(d.get("text",     [255, 255, 255]))
        self.accent   = tuple(d.get("accent",   [0, 200, 255]))
        self.win      = tuple(d.get("win",      [30, 180, 30]))
        self.lose     = tuple(d.get("lose",     [30, 30, 200]))
        self.tie      = tuple(d.get("tie",      [120, 120, 120]))

# Mediapipe-like connections for one hand
MP_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),            # thumb
    (0,5),(5,6),(6,7),(7,8),            # index
    (5,9),(9,10),(10,11),(11,12),       # middle
    (9,13),(13,14),(14,15),(15,16),     # ring
    (13,17),(17,18),(18,19),(19,20),    # pinky
    (0,17)                              # wrist to pinky base
]

# ---------- Gradient Backdrop ----------
_GRAD_CACHE = {}  # key: (w,h)

def apply_gradient_backdrop(frame, phase_t: float, color_a=(10,20,60), color_b=(60,20,100), alpha=0.25):
    """
    Animated horizontal gradient blended over frame.
    color_a/color_b are BGR tuples. phase_t is a phase angle (radians-like scalar).
    """
    h, w = frame.shape[:2]
    key = (w, h)
    if key not in _GRAD_CACHE:
        x = np.linspace(0.0, 1.0, w, dtype=np.float32)
        _GRAD_CACHE[key] = x
    else:
        x = _GRAD_CACHE[key]

    mix = 0.5 + 0.5 * np.sin(2.0 * np.pi * x + phase_t)
    mix = mix.reshape(1, w, 1)

    a = np.array(color_a, dtype=np.float32).reshape(1,1,3)
    b = np.array(color_b, dtype=np.float32).reshape(1,1,3)
    grad_row = a * (1.0 - mix) + b * mix
    grad = np.repeat(grad_row, h, axis=0).astype(np.uint8)

    out = cv2.addWeighted(grad, float(alpha), frame, 1.0 - float(alpha), 0.0)
    return out

# ---------- Vignette ----------
_VIG_CACHE = {}  # key: (w,h,strength,softness)

def apply_vignette(frame, strength=0.45, softness=0.35):
    """
    Multiply the frame by a radial mask: center ~1.0, edges ~1.0-strength.
    softness increases feathering.
    """
    h, w = frame.shape[:2]
    key = (w, h, float(strength), float(softness))
    mask = _VIG_CACHE.get(key)
    if mask is None:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        cx, cy = (w - 1) * 0.5, (h - 1) * 0.5
        dx = (xx - cx) / (w * 0.5)
        dy = (yy - cy) / (h * 0.5)
        r = np.sqrt(dx*dx + dy*dy)
        k = np.clip((r - (1.0 - softness)) / max(1e-6, softness), 0, 1)
        falloff = (1.0 - strength) + (1.0 - (1.0 - k)**2) * strength
        mask = 1.0 - (falloff - (1.0 - strength))
        mask = mask.astype(np.float32)
        _VIG_CACHE[key] = mask
    out = (frame.astype(np.float32) * mask[..., None]).clip(0,255).astype(np.uint8)
    return out

# ---------- Rounded Rect Mask Helper ----------
def _rounded_rect_mask(h, w, x0, y0, x1, y1, r):
    r = int(max(0, r))
    mask = np.zeros((h, w), dtype=np.uint8)
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    x0, y0 = max(0,x0), max(0,y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return mask
    cv2.rectangle(mask, (x0+r, y0), (x1-r, y1), 255, -1)
    cv2.rectangle(mask, (x0, y0+r), (x1, y1-r), 255, -1)
    cv2.ellipse(mask, (x0+r, y0+r), (r, r), 180, 0, 90, 255, -1)
    cv2.ellipse(mask, (x1-r, y0+r), (r, r), 270, 0, 90, 255, -1)
    cv2.ellipse(mask, (x0+r, y1-r), (r, r), 90, 0, 90, 255, -1)
    cv2.ellipse(mask, (x1-r, y1-r), (r, r), 0, 0, 90, 255, -1)
    return mask

# ---------- Glassmorphism Panel ----------
def draw_glass_panel(frame, x0, y0, x1, y1, blur_ksize=11, alpha=0.70, radius=16, border_alpha=0.35, theme=None):
    """
    Draw a frosted glass panel in-place:
      1) Blur the ROI of the existing frame
      2) Tint with theme.panel_bg (or white) with opacity alpha
      3) Rounded corners + soft border
    """
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    x0, y0 = max(0,x0), max(0,y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return

    roi = frame[y0:y1, x0:x1].copy()
    k = int(blur_ksize) if int(blur_ksize) % 2 == 1 else int(blur_ksize) + 1
    if k < 3: k = 3
    blurred = cv2.GaussianBlur(roi, (k, k), 0)

    panel_bg = tuple(getattr(theme, "panel_bg", (255,255,255)))
    tint = np.full_like(roi, panel_bg, dtype=np.uint8)
    frosted = cv2.addWeighted(blurred, 1.0 - float(alpha), tint, float(alpha), 0.0)

    mask = _rounded_rect_mask(h, w, x0, y0, x1, y1, int(radius))
    panel_mask = mask[y0:y1, x0:x1]
    inv = cv2.bitwise_not(panel_mask)

    base = frame[y0:y1, x0:x1]
    base_bg = cv2.bitwise_and(base, base, mask=inv)
    panel_fg = cv2.bitwise_and(frosted, frosted, mask=panel_mask)
    frame[y0:y1, x0:x1] = cv2.add(base_bg, panel_fg)

    if border_alpha > 0:
        border = frame[y0:y1, x0:x1].copy()
        cv2.rectangle(border, (0,0), (x1-x0-1, y1-y0-1), (255,255,255), 1, cv2.LINE_AA)
        frame[y0:y1, x0:x1] = cv2.addWeighted(border, float(border_alpha), frame[y0:y1, x0:x1], 1 - float(border_alpha), 0)

# ---------- Glow Trail ----------
def draw_glow_trail(frame, points, cfg: dict, color=(0,200,255)):
    """
    Draw a speed-weighted polyline trail based on a deque of wrist points.
    cfg keys: base_thickness, max_thickness, base_alpha, speed_gain
    """
    if points is None or len(points) < 2:
        return
    base_t = float(cfg.get("base_thickness", 3))
    max_t  = float(cfg.get("max_thickness", 9))
    base_a = float(cfg.get("base_alpha", 0.12))
    gain   = float(cfg.get("speed_gain", 0.85))

    overlay = frame.copy().astype(np.uint8)
    n = len(points)
    last_speed = 0.0
    for i in range(1, n):
        p0, p1 = points[i-1], points[i]
        dx, dy = (p1[0]-p0[0]), (p1[1]-p0[1])
        speed = (dx*dx + dy*dy) ** 0.5
        speed = 0.7 * last_speed + 0.3 * speed
        last_speed = speed

        age = i / n
        alpha = base_a * (age**0.8)
        thick = int(max(1, min(max_t, base_t + gain * (speed * 0.1))))

        cv2.line(overlay, p0, p1, color, thick, cv2.LINE_AA)
        frame[:] = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

# ---------- Result Spotlight ----------
def draw_result_spotlight(frame, t_norm: float, width_ratio=0.55, intensity=0.35):
    """
    Vertical light band sweeping L->R across the frame as t_norm goes 0..1.
    Adds brightness in a soft band. Returns modified frame.
    """
    h, w = frame.shape[:2]
    cx = int((-0.3 + 1.6 * max(0.0, min(1.0, t_norm))) * w)
    band_w = max(1, int(width_ratio * w))
    x = np.arange(w, dtype=np.float32)
    dist = np.abs(x - cx).reshape(1, w)
    sigma = band_w * 0.35
    mask_x = np.exp(-0.5 * (dist / max(1.0, sigma))**2).astype(np.float32)
    y = np.linspace(-1, 1, h, dtype=np.float32).reshape(h, 1)
    mask_y = np.exp(-0.5 * (y / 0.9)**2)
    mask = (mask_y * mask_x).astype(np.float32)
    add = (mask[..., None] * (intensity * 255.0)).astype(np.float32)
    out = np.clip(frame.astype(np.float32) + add, 0, 255).astype(np.uint8)
    return out

def draw_countdown(frame, seconds_left: int, locked: bool = False,
                   font_path: Optional[str] = None, use_ttf: bool = False,
                   theme: Optional[Theme] = None):
    h, w = frame.shape[:2]
    text = str(max(seconds_left, 0)) if seconds_left > 0 else "Shoot!"
    if use_ttf and font_path:
        bg = (theme.panel_bg if theme else (0,0,0))
        frame[:] = add_translucent_bar(frame, (0, int(h*0.38)), (w, int(h*0.62)), bg, 0.35)
        size = 92 if seconds_left > 0 else 72
        col = (theme.text if theme else (255,255,255))
        frame[:] = draw_ttf_center(frame, text, (w//2, h//2), font_path, size,
                                   color=col, glow=True, glow_color=(0,0,0), glow_w=6)
    else:
        scale = 3 if seconds_left > 0 else 2
        thickness = 6 if seconds_left > 0 else 4
        size_px, _ = cv2.getTextSize(text, FONT, scale, thickness)
        col = (theme.text if theme else (255,255,255))
        cv2.putText(frame, text, ((w - size_px[0]) // 2, (h + size_px[1]) // 2),
                    FONT, scale, col, thickness, cv2.LINE_AA)

    if locked:
        cv2.putText(frame, "LOCKED", (w - 160, 60), FONT, 0.8, (0, 200, 0), 2, cv2.LINE_AA)

def draw_moves(frame, player: str, ai: str,
               font_path: Optional[str] = None, use_ttf: bool = False,
               theme: Optional[Theme] = None):
    if use_ttf and font_path:
        acc = theme.accent if theme else (0,255,255)
        frame[:] = draw_ttf(frame, f"You: {player}", (20, 12), font_path, 26, acc)
        frame[:] = draw_ttf(frame, f"AI:  {ai}",    (20, 46), font_path, 26, acc)
    else:
        acc = theme.accent if theme else (0,255,255)
        cv2.putText(frame, f"You: {player}", (20, 40), FONT, 1, acc, 2, cv2.LINE_AA)
        cv2.putText(frame, f"AI:  {ai}",    (20, 80), FONT, 1, acc, 2, cv2.LINE_AA)

def _ease_out_cubic(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 1 - (1 - t) ** 3

def draw_result_animated(frame, result: str, t_norm: float,
                         font_path: Optional[str] = None, use_ttf: bool = False,
                         theme: Optional[Theme] = None):
    h, w = frame.shape[:2]
    t = _ease_out_cubic(t_norm)

    if theme:
        col_map = {"win": theme.win, "lose": theme.lose, "tie": theme.tie, "invalid": theme.accent}
        col = col_map.get(result, theme.text)
    else:
        colors = {"win": (30,180,30), "lose": (30,30,200), "tie": (120,120,120), "invalid": (0,140,255)}
        col = colors.get(result, (200,200,200))

    bar_h = int(h * 0.18)
    y = int(-bar_h + t * (bar_h + int(h * 0.05)))
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y), (w, y + bar_h), col, -1)
    frame[:] = cv2.addWeighted(overlay, 0.28, frame, 0.72, 0)

    text = result.upper()
    if use_ttf and font_path:
        size = int(60 + 26 * t)
        txt_col = theme.text if theme else (255,255,255)
        frame[:] = draw_ttf_center(frame, text, (w // 2, y + bar_h // 2), font_path, size,
                                   color=txt_col, glow=True, glow_color=(0,0,0), glow_w=8)
    else:
        scale = 1.2 + 0.4 * t
        thickness = 2 + int(2 * t)
        box, _ = cv2.getTextSize(text, FONT, scale, thickness)
        cx, cy = w // 2, y + bar_h // 2
        txt_col = theme.text if theme else (255,255,255)
        cv2.putText(frame, text, (cx - box[0] // 2, cy + box[1] // 2),
                    FONT, scale, txt_col, thickness, cv2.LINE_AA)

def draw_hint(frame, fps: float, font_path: Optional[str] = None, use_ttf: bool = False):
    h, w = frame.shape[:2]
    left = "SPACE: start  |  Q: quit  |  1/2/3: difficulty"
    right = f"FPS: {fps:.1f}"
    if use_ttf and font_path:
        frame[:] = draw_ttf(frame, left, (20, h - 28), font_path, 22, (255,255,255))
        frame[:] = draw_ttf_right(frame, right, (w - 20, 18), font_path, 22, (180,180,180))
        if fps > 0 and fps < 20:
            frame[:] = draw_ttf(frame, "Low FPS – reduce resolution in config.yaml",
                                (20, h - 56), font_path, 20, (0,200,255))
    else:
        cv2.putText(frame, left, (20, h - 20), FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, right, (w - 140, 30), FONT, 0.6, (180, 180, 180), 1, cv2.LINE_AA)
        if fps > 0 and fps < 20:
            cv2.putText(frame, "Low FPS – reduce resolution in config.yaml",
                        (20, h - 50), FONT, 0.6, (0, 200, 255), 1, cv2.LINE_AA)

def draw_warning(frame, msg: str, theme: Optional[Theme] = None):
    h, w = frame.shape[:2]
    bg = theme.panel_bg if theme else (30,30,30)
    acc = theme.accent if theme else (0,200,255)
    cv2.rectangle(frame, (0, 0), (w, 40), bg, -1)
    cv2.putText(frame, msg, (14, 28), FONT, 0.7, acc, 2, cv2.LINE_AA)

def draw_scoreboard(frame, wins: int, losses: int, ties: int, rounds: int, difficulty: str,
                    font_path: Optional[str] = None, use_ttf: bool = False,
                    theme: Optional[Theme] = None, use_glass: bool = False,
                    glass_cfg: Optional[dict] = None):
    h, w = frame.shape[:2]
    panel_w, panel_h = 260, 120
    x0, y0 = w - panel_w - 10, 10
    if use_glass:
        gc = glass_cfg or {}
        draw_glass_panel(frame, x0, y0, x0 + panel_w, y0 + panel_h,
                         blur_ksize=int(gc.get('blur_ksize', 11)),
                         alpha=float(gc.get('alpha', 0.70)),
                         radius=int(gc.get('radius', 16)),
                         border_alpha=float(gc.get('border_alpha', 0.35)),
                         theme=theme)
    else:
        bg = theme.panel_bg if theme else (30,30,30)
        cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), bg, -1)

    if use_ttf and font_path:
        txt = theme.text if theme else (255,255,255)
        acc = theme.accent if theme else (0,200,255)
        frame[:] = draw_ttf(frame, f"Rounds: {rounds}", (x0 + 10, y0 + 10), font_path, 22, txt)
        frame[:] = draw_ttf(frame, f"W / L / T: {wins} / {losses} / {ties}", (x0 + 10, y0 + 38), font_path, 22, (200,200,200))
        frame[:] = draw_ttf(frame, f"Difficulty: {difficulty}", (x0 + 10, y0 + 66), font_path, 22, acc)
    else:
        txt = theme.text if theme else (255,255,255)
        acc = theme.accent if theme else (0,200,255)
        cv2.putText(frame, f"Rounds: {rounds}", (x0 + 10, y0 + 30), FONT, 0.6, txt, 1, cv2.LINE_AA)
        cv2.putText(frame, f"W / L / T: {wins} / {losses} / {ties}", (x0 + 10, y0 + 55), FONT, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Difficulty: {difficulty}", (x0 + 10, y0 + 85), FONT, 0.6, acc, 1, cv2.LINE_AA)

def draw_hand_skeleton(frame, pts: List[Tuple[int,int]], trail: Optional["Deque[Tuple[int,int]]"] = None):
    if not pts or len(pts) < 21:
        return
    # bones
    for a, b in MP_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (255, 255, 255), 2, cv2.LINE_AA)
    # joints
    for i, p in enumerate(pts):
        r = 4 if i in (4, 8, 12, 16, 20) else 3
        cv2.circle(frame, p, r, (0, 255, 255), -1, cv2.LINE_AA)
    # wrist trail (landmark 0)
    if trail and len(trail) > 1:
        for i in range(1, len(trail)):
            cv2.line(frame, trail[i-1], trail[i], (0, 200, 255), 2, cv2.LINE_AA)

# --- Compatibility wrapper for older imports ---
def draw_result(frame, result: str):
    try:
        draw_result_animated(frame, result, 1.0)
    except Exception:
        h, w = frame.shape[:2]
        text = result.upper()
        size, _ = cv2.getTextSize(text, FONT, 1.8, 4)
        x = (w - size[0]) // 2
        y = int(h * 0.15)
        cv2.putText(frame, text, (x, y), FONT, 1.8, (255, 255, 255), 4, cv2.LINE_AA)

# --- Enhanced helpers appended below ---
def draw_hint(frame, fps: float, font_path: Optional[str] = None, use_ttf: bool = False,
              theme: Optional[Theme] = None, use_glass: bool = False,
              glass_cfg: Optional[dict] = None):
    h, w = frame.shape[:2]
    left = "SPACE: start  |  Q: quit  |  1/2/3: difficulty"
    right = f"FPS: {fps:.1f}"
    txt_col = theme.text if theme else (255,255,255)
    acc_col = theme.accent if theme else (180,180,180)
    # Optional glass strip at bottom
    if use_glass:
        gc = glass_cfg or {}
        y1 = h - 10
        y0 = max(0, y1 - 58)
        draw_glass_panel(frame, 12, y0, w - 12, y1,
                         blur_ksize=int(gc.get('blur_ksize', 11)),
                         alpha=float(gc.get('alpha', 0.70)),
                         radius=int(gc.get('radius', 16)),
                         border_alpha=float(gc.get('border_alpha', 0.35)),
                         theme=theme)
    if use_ttf and font_path:
        frame[:] = draw_ttf(frame, left, (20, h - 28), font_path, 22, txt_col)
        frame[:] = draw_ttf_right(frame, right, (w - 20, 18), font_path, 22, acc_col)
        if fps > 0 and fps < 20:
            frame[:] = draw_ttf(frame, "Low FPS - reduce resolution in config.yaml",
                                (20, h - 56), font_path, 20, acc_col)
    else:
        cv2.putText(frame, left, (20, h - 20), FONT, 0.6, txt_col, 1, cv2.LINE_AA)
        cv2.putText(frame, right, (w - 140, 30), FONT, 0.6, acc_col, 1, cv2.LINE_AA)
        if fps > 0 and fps < 20:
            cv2.putText(frame, "Low FPS - reduce resolution in config.yaml",
                        (20, h - 50), FONT, 0.6, acc_col, 1, cv2.LINE_AA)


def draw_countdown_morph(frame, current_txt: str, next_txt: str, t: float,
                         font_path: Optional[str], use_ttf: bool,
                         theme: Optional[Theme]):
    h, w = frame.shape[:2]
    t = max(0.0, min(1.0, float(t)))
    band_col = theme.panel_bg if theme else (0,0,0)
    base = add_translucent_bar(frame.copy(), (0, int(h*0.38)), (w, int(h*0.62)), band_col, 0.35)

    # Sizes
    def size_for(txt, base_size=92):
        return 72 if txt.lower() == "shoot!" else base_size

    cur_scale = 1.0 + 0.15 * t
    nxt_scale = 0.85 + 0.15 * t
    cur_size = int(size_for(current_txt) * cur_scale)
    nxt_size = int(size_for(next_txt) * nxt_scale)
    txt_col = theme.text if theme else (255,255,255)

    cur = base.copy()
    nxt = base.copy()
    if use_ttf and font_path:
        cur[:] = draw_ttf_center(cur, current_txt, (w//2, h//2), font_path, cur_size, color=txt_col, glow=True, glow_color=(0,0,0), glow_w=6)
        nxt[:] = draw_ttf_center(nxt, next_txt, (w//2, h//2), font_path, nxt_size, color=txt_col, glow=True, glow_color=(0,0,0), glow_w=6)
    else:
        for img, txt, size in ((cur, current_txt, cur_size), (nxt, next_txt, nxt_size)):
            scale = max(0.5, size/40.0)
            thickness = 2 + int(scale)
            box, _ = cv2.getTextSize(txt, FONT, scale, thickness)
            cv2.putText(img, txt, ((w - box[0])//2, (h + box[1])//2), FONT, scale, txt_col, thickness, cv2.LINE_AA)

    out = cv2.addWeighted(cur, 1.0 - t, nxt, t, 0)
    frame[:] = out


def apply_bloom(frame, threshold=220, intensity=0.55, blur_ksize=17):
    k = int(blur_ksize)
    if k % 2 == 0:
        k += 1
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    _, mask = cv2.threshold(v, threshold, 255, cv2.THRESH_TOZERO)
    bright = cv2.merge([mask, mask, mask])
    bright_blur = cv2.GaussianBlur(bright, (k, k), 0)
    out = cv2.addWeighted(frame, 1.0, bright_blur, float(intensity), 0)
    return np.clip(out, 0, 255).astype(frame.dtype)


def draw_start_screen(frame, difficulty_name: str, font_path: Optional[str],
                      use_ttf: bool, theme: Optional[Theme]):
    h, w = frame.shape[:2]
    # backdrop
    overlay = frame.copy()
    bg = theme.panel_bg if theme else (30,30,30)
    cv2.rectangle(overlay, (int(w*0.08), int(h*0.18)), (int(w*0.92), int(h*0.82)), bg, -1)
    frame[:] = cv2.addWeighted(overlay, 0.8, frame, 0.2, 0)

    title = "Gesture RPS"
    subtitle = "Choose difficulty: 1=Random  2=Adaptive  3=Markov"
    footer = "T: theme  •  B: bloom  •  SPACE: start  •  Q: quit"
    col_txt = theme.text if theme else (255,255,255)
    col_acc = theme.accent if theme else (0,200,255)

    if use_ttf and font_path:
        frame[:] = draw_ttf_center(frame, title, (w//2, int(h*0.35)), font_path, 56, color=col_txt, glow=True, glow_color=(0,0,0), glow_w=6)
        frame[:] = draw_ttf_center(frame, subtitle, (w//2, int(h*0.50)), font_path, 28, color=col_acc)
        frame[:] = draw_ttf_center(frame, f"Difficulty: {difficulty_name}", (w//2, int(h*0.62)), font_path, 26, color=col_acc)
        frame[:] = draw_ttf_center(frame, footer, (w//2, int(h*0.76)), font_path, 22, color=col_txt)
    else:
        cv2.putText(frame, title, (int(w*0.18), int(h*0.36)), FONT, 1.6, col_txt, 3, cv2.LINE_AA)
        cv2.putText(frame, subtitle, (int(w*0.10), int(h*0.50)), FONT, 0.8, col_acc, 2, cv2.LINE_AA)
        cv2.putText(frame, f"Difficulty: {difficulty_name}", (int(w*0.28), int(h*0.62)), FONT, 0.8, col_acc, 2, cv2.LINE_AA)
        cv2.putText(frame, footer, (int(w*0.14), int(h*0.76)), FONT, 0.7, col_txt, 1, cv2.LINE_AA)
