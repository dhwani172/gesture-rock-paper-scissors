import os
import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image

def _bgr_to_rgb(c):
    return (int(c[2]), int(c[1]), int(c[0]))

def _load_font(font_path: str, font_size: int):
    """Load .otf/.ttf safely; return None on failure."""
    try:
        if not font_path:
            return None
        path = os.path.normpath(font_path)
        return ImageFont.truetype(path, font_size)
    except Exception:
        return None

def draw_ttf_center(frame_bgr, text: str, center_xy, font_path: str, font_size: int,
                    color=(255, 255, 255), glow=False, glow_color=(0, 0, 0), glow_w=6):
    """Draw centered text using PIL if font loads; else return frame unchanged."""
    font = _load_font(font_path, font_size)
    if font is None:
        return frame_bgr

    img = Image.fromarray(frame_bgr[:, :, ::-1])  # BGR->RGB
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), text, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = int(center_xy[0] - w / 2)
    y = int(center_xy[1] - h / 2)
    fill = _bgr_to_rgb(color)
    stroke = _bgr_to_rgb(glow_color)
    draw.text((x, y), text, font=font, fill=fill,
              stroke_width=(glow_w if glow else 0), stroke_fill=stroke)
    return np.array(img)[:, :, ::-1]  # RGB->BGR

def draw_ttf(frame_bgr, text: str, topleft_xy, font_path: str, font_size: int,
             color=(255, 255, 255), glow=False, glow_color=(0, 0, 0), glow_w=2):
    """Draw top-left anchored text with PIL if font loads; else return frame unchanged."""
    font = _load_font(font_path, font_size)
    if font is None:
        return frame_bgr

    img = Image.fromarray(frame_bgr[:, :, ::-1])
    draw = ImageDraw.Draw(img)
    fill = _bgr_to_rgb(color)
    stroke = _bgr_to_rgb(glow_color)
    draw.text(topleft_xy, text, font=font, fill=fill,
              stroke_width=(glow_w if glow else 0), stroke_fill=stroke)
    return np.array(img)[:, :, ::-1]

def draw_ttf_right(frame_bgr, text: str, top_right_xy, font_path: str, font_size: int,
                   color=(255, 255, 255), glow=False, glow_color=(0, 0, 0), glow_w=2):
    """Draw text right-aligned to (x_right, y_top)."""
    font = _load_font(font_path, font_size)
    if font is None:
        return frame_bgr

    img = Image.fromarray(frame_bgr[:, :, ::-1])
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    x = int(top_right_xy[0] - w)
    y = int(top_right_xy[1])
    fill = _bgr_to_rgb(color)
    stroke = _bgr_to_rgb(glow_color)
    draw.text((x, y), text, font=font, fill=fill,
              stroke_width=(glow_w if glow else 0), stroke_fill=stroke)
    return np.array(img)[:, :, ::-1]

def add_translucent_bar(frame_bgr, top_left, bottom_right, color_bgr=(0, 0, 0), alpha=0.35):
    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, top_left, bottom_right, color_bgr, -1)
    return cv2.addWeighted(overlay, alpha, frame_bgr, 1 - alpha, 0)
