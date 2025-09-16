import av
import cv2
import numpy as np
import yaml
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

from src.gesture_rps.gesture_detector import GestureDetector
from src.gesture_rps.ui_overlay import draw_hand_skeleton

st.set_page_config(page_title="Gesture RPS", layout="wide")

with open("src/gesture_rps/config.yaml", "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

class Transformer(VideoTransformerBase):
    def __init__(self):
        self.detector = GestureDetector(CFG)

    def recv(self, frame: av.VideoFrame):
        img = frame.to_ndarray(format="bgr24")
        pts = self.detector.get_hand_points_for_overlay(img)
        if pts:
            draw_hand_skeleton(img, pts, None)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

st.title("Gesture RPS — Streamlit demo")
webrtc_streamer(
    key="rps",
    mode=WebRtcMode.SENDRECV,
    video_transformer_factory=Transformer,
    media_stream_constraints={"video": True, "audio": False},
)
st.caption("Tip: good lighting helps tracking.")
