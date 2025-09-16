import cv2
import numpy as np
import mediapipe as mp

# Indices for finger joints (MediaPipe Hands)
FINGER_TIPS = [8, 12, 16, 20]
FINGER_PIPS = [6, 10, 14, 18]
FINGER_MCPS = [5, 9, 13, 17]

class GestureDetector:
    """
    Day 2/3: multi-frame voting + basic guards + hand-points overlay helper.
    """
    def __init__(self, config: dict):
        self.cfg = config
        self.mp_hands = mp.solutions.hands
        mp_cfg = self.cfg.get("mediapipe", {})
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            model_complexity=int(mp_cfg.get("model_complexity", 0)),
            max_num_hands=int(mp_cfg.get("max_num_hands", 1)),
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        # Performance helpers
        self.proc_scale = float(self.cfg.get("video", {}).get("process_scale", 1.0))
        self.detect_stride = max(1, int(mp_cfg.get("detect_stride", 1)))
        self._overlay_counter = 0
        self._last_pts = None

    # ---------- Utility guards ----------
    @staticmethod
    def _mean_brightness(frame_bgr) -> float:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray))

    @staticmethod
    def _motion_score(frames, last_n=3) -> float:
        """
        Avg abs-diff across last_n-1 consecutive frames (0..255).
        """
        if len(frames) < 2:
            return 0.0
        last = frames[-last_n:] if len(frames) >= last_n else frames
        diffs = []
        for a, b in zip(last[:-1], last[1:]):
            g1 = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
            g2 = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
            diffs.append(np.mean(cv2.absdiff(g1, g2)))
        return float(np.mean(diffs)) if diffs else 0.0

    # ---------- Single-frame classify ----------
    def _classify_from_landmarks(self, lm):
        """
        Simple rules based on Tip vs PIP Y positions for 4 fingers (thumb ignored).
        """
        extended = 0
        idx_mid_extended = [False, False, False, False]
        for k, (tip, pip) in enumerate(zip(FINGER_TIPS, FINGER_PIPS)):
            tip_y = lm[tip].y
            pip_y = lm[pip].y
            is_ext = tip_y < pip_y  # smaller y is higher on screen → extended
            idx_mid_extended[k] = is_ext
            if is_ext:
                extended += 1

        if extended <= 1:
            return "rock", 0.9
        if idx_mid_extended[0] and idx_mid_extended[1] and not idx_mid_extended[2] and not idx_mid_extended[3]:
            return "scissors", 0.9
        if extended >= 3:
            return "paper", 0.9
        return "invalid", 0.4

    def _classify_from_landmarks_v2(self, lm):
        """
        More robust rules using a Y-margin and an index–middle separation check.
        Prevents single index finger from being misread as scissors/rock.
        """
        thr = self.cfg.get("thresholds", {})
        ext_margin = float(thr.get("ext_y_margin", 0.02))
        ext_score_margin = float(thr.get("ext_score_margin", 0.04))
        min_sc_gap = float(thr.get("min_scissor_gap_norm", 0.20))  # normalized by hand bbox size
        min_sc_gap_both = float(thr.get("min_scissor_gap_both_extended", 0.30))
        sc_gap_fallback = float(thr.get("scissor_fallback_gap_norm", max(0.15, 0.8 * min_sc_gap)))
        paper_min_ext = int(thr.get("paper_min_extended", 4))
        paper_min_ext_loose = int(thr.get("paper_min_extended_loose", 3))
        paper_spread_min = float(thr.get("paper_spread_min", 0.42))

        def is_extended_y(tip, pip):
            return lm[tip].y < lm[pip].y - ext_margin

        # Orientation-robust extension using distances from wrist (0)
        def extended_geom(tip, pip, mcp, hand_size):
            import math
            wx, wy = lm[0].x, lm[0].y
            tx, ty = lm[tip].x, lm[tip].y
            px, py = lm[pip].x, lm[pip].y
            # score: how much further tip is from wrist than pip
            st = math.hypot(tx - wx, ty - wy)
            sp = math.hypot(px - wx, py - wy)
            return (st - sp) / max(hand_size, 1e-6) > ext_score_margin

        # Compute hand size first for geometric check and gaps
        xs = [p.x for p in lm]
        ys = [p.y for p in lm]
        bbox_w = max(xs) - min(xs)
        bbox_h = max(ys) - min(ys)
        hand_size = max(bbox_w, bbox_h) if (bbox_w > 0 and bbox_h > 0) else 1e-6

        # Combine y-test with geometric test for robustness
        flags = [
            is_extended_y(t, p) or extended_geom(t, p, m, hand_size)
            for (t, p, m) in zip(FINGER_TIPS, FINGER_PIPS, FINGER_MCPS)
        ]
        index_ext, middle_ext, ring_ext, pinky_ext = flags
        extended = sum(1 for f in flags if f)
        # Index–middle fingertip separation
        dx = lm[8].x - lm[12].x
        dy = lm[8].y - lm[12].y
        idx_mid_gap = float(np.hypot(dx, dy)) / hand_size
        # Thumb tip to index MCP separation (thumb-up check)
        tdx = lm[4].x - lm[5].x
        tdy = lm[4].y - lm[5].y
        thumb_gap = float(np.hypot(tdx, tdy)) / hand_size
        max_thumb_gap_for_rock = float(thr.get("max_thumb_gap_for_rock_norm", 0.33))

        if extended == 0:
            # Closed fist is rock, but thumbs-up should not be rock
            if thumb_gap > max_thumb_gap_for_rock:
                return "invalid", 0.80
            return "rock", 0.95

        if index_ext and not middle_ext and not ring_ext and not pinky_ext:
            return "invalid", 0.60

        # Paper first: if 3+ fingers are extended (typical open hand) favor paper
        # Strict: 4 extended
        if extended >= paper_min_ext:
            return "paper", 0.90
        # Loose: >=3 extended and wide splay between index and pinky
        ip_dx = lm[8].x - lm[20].x
        ip_dy = lm[8].y - lm[20].y
        ip_spread = float(np.hypot(ip_dx, ip_dy)) / hand_size
        if extended >= paper_min_ext_loose and ip_spread >= paper_spread_min:
            return "paper", 0.82

        # Scissors: index + middle extended; allow ring/pinky states but demand larger gap if both extended
        if index_ext and middle_ext and not (ring_ext and pinky_ext):
            required_gap = min_sc_gap
            if idx_mid_gap >= required_gap:
                return "scissors", 0.93
            if idx_mid_gap >= sc_gap_fallback:
                return "scissors", 0.85
            return "invalid", 0.55

        if extended <= 1:
            return "rock", 0.75
        return "invalid", 0.40

    def _predict_single(self, frame_bgr):
        """
        Returns (label, conf, landmarks_count).
        Chooses the best-visible hand if multiple are present.
        """
        # Optional downscale for performance (normalized landmarks are resolution-agnostic)
        if 0.5 <= self.proc_scale < 1.0:
            frame_in = cv2.resize(frame_bgr, None, fx=self.proc_scale, fy=self.proc_scale, interpolation=cv2.INTER_AREA)
        else:
            frame_in = frame_bgr
        rgb = cv2.cvtColor(frame_in, cv2.COLOR_BGR2RGB)
        res = self.hands.process(rgb)
        if not res.multi_hand_landmarks:
            return "invalid", 0.0, 0

        # Prefer hand with highest handedness score (more stable)
        best = 0
        best_score = -1.0
        if res.multi_handedness:
            for i, handed in enumerate(res.multi_handedness):
                score = handed.classification[0].score
                if score > best_score:
                    best_score = score
                    best = i

        lm = res.multi_hand_landmarks[best].landmark
        label, conf = self._classify_from_landmarks_v2(lm)
        return label, conf, 21  # 21 landmarks when a hand is detected

    # ---------- Multi-frame inference with guards ----------
    def predict_gesture(self, frames):
        """
        Multi-frame inference with majority voting and guards.
        Returns: (label, confidence, reason)
        reason: '' on success or a short message for redo.
        """
        if not frames:
            return "invalid", 0.0, "no-frames"

        thr = self.cfg.get("thresholds", {})
        min_brightness = float(thr.get("min_brightness", 40.0))
        max_motion_last = float(thr.get("max_motion_last", 12.0))
        min_landmarks = int(thr.get("min_landmarks", 15))
        gesture_conf = float(thr.get("gesture_conf", 0.60))

        # Guards: brightness & final motion
        if self._mean_brightness(frames[-1]) < min_brightness:
            return "invalid", 0.0, "too-dark"

        motion = self._motion_score(frames, last_n=3)
        if motion > max_motion_last:
            return "invalid", 0.0, "too-much-motion"

        labels, confs, lm_counts = [], [], []
        for f in frames:
            l, c, k = self._predict_single(f)
            labels.append(l)
            confs.append(c)
            lm_counts.append(k)

        if max(lm_counts) < min_landmarks:
            return "invalid", 0.0, "low-occlusion"

        # Voting (ignore 'invalid' unless nothing else)
        votes = {"rock": 0, "paper": 0, "scissors": 0}
        for l in labels:
            if l in votes:
                votes[l] += 1

        if sum(votes.values()) == 0:
            return "invalid", 0.0, "no-valid-frames"

        label = max(votes.items(), key=lambda x: x[1])[0]
        conf = float(np.mean([c for (l, c) in zip(labels, confs) if l == label])) if votes[label] > 0 else 0.0

        if conf < gesture_conf:
            return "invalid", conf, "low-confidence"

        return label, conf, ""

    # ---------- Live overlay helper ----------
    def get_hand_points_for_overlay(self, frame_bgr):
        """
        Returns a list of (x,y) pixel points for the best-visible hand (21 landmarks), or None.
        Lightweight call for per-frame overlay (idle/countdown/result).
        """
        # stride updates to improve FPS
        self._overlay_counter += 1
        do_process = (self._overlay_counter % self.detect_stride) == 0
        if do_process:
            if 0.5 <= self.proc_scale < 1.0:
                frame_in = cv2.resize(frame_bgr, None, fx=self.proc_scale, fy=self.proc_scale, interpolation=cv2.INTER_AREA)
            else:
                frame_in = frame_bgr
            rgb = cv2.cvtColor(frame_in, cv2.COLOR_BGR2RGB)
            res = self.hands.process(rgb)
        else:
            res = None
        if not res or not res.multi_hand_landmarks:
            return self._last_pts

        best_i = 0
        best_score = -1.0
        if res.multi_handedness:
            for i, handed in enumerate(res.multi_handedness):
                score = handed.classification[0].score
                if score > best_score:
                    best_score = score
                    best_i = i

        lm = res.multi_hand_landmarks[best_i].landmark
        h, w = frame_bgr.shape[:2]
        pts = [(int(p.x * w), int(p.y * h)) for p in lm]
        self._last_pts = pts
        return pts
