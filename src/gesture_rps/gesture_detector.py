# gesture_detector.py
import cv2
import numpy as np
import mediapipe as mp

# ---- Landmark index groups (MediaPipe Hands) ----
FINGER_TIPS = [8, 12, 16, 20]
FINGER_PIPS = [6, 10, 14, 18]
FINGER_MCPS = [5, 9, 13, 17]


class GestureDetector:
    """
    Multi-frame voting detector with strict rules for paper/scissors/invalid
    and a more permissive (but safe) rock classifier. Includes an overlay helper.
    """

    # ──────────────────────────────────────────────────────────────────────────
    # Init
    # ──────────────────────────────────────────────────────────────────────────
    def __init__(self, config: dict):
        self.cfg = config or {}
        self.mp_hands = mp.solutions.hands
        mp_cfg = self.cfg.get("mediapipe", {})
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            model_complexity=int(mp_cfg.get("model_complexity", 0)),
            max_num_hands=int(mp_cfg.get("max_num_hands", 1)),
            min_detection_confidence=float(mp_cfg.get("min_detection_confidence", 0.5)),
            min_tracking_confidence=float(mp_cfg.get("min_tracking_confidence", 0.5)),
        )
        # Performance
        self.proc_scale = float(self.cfg.get("video", {}).get("process_scale", 1.0))
        self.detect_stride = max(1, int(mp_cfg.get("detect_stride", 1)))

        # Overlay cache
        self._overlay_counter = 0
        self._last_pts = None

    # ──────────────────────────────────────────────────────────────────────────
    # Guards for multi-frame voting
    # ──────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _mean_brightness(frame_bgr) -> float:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray))

    @staticmethod
    def _motion_score(frames, last_n=3) -> float:
        if len(frames) < 2:
            return 0.0
        last = frames[-last_n:] if len(frames) >= last_n else frames
        diffs = []
        for a, b in zip(last[:-1], last[1:]):
            g1 = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
            g2 = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
            diffs.append(np.mean(cv2.absdiff(g1, g2)))
        return float(np.mean(diffs)) if diffs else 0.0

    # ──────────────────────────────────────────────────────────────────────────
    # Legacy baseline (kept for reference)
    # ──────────────────────────────────────────────────────────────────────────
    def _classify_from_landmarks(self, lm):
        extended = 0
        idx_mid_extended = [False, False, False, False]
        for k, (tip, pip) in enumerate(zip(FINGER_TIPS, FINGER_PIPS)):
            is_ext = lm[tip].y < lm[pip].y
            idx_mid_extended[k] = is_ext
            extended += int(is_ext)

        if extended <= 1:
            return "rock", 0.9
        if idx_mid_extended[0] and idx_mid_extended[1] and not idx_mid_extended[2] and not idx_mid_extended[3]:
            return "scissors", 0.9
        if extended >= 3:
            return "paper", 0.9
        return "invalid", 0.4

    # ──────────────────────────────────────────────────────────────────────────
    # Main rule-based classifier (strict paper/scissors, permissive rock)
    # ──────────────────────────────────────────────────────────────────────────
    def _classify_from_landmarks_v2(self, lm):
        thr = self.cfg.get("thresholds", {})

        # Extension/geometry thresholds
        ext_margin = float(thr.get("ext_y_margin", 0.02))
        ext_score_margin = float(thr.get("ext_score_margin", 0.05))   # "strong" extension
        ext_score_loose = float(thr.get("ext_score_loose_norm", 0.02))  # "slight" extension

        # Scissors & Paper strictness
        min_sc_gap = float(thr.get("min_scissor_gap_norm", 0.24))
        sc_gap_fallback = float(thr.get("scissor_fallback_gap_norm", 0.20))
        paper_spread_min_strict = float(thr.get("paper_spread_min_strict", 0.55))
        min_adj_gap_for_paper = float(thr.get("paper_min_adj_gap_norm", 0.22))
        min_thumb_gap_for_paper = float(thr.get("min_thumb_gap_for_paper_norm", 0.26))

        # Rock guards (loosened)
        max_thumb_gap_for_rock = float(thr.get("max_thumb_gap_for_rock_norm", 0.33))
        max_thumb_gap_for_rock_loose = float(thr.get("max_thumb_gap_for_rock_loose_norm", 0.38))
        thumb_ext_thresh = float(thr.get("thumb_ext_thresh_norm", 0.20))
        rock_max_avg_adj = float(thr.get("rock_max_avg_adj_gap_norm", 0.19))
        rock_max_spread = float(thr.get("rock_max_index_pinky_spread_norm", 0.45))

        # Helpers
        def is_extended_y(tip, pip):
            return lm[tip].y < lm[pip].y - ext_margin

        def ext_score(tip, pip, hand_size):
            import math
            wx, wy = lm[0].x, lm[0].y
            tx, ty = lm[tip].x, lm[tip].y
            px, py = lm[pip].x, lm[pip].y
            return (math.hypot(tx - wx, ty - wy) - math.hypot(px - wx, py - wy)) / max(hand_size, 1e-6)

        import math
        def gap(a_tip, b_tip, hs):
            return math.hypot(lm[a_tip].x - lm[b_tip].x, lm[a_tip].y - lm[b_tip].y) / hs

        # Scale
        xs = [p.x for p in lm]; ys = [p.y for p in lm]
        bbox_w = max(xs) - min(xs); bbox_h = max(ys) - min(ys)
        hand_size = max(bbox_w, bbox_h) if (bbox_w > 0 and bbox_h > 0) else 1e-6

        # Non-thumb finger states
        flags_y = [is_extended_y(t, p) for (t, p) in zip(FINGER_TIPS, FINGER_PIPS)]
        scores = [ext_score(t, p, hand_size) for (t, p) in zip(FINGER_TIPS, FINGER_PIPS)]
        strong_ext = [s > ext_score_margin for s in scores]
        slight_ext = [s > ext_score_loose for s in scores]

        index_ext, middle_ext, ring_ext, pinky_ext = [bool(x) for x in flags_y]
        extended_count_y = sum(1 for f in flags_y if f)
        strong_count = sum(1 for f in strong_ext if f)
        slight_count = sum(1 for f in slight_ext if f)

        # Gaps / spreads
        idx_mid_gap = gap(8, 12, hand_size)
        mid_ring_gap = gap(12, 16, hand_size)
        ring_pinky_gap = gap(16, 20, hand_size)
        index_pinky_spread = gap(8, 20, hand_size)
        avg_adj = (idx_mid_gap + mid_ring_gap + ring_pinky_gap) / 3.0

        # Thumb openness
        thumb_gap = gap(4, 5, hand_size)
        thumb_extended = thumb_gap >= thumb_ext_thresh

        # ---------- ROCK (loosened) ----------
        # Accept when: no strong extension; at most 1 slight extension;
        # adjacent gaps + spread are small; thumb not clearly open.
        if strong_count == 0 and slight_count <= 1:
            if (avg_adj <= rock_max_avg_adj and
                index_pinky_spread <= rock_max_spread and
                thumb_gap <= max_thumb_gap_for_rock_loose and
                not thumb_extended):
                return "rock", 0.90

        # If absolutely closed: old strict rule (still allowed)
        if extended_count_y == 0:
            if thumb_gap > max_thumb_gap_for_rock or thumb_extended:
                return "invalid", 0.80
            return "rock", 0.92

        # Single-finger-up → invalid
        if extended_count_y == 1:
            return "invalid", 0.70
        if index_ext and not (middle_ext or ring_ext or pinky_ext):
            return "invalid", 0.70

        # ---------- PAPER (strict + relaxed fallback) ----------
        flags_both = [fy and (sc > ext_score_margin) for fy, sc in zip(flags_y, scores)]
        all_four_strong = all(flags_both)
        all_four_y = all(flags_y)

        if all_four_strong:
            if (thumb_gap >= min_thumb_gap_for_paper and
                index_pinky_spread >= paper_spread_min_strict and
                idx_mid_gap >= min_adj_gap_for_paper and
                mid_ring_gap >= min_adj_gap_for_paper and
                ring_pinky_gap >= min_adj_gap_for_paper):
                return "paper", 0.92
            # relaxed fallback for slanted hands
            if (thumb_gap >= min_thumb_gap_for_paper * 0.85 and
                index_pinky_spread >= max(0.48, paper_spread_min_strict - 0.07) and
                avg_adj >= max(0.18, min_adj_gap_for_paper - 0.04)):
                return "paper", 0.86
            return "invalid", 0.75

        if all_four_y:
            if (thumb_gap >= min_thumb_gap_for_paper * 0.90 and
                index_pinky_spread >= max(0.50, paper_spread_min_strict - 0.05) and
                avg_adj >= max(0.20, min_adj_gap_for_paper - 0.02)):
                return "paper", 0.84

        # 3-finger shapes are not paper
        if extended_count_y == 3:
            return "invalid", 0.68

        # ---------- SCISSORS ----------
        if index_ext and middle_ext and (not ring_ext) and (not pinky_ext):
            if idx_mid_gap >= min_sc_gap:
                return "scissors", 0.93
            if idx_mid_gap >= sc_gap_fallback:
                return "scissors", 0.85
            return "invalid", 0.55

        # ---------- Fallback ----------
        return "invalid", 0.40

    # ──────────────────────────────────────────────────────────────────────────
    # Single-frame prediction
    # ──────────────────────────────────────────────────────────────────────────
    def _predict_single(self, frame_bgr):
        if 0.5 <= self.proc_scale < 1.0:
            frame_in = cv2.resize(frame_bgr, None, fx=self.proc_scale, fy=self.proc_scale, interpolation=cv2.INTER_AREA)
        else:
            frame_in = frame_bgr

        rgb = cv2.cvtColor(frame_in, cv2.COLOR_BGR2RGB)
        res = self.hands.process(rgb)
        if not res.multi_hand_landmarks:
            return "invalid", 0.0, 0

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
        return label, conf, 21

    # ──────────────────────────────────────────────────────────────────────────
    # Multi-frame voting
    # ──────────────────────────────────────────────────────────────────────────
    def predict_gesture(self, frames):
        if not frames:
            return "invalid", 0.0, "no-frames"

        thr = self.cfg.get("thresholds", {})
        min_brightness = float(thr.get("min_brightness", 40.0))
        max_motion_last = float(thr.get("max_motion_last", 12.0))
        min_landmarks = int(thr.get("min_landmarks", 15))
        gesture_conf = float(thr.get("gesture_conf", 0.60))

        if self._mean_brightness(frames[-1]) < min_brightness:
            return "invalid", 0.0, "too-dark"
        if self._motion_score(frames, last_n=3) > max_motion_last:
            return "invalid", 0.0, "too-much-motion"

        labels, confs, lm_counts = [], [], []
        for f in frames:
            l, c, k = self._predict_single(f)
            labels.append(l); confs.append(c); lm_counts.append(k)

        if max(lm_counts) < min_landmarks:
            return "invalid", 0.0, "low-occlusion"

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

    # ──────────────────────────────────────────────────────────────────────────
    # Overlay helper (21 pixel coordinates)
    # ──────────────────────────────────────────────────────────────────────────
    def get_hand_points_for_overlay(self, frame_bgr):
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
