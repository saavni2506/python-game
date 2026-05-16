from __future__ import annotations

import math
import time
from dataclasses import dataclass

import cv2
import mediapipe as mp
import pygame

from controllers.base_controller import BaseController, MovementState
from core.signal_smoothing import OneEuroFilter
from core.vision_preprocess import LightingNormalizer
from gesture_ml import (
    GestureMLRuntime,
    GestureSampleLogger,
    apply_prediction_to_state,
    extract_hand_motion_features,
)


@dataclass(slots=True)
class HandGestureInfo:
    wrist_x: float
    wrist_y: float
    extended_count: int
    is_open: bool
    is_fist: bool


class HandController(BaseController):
    def __init__(
        self,
        mode_config,
        camera_index: int = 0,
        calibration_data: dict[str, float] | None = None,
    ) -> None:
        super().__init__(mode_config, camera_index)

        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            model_complexity=0,
            max_num_hands=2,
            min_detection_confidence=0.40,
            min_tracking_confidence=0.40,
        )
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            smooth_landmarks=True,
            min_detection_confidence=0.40,
            min_tracking_confidence=0.40,
        )
        self.lighting = LightingNormalizer()
        self._last_ts = time.time()
        self._frame_dt = 1.0 / 30.0

        self._f_left_wrist_x = OneEuroFilter()
        self._f_left_wrist_y = OneEuroFilter()
        self._f_right_wrist_x = OneEuroFilter()
        self._f_right_wrist_y = OneEuroFilter()
        self._f_left_shoulder_y = OneEuroFilter()
        self._f_right_shoulder_y = OneEuroFilter()

        self.last_jump_time = 0.0
        self.left_hand_rest_y: float | None = None
        self.right_hand_rest_y: float | None = None
        self.body_scale: float | None = None
        self.arm_length: float | None = None
        self.dominant_hand_score: float | None = None

        self.left_open_hold_frames = 0
        self.right_open_hold_frames = 0
        self.both_open_hold_frames = 0
        self.both_fist_hold_frames = 0
        self.duck_hold_frames = 0
        self.jump_hold_frames = 0
        self.jump_armed = True
        self.ml_runtime = GestureMLRuntime.for_stream("hand")
        self.ml_logger = GestureSampleLogger(self.ml_runtime.config)
        self.apply_calibration(calibration_data)

    def get_movement(self) -> tuple[MovementState, pygame.Surface | None]:
        state = MovementState(
            message=(
                "Leg-Free: LEFT OPEN=left lane, RIGHT OPEN=right lane, BOTH OPEN=center, "
                "BOTH FIST=duck, hands above shoulders=jump."
            ),
        )

        frame, hand_results, pose_results, status_message = self._read_tracking_results()
        if frame is None:
            state.message = status_message
            state.confidence_reason = status_message
            return state, None

        hands_by_side = self._extract_hands_by_screen_side(hand_results)
        left_hand = hands_by_side.get("left")
        right_hand = hands_by_side.get("right")

        self._update_dt()
        left_hand = self._smooth_hand(left_hand, "left")
        right_hand = self._smooth_hand(right_hand, "right")

        left_open = left_hand is not None and left_hand.is_open
        right_open = right_hand is not None and right_hand.is_open
        left_fist = left_hand is not None and left_hand.is_fist
        right_fist = right_hand is not None and right_hand.is_fist

        only_left_open = left_open and right_hand is None
        only_right_open = right_open and left_hand is None
        both_open = left_open and right_open
        both_fist = left_fist and right_fist
        any_fist = left_fist or right_fist
        low_duck_pose = self._is_low_duck_pose(left_hand, right_hand)
        jump_pose = self._is_jump_pose(left_hand, right_hand, pose_results)

        self.left_open_hold_frames = self._step_hold(self.left_open_hold_frames, only_left_open)
        self.right_open_hold_frames = self._step_hold(self.right_open_hold_frames, only_right_open)
        self.both_open_hold_frames = self._step_hold(self.both_open_hold_frames, both_open)
        self.both_fist_hold_frames = self._step_hold(self.both_fist_hold_frames, both_fist)
        allow_single_fist_duck = (
            self.mode_config.gesture_profile == "disabled_leg"
            and any_fist
            and not (left_open or right_open)
        )
        self.duck_hold_frames = self._step_hold(
            self.duck_hold_frames,
            both_fist or low_duck_pose or allow_single_fist_duck,
        )
        self.jump_hold_frames = self._step_hold(self.jump_hold_frames, jump_pose)

        dominant = self._dominant_hand()
        left_required = 1 if dominant == "left" else 2
        right_required = 1 if dominant == "right" else 2

        if self.jump_hold_frames >= 2:
            state.tracked = True
            state.lane = 1
            if self.jump_armed:
                state.jump = self._trigger_jump()
            if state.jump:
                self.jump_armed = False
            state.message = "JUMP: both wrists above shoulder line."
            state.gesture = "JUMP"
            state.confidence = self._jump_confidence(left_hand, right_hand, pose_results)
        elif self.duck_hold_frames >= 2:
            state.tracked = True
            state.lane = 1
            state.duck = True
            state.message = "DUCK: both fists or lower both hands below rest height."
            state.gesture = "DUCK"
            if both_fist:
                state.confidence = min(self._fist_confidence(left_hand), self._fist_confidence(right_hand))
            elif low_duck_pose:
                state.confidence = self._low_duck_confidence(left_hand, right_hand)
            else:
                state.confidence = max(self._fist_confidence(left_hand), self._fist_confidence(right_hand))
        elif self.left_open_hold_frames >= left_required:
            state.tracked = True
            state.lane = 0
            state.message = "LEFT HAND OPEN detected: move left."
            state.gesture = "LEFT"
            state.confidence = self._open_confidence(left_hand)
        elif self.right_open_hold_frames >= right_required:
            state.tracked = True
            state.lane = 2
            state.message = "RIGHT HAND OPEN detected: move right."
            state.gesture = "RIGHT"
            state.confidence = self._open_confidence(right_hand)
        elif self.both_open_hold_frames >= 2:
            state.tracked = True
            state.lane = 1
            state.message = "BOTH HANDS OPEN detected: center lane."
            state.gesture = "CENTER"
            state.confidence = min(self._open_confidence(left_hand), self._open_confidence(right_hand))
        elif left_hand is not None or right_hand is not None:
            state.message = "Show open palm on left/right side. For jump, raise both wrists above shoulders."
            state.confidence_reason = self._failure_reason(
                left_hand,
                right_hand,
                left_open,
                right_open,
                left_fist,
                right_fist,
                pose_results,
            )
        else:
            state.message = "Leg-Free: show one or both hands clearly in frame."
            state.confidence_reason = "Hands not visible. Keep both hands in frame."

        if not jump_pose:
            self.jump_armed = True

        heuristic_state = state
        features = extract_hand_motion_features(
            left_hand=left_hand,
            right_hand=right_hand,
            pose_results=pose_results,
            left_hand_rest_y=self.left_hand_rest_y,
            right_hand_rest_y=self.right_hand_rest_y,
        )
        prediction = self.ml_runtime.predict(features)
        self.ml_logger.maybe_record(
            self.ml_runtime.latest_sequence(),
            heuristic_state,
            metadata={
                "mode_key": self.mode_config.key,
                "gesture_profile": self.mode_config.gesture_profile,
            },
        )
        state = apply_prediction_to_state(
            heuristic_state,
            prediction,
            override_gameplay=self.ml_runtime.config.override_gameplay,
        )

        camera_surface = self._to_pygame_surface(frame)
        return state, camera_surface

    def release_resources(self) -> None:
        if self.cap.isOpened():
            self.cap.release()
        self.hands.close()
        self.pose.close()

    def get_calibration_sample(self) -> tuple[dict[str, float] | None, str, pygame.Surface | None]:
        frame, hand_results, pose_results, status_message = self._read_tracking_results()
        if frame is None:
            return None, status_message, None

        hands = self._extract_hands_by_screen_side(hand_results)
        self._update_dt()
        left_hand = self._smooth_hand(hands.get("left"), "left")
        right_hand = self._smooth_hand(hands.get("right"), "right")

        preview = self._to_pygame_surface(frame)
        if left_hand is None or right_hand is None:
            return None, "Show both hands at comfortable neutral height.", preview

        sample: dict[str, float] = {
            "left_hand_rest_y": left_hand.wrist_y,
            "right_hand_rest_y": right_hand.wrist_y,
        }
        sample.update(self._extract_pose_metrics(pose_results))
        return sample, "Hold your hands steady...", preview

    def apply_calibration(self, calibration_data: dict[str, float] | None) -> None:
        if not calibration_data:
            return
        left = calibration_data.get("left_hand_rest_y", self.left_hand_rest_y)
        right = calibration_data.get("right_hand_rest_y", self.right_hand_rest_y)
        if left is not None:
            self.left_hand_rest_y = min(0.82, max(0.36, float(left)))
        if right is not None:
            self.right_hand_rest_y = min(0.82, max(0.36, float(right)))
        self.body_scale = calibration_data.get("pose_body_scale", self.body_scale)
        self.arm_length = calibration_data.get("pose_arm_length", self.arm_length)
        self.dominant_hand_score = calibration_data.get("pose_dominant_hand_score", self.dominant_hand_score)

    def _trigger_jump(self) -> bool:
        now = time.time()
        if (now - self.last_jump_time) >= self.mode_config.jump_cooldown:
            self.last_jump_time = now
            return True
        return False

    def _read_tracking_results(self):
        if not self.cap.isOpened():
            return None, None, None, "Camera unavailable. Keyboard fallback: left/right + up/down."

        ok, frame = self.cap.read()
        if not ok:
            return None, None, None, "Camera frame unavailable. Keyboard fallback: left/right + up/down."

        frame = cv2.flip(frame, 1)
        frame = self.lighting.apply(frame)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = self.hands.process(rgb)
        pose_results = self.pose.process(rgb)

        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                )

        return frame, hand_results, pose_results, "Show one or both hands in frame."

    def _extract_hands_by_screen_side(self, hand_results) -> dict[str, HandGestureInfo]:
        if hand_results is None or not hand_results.multi_hand_landmarks:
            return {}

        by_side: dict[str, tuple[float, HandGestureInfo]] = {}
        for hand_landmarks in hand_results.multi_hand_landmarks:
            info = self._summarize_hand(hand_landmarks)
            side = "left" if info.wrist_x < 0.50 else "right"
            side_center_x = 0.25 if side == "left" else 0.75
            closeness = abs(info.wrist_x - side_center_x)

            existing = by_side.get(side)
            if existing is None:
                by_side[side] = (closeness, info)
                continue

            # Prefer clearer hand posture and then better side placement.
            existing_info = existing[1]
            better_posture = info.extended_count > existing_info.extended_count
            better_side_fit = closeness < existing[0]
            if better_posture or better_side_fit:
                by_side[side] = (closeness, info)

        return {side: entry[1] for side, entry in by_side.items()}

    def _summarize_hand(self, hand_landmarks) -> HandGestureInfo:
        landmarks = hand_landmarks.landmark
        wrist = landmarks[self.mp_hands.HandLandmark.WRIST]

        extended_count = 0
        finger_pairs = (
            (self.mp_hands.HandLandmark.INDEX_FINGER_TIP, self.mp_hands.HandLandmark.INDEX_FINGER_PIP),
            (self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP, self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP),
            (self.mp_hands.HandLandmark.RING_FINGER_TIP, self.mp_hands.HandLandmark.RING_FINGER_PIP),
            (self.mp_hands.HandLandmark.PINKY_TIP, self.mp_hands.HandLandmark.PINKY_PIP),
        )

        for tip_lm, pip_lm in finger_pairs:
            tip = landmarks[tip_lm]
            pip = landmarks[pip_lm]
            if self._distance_sq(tip, wrist) > (self._distance_sq(pip, wrist) * 1.14):
                extended_count += 1

        thumb_tip = landmarks[self.mp_hands.HandLandmark.THUMB_TIP]
        thumb_ip = landmarks[self.mp_hands.HandLandmark.THUMB_IP]
        if self._distance_sq(thumb_tip, wrist) > (self._distance_sq(thumb_ip, wrist) * 1.12):
            extended_count += 1

        is_open = extended_count >= 3
        is_fist = extended_count <= 2

        return HandGestureInfo(
            wrist_x=wrist.x,
            wrist_y=wrist.y,
            extended_count=extended_count,
            is_open=is_open,
            is_fist=is_fist,
        )

    def _is_jump_pose(self, left_hand: HandGestureInfo | None, right_hand: HandGestureInfo | None, pose_results) -> bool:
        if left_hand is None or right_hand is None:
            return False
        if pose_results is None or pose_results.pose_landmarks is None:
            return False

        landmarks = pose_results.pose_landmarks.landmark
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        if left_shoulder.visibility < 0.20 or right_shoulder.visibility < 0.20:
            return False

        left_shoulder_y = self._smooth(self._f_left_shoulder_y, left_shoulder.y)
        right_shoulder_y = self._smooth(self._f_right_shoulder_y, right_shoulder.y)
        margin = 0.012 * self._body_scale_factor() * self._arm_scale_factor()
        return (
            left_hand.wrist_y < (left_shoulder_y - margin)
            and right_hand.wrist_y < (right_shoulder_y - margin)
        )

    @staticmethod
    def _step_hold(current: int, condition: bool) -> int:
        if condition:
            return min(8, current + 1)
        return max(0, current - 1)

    def _body_scale_factor(self) -> float:
        if self.body_scale is None:
            return 1.0
        default_scale = 0.48
        factor = self.body_scale / default_scale
        return max(0.75, min(1.35, factor))

    def _arm_scale_factor(self) -> float:
        if self.arm_length is None:
            return 1.0
        default_arm = 0.27
        factor = self.arm_length / default_arm
        return max(0.80, min(1.25, factor))

    def _update_dt(self) -> None:
        now = time.time()
        dt = now - self._last_ts if self._last_ts else (1.0 / 30.0)
        self._last_ts = now
        self._frame_dt = max(1.0 / 120.0, min(1.0 / 10.0, dt))

    def _smooth(self, filt: OneEuroFilter, value: float) -> float:
        return filt.apply(value, self._frame_dt)

    def _smooth_hand(self, hand: HandGestureInfo | None, side: str) -> HandGestureInfo | None:
        if hand is None:
            return None
        if side == "left":
            fx = self._f_left_wrist_x
            fy = self._f_left_wrist_y
        else:
            fx = self._f_right_wrist_x
            fy = self._f_right_wrist_y
        return HandGestureInfo(
            wrist_x=self._smooth(fx, hand.wrist_x),
            wrist_y=self._smooth(fy, hand.wrist_y),
            extended_count=hand.extended_count,
            is_open=hand.is_open,
            is_fist=hand.is_fist,
        )

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, value))

    def _open_confidence(self, hand: HandGestureInfo | None) -> float:
        if hand is None:
            return 0.0
        return self._clamp01((hand.extended_count - 2) / 3.0)

    def _fist_confidence(self, hand: HandGestureInfo | None) -> float:
        if hand is None:
            return 0.0
        return self._clamp01((3 - hand.extended_count) / 3.0)

    def _jump_confidence(
        self,
        left_hand: HandGestureInfo | None,
        right_hand: HandGestureInfo | None,
        pose_results,
    ) -> float:
        if left_hand is None or right_hand is None:
            return 0.0
        if pose_results is None or pose_results.pose_landmarks is None:
            return 0.0

        landmarks = pose_results.pose_landmarks.landmark
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        if left_shoulder.visibility < 0.20 or right_shoulder.visibility < 0.20:
            return 0.0

        left_shoulder_y = self._smooth(self._f_left_shoulder_y, left_shoulder.y)
        right_shoulder_y = self._smooth(self._f_right_shoulder_y, right_shoulder.y)
        margin = 0.012 * self._body_scale_factor() * self._arm_scale_factor()
        left_delta = (left_shoulder_y - left_hand.wrist_y) - margin
        right_delta = (right_shoulder_y - right_hand.wrist_y) - margin
        score = min(left_delta, right_delta)
        return self._clamp01(score / max(0.001, margin * 1.5))

    def _low_duck_confidence(self, left_hand: HandGestureInfo | None, right_hand: HandGestureInfo | None) -> float:
        if left_hand is None or right_hand is None:
            return 0.0
        if self.left_hand_rest_y is None or self.right_hand_rest_y is None:
            return 0.0
        margin = 0.08 * self._body_scale_factor() * self._arm_scale_factor()
        left_delta = left_hand.wrist_y - (self.left_hand_rest_y + margin)
        right_delta = right_hand.wrist_y - (self.right_hand_rest_y + margin)
        score = min(left_delta, right_delta)
        return self._clamp01(score / max(0.001, margin * 1.5))

    def _failure_reason(
        self,
        left_hand: HandGestureInfo | None,
        right_hand: HandGestureInfo | None,
        left_open: bool,
        right_open: bool,
        left_fist: bool,
        right_fist: bool,
        pose_results,
    ) -> str:
        if left_hand is None and right_hand is None:
            return "Hands not visible. Keep both hands in frame."
        if left_hand is None or right_hand is None:
            return "Show both hands for jump or duck."
        if pose_results is None or pose_results.pose_landmarks is None:
            return "Keep shoulders visible for jump."
        if not (left_open or right_open or left_fist or right_fist):
            return "Open palm to steer; fist or lower hands to duck."
        return "Hold a clear gesture for two frames."

    def _dominant_hand(self) -> str | None:
        if self.dominant_hand_score is None:
            return None
        if self.dominant_hand_score > 0.12:
            return "right"
        if self.dominant_hand_score < -0.12:
            return "left"
        return None

    def _is_low_duck_pose(self, left_hand: HandGestureInfo | None, right_hand: HandGestureInfo | None) -> bool:
        if self.mode_config.gesture_profile != "disabled_leg":
            return False
        if left_hand is None or right_hand is None:
            return False
        if self.left_hand_rest_y is None or self.right_hand_rest_y is None:
            return False

        margin = 0.08 * self._body_scale_factor() * self._arm_scale_factor()
        left_threshold = min(0.98, self.left_hand_rest_y + margin)
        right_threshold = min(0.98, self.right_hand_rest_y + margin)
        return left_hand.wrist_y > left_threshold and right_hand.wrist_y > right_threshold

    def _extract_pose_metrics(self, pose_results) -> dict[str, float]:
        if pose_results is None or pose_results.pose_landmarks is None:
            return {}

        landmarks = pose_results.pose_landmarks.landmark
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]

        shoulder_mid_y = (left_shoulder.y + right_shoulder.y) * 0.5
        hip_mid_y = (left_hip.y + right_hip.y) * 0.5
        shoulder_width = abs(right_shoulder.x - left_shoulder.x)
        torso_length = abs(shoulder_mid_y - hip_mid_y)

        metrics: dict[str, float] = {}
        if shoulder_width > 0:
            metrics["pose_shoulder_width"] = shoulder_width
        if torso_length > 0:
            metrics["pose_torso_length"] = torso_length

        nose = landmarks[self.mp_pose.PoseLandmark.NOSE.value]
        left_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        right_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        ankle_y = None
        ankle_threshold = 0.25
        if left_ankle.visibility > ankle_threshold and right_ankle.visibility > ankle_threshold:
            ankle_y = (left_ankle.y + right_ankle.y) * 0.5
        elif left_ankle.visibility > ankle_threshold:
            ankle_y = left_ankle.y
        elif right_ankle.visibility > ankle_threshold:
            ankle_y = right_ankle.y

        body_height = None
        if ankle_y is not None and nose.visibility > 0.30:
            body_height = abs(ankle_y - nose.y)

        body_scale = None
        if body_height is not None and 0.25 < body_height < 0.95:
            body_scale = body_height
        elif torso_length > 0:
            body_scale = torso_length * 2.4

        if body_scale is not None:
            if shoulder_width > 0:
                body_scale = max(body_scale, shoulder_width * 2.0)
            metrics["pose_body_scale"] = max(0.30, min(0.90, body_scale))

        left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
        right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
        wrist_threshold = 0.25
        left_arm = None
        right_arm = None
        if left_wrist.visibility > wrist_threshold:
            left_arm = self._distance(left_shoulder, left_wrist)
        if right_wrist.visibility > wrist_threshold:
            right_arm = self._distance(right_shoulder, right_wrist)

        if left_arm is not None and right_arm is not None:
            metrics["pose_arm_length"] = (left_arm + right_arm) * 0.5
        elif left_arm is not None:
            metrics["pose_arm_length"] = left_arm
        elif right_arm is not None:
            metrics["pose_arm_length"] = right_arm

        left_vis = left_wrist.visibility if left_wrist.visibility > wrist_threshold else 0.0
        right_vis = right_wrist.visibility if right_wrist.visibility > wrist_threshold else 0.0
        if left_vis > 0.0 and right_vis > 0.0:
            metrics["pose_dominant_hand_score"] = right_vis - left_vis
        elif right_vis > 0.0:
            metrics["pose_dominant_hand_score"] = 1.0
        elif left_vis > 0.0:
            metrics["pose_dominant_hand_score"] = -1.0

        return metrics

    @staticmethod
    def _distance(a, b) -> float:
        dx = a.x - b.x
        dy = a.y - b.y
        return math.sqrt((dx * dx) + (dy * dy))

    @staticmethod
    def _distance_sq(a, b) -> float:
        dx = a.x - b.x
        dy = a.y - b.y
        return (dx * dx) + (dy * dy)

    @staticmethod
    def _to_pygame_surface(frame_bgr) -> pygame.Surface:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return pygame.image.frombuffer(
            frame_rgb.tobytes(),
            (frame_rgb.shape[1], frame_rgb.shape[0]),
            "RGB",
        ).copy()
