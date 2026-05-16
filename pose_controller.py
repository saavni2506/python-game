from __future__ import annotations

import math
import time

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
    extract_pose_motion_features,
)


class PoseController(BaseController):
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

        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.pose_landmark = self.mp_pose.PoseLandmark
        self.lighting = LightingNormalizer()
        self._last_ts = time.time()
        self._frame_dt = 1.0 / 30.0

        self._f_nose_y = OneEuroFilter()
        self._f_left_shoulder_x = OneEuroFilter()
        self._f_left_shoulder_y = OneEuroFilter()
        self._f_right_shoulder_x = OneEuroFilter()
        self._f_right_shoulder_y = OneEuroFilter()
        self._f_left_wrist_x = OneEuroFilter()
        self._f_left_wrist_y = OneEuroFilter()
        self._f_right_wrist_x = OneEuroFilter()
        self._f_right_wrist_y = OneEuroFilter()
        self._f_left_hip_x = OneEuroFilter()
        self._f_left_hip_y = OneEuroFilter()
        self._f_right_hip_x = OneEuroFilter()
        self._f_right_hip_y = OneEuroFilter()

        self.baseline_torso_x: float | None = None
        self.baseline_shoulder_y: float | None = None
        self.baseline_left_wrist_y: float | None = None
        self.baseline_right_wrist_y: float | None = None
        self.body_scale: float | None = None
        self.arm_length: float | None = None
        self.dominant_hand_score: float | None = None
        self.last_jump_time = 0.0
        self.smoothed_lane = 1.0
        self.elderly_jump_hold_frames = 0
        self.duck_filter = 0.0
        self.disabled_hand_jump_hold_frames = 0
        self.disabled_hand_jump_armed = True
        self.disabled_hand_duck_filter = 0.0
        self.ml_runtime = GestureMLRuntime.for_stream("pose")
        self.ml_logger = GestureSampleLogger(self.ml_runtime.config)

        self._handlers = {
            "kids": self._handle_kids_profile,
            "elderly": self._handle_elderly_profile,
            "disabled_hand": self._handle_disabled_hand_profile,
        }

        self._kids_required = [
            self.pose_landmark.NOSE.value,
            self.pose_landmark.LEFT_SHOULDER.value,
            self.pose_landmark.RIGHT_SHOULDER.value,
            self.pose_landmark.LEFT_WRIST.value,
            self.pose_landmark.RIGHT_WRIST.value,
            self.pose_landmark.LEFT_HIP.value,
            self.pose_landmark.RIGHT_HIP.value,
        ]
        self._elderly_required = self._kids_required
        self._disabled_hand_required = [
            self.pose_landmark.LEFT_SHOULDER.value,
            self.pose_landmark.RIGHT_SHOULDER.value,
            self.pose_landmark.LEFT_HIP.value,
            self.pose_landmark.RIGHT_HIP.value,
        ]

        self.apply_calibration(calibration_data)

    def get_movement(self) -> tuple[MovementState, pygame.Surface | None]:
        default_state = MovementState(
            message="No pose detected. Stand where shoulders and hips are visible.",
        )
        default_state.confidence_reason = default_state.message

        frame, landmarks, pose_landmarks, default_message = self._read_pose_landmarks()
        if frame is None:
            default_state.message = default_message
            default_state.confidence_reason = default_message
            return default_state, None

        if landmarks is None:
            default_state.message = default_message
            default_state.confidence_reason = default_message
            camera_surface = self._to_pygame_surface(frame)
            return default_state, camera_surface

        self._update_dt()
        handler = self._handlers.get(self.mode_config.gesture_profile, self._handle_kids_profile)
        heuristic_state = handler(landmarks)
        features = extract_pose_motion_features(landmarks)
        prediction = self.ml_runtime.predict(features)
        self.ml_logger.maybe_record(
            self.ml_runtime.latest_sequence(),
            heuristic_state,
            metadata={
                "mode_key": self.mode_config.key,
                "gesture_profile": self.mode_config.gesture_profile,
            },
        )
        movement_state = apply_prediction_to_state(
            heuristic_state,
            prediction,
            override_gameplay=self.ml_runtime.config.override_gameplay,
        )

        self._draw_pose_landmarks(frame, pose_landmarks)
        camera_surface = self._to_pygame_surface(frame)
        return movement_state, camera_surface

    def release_resources(self) -> None:
        if self.cap.isOpened():
            self.cap.release()
        self.pose.close()

    def get_calibration_sample(self) -> tuple[dict[str, float] | None, str, pygame.Surface | None]:
        frame, landmarks, pose_landmarks, message = self._read_pose_landmarks()
        if frame is None:
            return None, message, None

        if landmarks is None:
            return None, "No pose detected. Stand naturally and keep torso visible.", self._to_pygame_surface(frame)

        self._draw_pose_landmarks(frame, pose_landmarks)
        preview = self._to_pygame_surface(frame)
        sample = self._extract_calibration_sample(landmarks)
        if sample is None:
            return None, message, preview
        return sample, "Hold still in a neutral position...", preview

    def apply_calibration(self, calibration_data: dict[str, float] | None) -> None:
        if not calibration_data:
            return
        self.baseline_torso_x = calibration_data.get("pose_baseline_torso_x", self.baseline_torso_x)
        self.baseline_shoulder_y = calibration_data.get("pose_baseline_shoulder_y", self.baseline_shoulder_y)
        self.baseline_left_wrist_y = calibration_data.get("pose_baseline_left_wrist_y", self.baseline_left_wrist_y)
        self.baseline_right_wrist_y = calibration_data.get("pose_baseline_right_wrist_y", self.baseline_right_wrist_y)
        self.body_scale = calibration_data.get("pose_body_scale", self.body_scale)
        self.arm_length = calibration_data.get("pose_arm_length", self.arm_length)
        self.dominant_hand_score = calibration_data.get("pose_dominant_hand_score", self.dominant_hand_score)
        smoothed_lane = calibration_data.get("smoothed_lane")
        if smoothed_lane is not None:
            self.smoothed_lane = float(smoothed_lane)

    def _to_pygame_surface(self, frame_bgr) -> pygame.Surface:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return pygame.image.frombuffer(
            frame_rgb.tobytes(),
            (frame_rgb.shape[1], frame_rgb.shape[0]),
            "RGB",
        ).copy()

    def _read_pose_landmarks(self):
        if not self.cap.isOpened():
            return None, None, None, "Camera unavailable. Keyboard fallback: left/right + up/down."

        ok, frame = self.cap.read()
        if not ok:
            return None, None, None, "Camera frame unavailable. Keyboard fallback: left/right + up/down."

        frame = cv2.flip(frame, 1)
        frame = self.lighting.apply(frame)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        landmarks = results.pose_landmarks.landmark if results.pose_landmarks else None
        return frame, landmarks, results.pose_landmarks, "No pose detected. Stand where shoulders and hips are visible."

    def _draw_pose_landmarks(self, frame, pose_landmarks) -> None:
        if pose_landmarks is None:
            return
        self.mp_draw.draw_landmarks(
            frame,
            pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
        )

    def _extract_calibration_sample(self, landmarks) -> dict[str, float] | None:
        profile = self.mode_config.gesture_profile
        if profile in ("kids", "elderly"):
            required = self._kids_required
            if not self._visibility_ok(landmarks, required):
                return None
            left_wrist = landmarks[self.pose_landmark.LEFT_WRIST.value]
            right_wrist = landmarks[self.pose_landmark.RIGHT_WRIST.value]
        elif profile == "disabled_hand":
            required = self._disabled_hand_required
            if not self._visibility_ok(landmarks, required):
                return None
            left_wrist = None
            right_wrist = None
        else:
            return None

        left_shoulder = landmarks[self.pose_landmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.pose_landmark.RIGHT_SHOULDER.value]
        left_hip = landmarks[self.pose_landmark.LEFT_HIP.value]
        right_hip = landmarks[self.pose_landmark.RIGHT_HIP.value]

        shoulder_mid_x = (left_shoulder.x + right_shoulder.x) * 0.5
        shoulder_mid_y = (left_shoulder.y + right_shoulder.y) * 0.5
        hip_mid_x = (left_hip.x + right_hip.x) * 0.5
        hip_mid_y = (left_hip.y + right_hip.y) * 0.5
        torso_mid_x = (shoulder_mid_x + hip_mid_x) * 0.5
        shoulder_width = abs(right_shoulder.x - left_shoulder.x)
        torso_length = abs(shoulder_mid_y - hip_mid_y)
        sample = {
            "pose_baseline_torso_x": torso_mid_x,
            "pose_baseline_shoulder_y": shoulder_mid_y,
            "smoothed_lane": 1.0,
        }
        if left_wrist is not None and right_wrist is not None:
            sample["pose_baseline_left_wrist_y"] = left_wrist.y
            sample["pose_baseline_right_wrist_y"] = right_wrist.y

        if shoulder_width > 0:
            sample["pose_shoulder_width"] = shoulder_width
        if torso_length > 0:
            sample["pose_torso_length"] = torso_length

        body_scale = self._estimate_body_scale(landmarks, shoulder_mid_y, hip_mid_y, shoulder_width, torso_length)
        if body_scale is not None:
            sample["pose_body_scale"] = body_scale

        arm_length = self._estimate_arm_length(left_shoulder, right_shoulder, left_wrist, right_wrist)
        if arm_length is not None:
            sample["pose_arm_length"] = arm_length

        dominant_score = self._estimate_dominant_hand_score(left_wrist, right_wrist)
        if dominant_score is not None:
            sample["pose_dominant_hand_score"] = dominant_score
        return sample

    def _visibility_ok(
        self,
        landmarks,
        indices: list[int],
        threshold: float = 0.45,
    ) -> bool:
        return all(landmarks[index].visibility > threshold for index in indices)

    def _update_dt(self) -> None:
        now = time.time()
        dt = now - self._last_ts if self._last_ts else (1.0 / 30.0)
        self._last_ts = now
        self._frame_dt = max(1.0 / 120.0, min(1.0 / 10.0, dt))

    def _smooth(self, filt: OneEuroFilter, value: float) -> float:
        return filt.apply(value, self._frame_dt)

    def _smooth_xy(self, landmark, filt_x: OneEuroFilter, filt_y: OneEuroFilter) -> tuple[float, float]:
        return self._smooth(filt_x, landmark.x), self._smooth(filt_y, landmark.y)

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, value))

    def _lane_confidence(self, torso_delta: float, lean_threshold: float, lane: int) -> float:
        if lean_threshold <= 0.0:
            return 0.0
        if lane == 1:
            return self._clamp01(1.0 - (abs(torso_delta) / (lean_threshold * 1.2)))
        return self._clamp01(abs(torso_delta) / (lean_threshold * 1.5))

    def _jump_confidence(
        self,
        left_wrist_y: float,
        right_wrist_y: float,
        left_shoulder_y: float,
        right_shoulder_y: float,
        margin: float,
    ) -> float:
        left_delta = (left_shoulder_y - left_wrist_y) - margin
        right_delta = (right_shoulder_y - right_wrist_y) - margin
        score = min(left_delta, right_delta)
        return self._clamp01(score / max(0.001, margin * 1.5))

    def _duck_confidence(
        self,
        nose_y: float,
        shoulder_mid_y: float,
        bend_margin: float,
        baseline_shoulder_y: float,
        shoulder_drop_margin: float,
    ) -> float:
        down_by_nose = nose_y - (shoulder_mid_y + bend_margin)
        down_by_shoulders = shoulder_mid_y - (baseline_shoulder_y + shoulder_drop_margin)
        score = max(down_by_nose, down_by_shoulders)
        return self._clamp01(score / max(0.001, bend_margin * 1.5))

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

    def _estimate_body_scale(
        self,
        landmarks,
        shoulder_mid_y: float,
        hip_mid_y: float,
        shoulder_width: float,
        torso_length: float,
    ) -> float | None:
        nose = landmarks[self.pose_landmark.NOSE.value]
        left_ankle = landmarks[self.pose_landmark.LEFT_ANKLE.value]
        right_ankle = landmarks[self.pose_landmark.RIGHT_ANKLE.value]

        ankle_y: float | None = None
        ankle_threshold = 0.25
        if left_ankle.visibility > ankle_threshold and right_ankle.visibility > ankle_threshold:
            ankle_y = (left_ankle.y + right_ankle.y) * 0.5
        elif left_ankle.visibility > ankle_threshold:
            ankle_y = left_ankle.y
        elif right_ankle.visibility > ankle_threshold:
            ankle_y = right_ankle.y

        body_height: float | None = None
        if ankle_y is not None and nose.visibility > 0.30:
            body_height = abs(ankle_y - nose.y)

        body_scale: float | None = None
        if body_height is not None and 0.25 < body_height < 0.95:
            body_scale = body_height
        elif torso_length > 0:
            body_scale = torso_length * 2.4

        if body_scale is None:
            return None

        if shoulder_width > 0:
            body_scale = max(body_scale, shoulder_width * 2.0)
        return max(0.30, min(0.90, body_scale))

    def _estimate_arm_length(
        self,
        left_shoulder,
        right_shoulder,
        left_wrist,
        right_wrist,
    ) -> float | None:
        left_arm = None
        right_arm = None
        if left_wrist is not None:
            left_arm = self._distance(left_shoulder, left_wrist)
        if right_wrist is not None:
            right_arm = self._distance(right_shoulder, right_wrist)

        if left_arm is not None and right_arm is not None:
            return (left_arm + right_arm) * 0.5
        return left_arm or right_arm

    @staticmethod
    def _estimate_dominant_hand_score(left_wrist, right_wrist) -> float | None:
        if left_wrist is None and right_wrist is None:
            return None
        if left_wrist is not None and right_wrist is not None:
            return right_wrist.visibility - left_wrist.visibility
        if right_wrist is not None:
            return 1.0
        return -1.0

    @staticmethod
    def _distance(a, b) -> float:
        dx = a.x - b.x
        dy = a.y - b.y
        return math.sqrt((dx * dx) + (dy * dy))

    def _smooth_lane(self, target_lane: int, smoothing: float | None = None) -> int:
        alpha = smoothing if smoothing is not None else self.mode_config.lane_smoothing
        self.smoothed_lane = (self.smoothed_lane * (1.0 - alpha)) + (target_lane * alpha)
        return max(0, min(2, int(round(self.smoothed_lane))))

    def _trigger_jump(self) -> bool:
        now = time.time()
        if (now - self.last_jump_time) >= self.mode_config.jump_cooldown:
            self.last_jump_time = now
            return True
        return False

    def _handle_kids_profile(self, landmarks) -> MovementState:
        if not self._visibility_ok(landmarks, self._kids_required):
            return MovementState(
                message="Kids Mode: keep nose, shoulders, wrists, and hips visible.",
                gesture="NONE",
                confidence=0.0,
                confidence_reason="Pose not visible. Keep nose, shoulders, wrists, and hips in view.",
            )

        nose = landmarks[self.pose_landmark.NOSE.value]
        left_shoulder = landmarks[self.pose_landmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.pose_landmark.RIGHT_SHOULDER.value]
        left_wrist = landmarks[self.pose_landmark.LEFT_WRIST.value]
        right_wrist = landmarks[self.pose_landmark.RIGHT_WRIST.value]
        left_hip = landmarks[self.pose_landmark.LEFT_HIP.value]
        right_hip = landmarks[self.pose_landmark.RIGHT_HIP.value]

        nose_y = self._smooth(self._f_nose_y, nose.y)
        left_shoulder_x, left_shoulder_y = self._smooth_xy(left_shoulder, self._f_left_shoulder_x, self._f_left_shoulder_y)
        right_shoulder_x, right_shoulder_y = self._smooth_xy(right_shoulder, self._f_right_shoulder_x, self._f_right_shoulder_y)
        left_wrist_x, left_wrist_y = self._smooth_xy(left_wrist, self._f_left_wrist_x, self._f_left_wrist_y)
        right_wrist_x, right_wrist_y = self._smooth_xy(right_wrist, self._f_right_wrist_x, self._f_right_wrist_y)
        left_hip_x, left_hip_y = self._smooth_xy(left_hip, self._f_left_hip_x, self._f_left_hip_y)
        right_hip_x, right_hip_y = self._smooth_xy(right_hip, self._f_right_hip_x, self._f_right_hip_y)

        sensitivity = max(0.55, self.mode_config.movement_sensitivity)
        scale = self._body_scale_factor()
        arm_scale = self._arm_scale_factor()
        shoulder_width = abs(right_shoulder_x - left_shoulder_x)
        shoulder_mid_x = (left_shoulder_x + right_shoulder_x) * 0.5
        shoulder_mid_y = (left_shoulder_y + right_shoulder_y) * 0.5
        hip_mid_x = (left_hip_x + right_hip_x) * 0.5
        torso_mid_x = (shoulder_mid_x + hip_mid_x) * 0.5

        if self.baseline_torso_x is None:
            self.baseline_torso_x = torso_mid_x
        if self.baseline_shoulder_y is None:
            self.baseline_shoulder_y = shoulder_mid_y

        # Kids mode expects larger, clear body-lean gestures for lane control.
        lean_threshold = max(0.03, shoulder_width * (0.31 / sensitivity))
        torso_delta = torso_mid_x - self.baseline_torso_x
        target_lane = 1
        if torso_delta < -lean_threshold:
            target_lane = 0
        elif torso_delta > lean_threshold:
            target_lane = 2
        lane = self._smooth_lane(target_lane, smoothing=max(0.24, self.mode_config.lane_smoothing))

        if abs(torso_delta) < (lean_threshold * 0.60):
            self.baseline_torso_x = (self.baseline_torso_x * 0.92) + (torso_mid_x * 0.08)

        # Jump is triggered by raising both hands well above shoulder line.
        hand_raise_margin = (0.055 / sensitivity) * scale * arm_scale
        jump_pose = (
            left_wrist_y < (left_shoulder_y - hand_raise_margin)
            and right_wrist_y < (right_shoulder_y - hand_raise_margin)
        )
        if self.baseline_left_wrist_y is not None and self.baseline_right_wrist_y is not None:
            jump_pose = jump_pose and (
                left_wrist_y < (self.baseline_left_wrist_y - ((0.13 / sensitivity) * scale * arm_scale))
                and right_wrist_y < (self.baseline_right_wrist_y - ((0.13 / sensitivity) * scale * arm_scale))
            )
        jump = jump_pose and self._trigger_jump()

        # Duck is a slight forward bend / downward torso shift.
        bend_margin = (0.078 / sensitivity) * scale
        down_pose = (
            nose_y > (shoulder_mid_y + bend_margin)
            or shoulder_mid_y > (self.baseline_shoulder_y + ((0.05 / sensitivity) * scale))
        )

        if not down_pose:
            self.baseline_shoulder_y = (self.baseline_shoulder_y * 0.96) + (shoulder_mid_y * 0.04)

        lane_confidence = self._lane_confidence(torso_delta, lean_threshold, lane)
        jump_confidence = self._jump_confidence(
            left_wrist_y,
            right_wrist_y,
            left_shoulder_y,
            right_shoulder_y,
            hand_raise_margin,
        )
        duck_confidence = self._duck_confidence(
            nose_y,
            shoulder_mid_y,
            bend_margin,
            self.baseline_shoulder_y,
            (0.05 / sensitivity) * scale,
        )

        gesture = "CENTER"
        confidence = lane_confidence
        if lane == 0:
            gesture = "LEFT"
        elif lane == 2:
            gesture = "RIGHT"

        if jump:
            gesture = "JUMP"
            confidence = jump_confidence
        elif down_pose:
            gesture = "DUCK"
            confidence = duck_confidence

        return MovementState(
            lane=lane,
            jump=jump,
            duck=down_pose,
            tracked=True,
            message="Kids Mode: lean wide to move, both hands high to jump, bend forward to duck.",
            gesture=gesture,
            confidence=confidence,
        )

    def _handle_elderly_profile(self, landmarks) -> MovementState:
        if not self._visibility_ok(landmarks, self._elderly_required):
            return MovementState(
                message="Elderly Mode: keep shoulders, wrists, and hips visible.",
                gesture="NONE",
                confidence=0.0,
                confidence_reason="Pose not visible. Keep shoulders, wrists, and hips in view.",
            )

        nose = landmarks[self.pose_landmark.NOSE.value]
        left_shoulder = landmarks[self.pose_landmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.pose_landmark.RIGHT_SHOULDER.value]
        left_wrist = landmarks[self.pose_landmark.LEFT_WRIST.value]
        right_wrist = landmarks[self.pose_landmark.RIGHT_WRIST.value]

        nose_y = self._smooth(self._f_nose_y, nose.y)
        left_shoulder_x, left_shoulder_y = self._smooth_xy(left_shoulder, self._f_left_shoulder_x, self._f_left_shoulder_y)
        right_shoulder_x, right_shoulder_y = self._smooth_xy(right_shoulder, self._f_right_shoulder_x, self._f_right_shoulder_y)
        left_wrist_x, left_wrist_y = self._smooth_xy(left_wrist, self._f_left_wrist_x, self._f_left_wrist_y)
        right_wrist_x, right_wrist_y = self._smooth_xy(right_wrist, self._f_right_wrist_x, self._f_right_wrist_y)

        sensitivity = max(0.55, self.mode_config.movement_sensitivity)
        scale = self._body_scale_factor()
        arm_scale = self._arm_scale_factor()
        shoulder_mid_x = (left_shoulder_x + right_shoulder_x) * 0.5
        shoulder_mid_y = (left_shoulder_y + right_shoulder_y) * 0.5
        shoulder_width = abs(right_shoulder_x - left_shoulder_x)

        if self.baseline_shoulder_y is None:
            self.baseline_shoulder_y = shoulder_mid_y

        # Elderly mode maps calm two-hand side extensions to lane changes.
        side_threshold = max(0.05, shoulder_width * (0.56 / sensitivity))
        both_left = (
            left_wrist_x < (shoulder_mid_x - side_threshold)
            and right_wrist_x < (shoulder_mid_x - (side_threshold * 0.72))
        )
        both_right = (
            left_wrist_x > (shoulder_mid_x + (side_threshold * 0.72))
            and right_wrist_x > (shoulder_mid_x + side_threshold)
        )

        target_lane = 1
        if both_left:
            target_lane = 0
        elif both_right:
            target_lane = 2
        lane = self._smooth_lane(target_lane, smoothing=min(0.16, self.mode_config.lane_smoothing))

        # Require a stable multi-frame raise for jump to avoid sudden spikes.
        hands_above_head = (
            left_wrist_y < (nose_y - ((0.01 / sensitivity) * scale * arm_scale))
            and right_wrist_y < (nose_y - ((0.01 / sensitivity) * scale * arm_scale))
        )
        if self.baseline_left_wrist_y is not None and self.baseline_right_wrist_y is not None:
            hands_above_head = hands_above_head and (
                left_wrist_y < (self.baseline_left_wrist_y - ((0.09 / sensitivity) * scale * arm_scale))
                and right_wrist_y < (self.baseline_right_wrist_y - ((0.09 / sensitivity) * scale * arm_scale))
            )
        if hands_above_head:
            self.elderly_jump_hold_frames += 1
        else:
            self.elderly_jump_hold_frames = 0

        jump = self.elderly_jump_hold_frames >= 3 and self._trigger_jump()
        if jump:
            self.elderly_jump_hold_frames = 0

        # Gentle forward bend/downward shoulder shift triggers duck.
        forward_bend = nose_y > (shoulder_mid_y + ((0.098 / sensitivity) * scale))
        gentle_shoulder_drop = shoulder_mid_y > (self.baseline_shoulder_y + ((0.038 / sensitivity) * scale))
        duck_pose = forward_bend or gentle_shoulder_drop

        self.duck_filter = (self.duck_filter * 0.82) + (0.18 if duck_pose else 0.0)
        duck = self.duck_filter > 0.48

        if not duck:
            self.baseline_shoulder_y = (self.baseline_shoulder_y * 0.98) + (shoulder_mid_y * 0.02)

        left_margin = min(
            (shoulder_mid_x - side_threshold) - left_wrist_x,
            (shoulder_mid_x - (side_threshold * 0.72)) - right_wrist_x,
        )
        right_margin = min(
            left_wrist_x - (shoulder_mid_x + (side_threshold * 0.72)),
            right_wrist_x - (shoulder_mid_x + side_threshold),
        )
        if lane == 0:
            lane_confidence = self._clamp01(left_margin / max(0.001, side_threshold * 0.8))
        elif lane == 2:
            lane_confidence = self._clamp01(right_margin / max(0.001, side_threshold * 0.8))
        else:
            center_disp = max(abs(left_wrist_x - shoulder_mid_x), abs(right_wrist_x - shoulder_mid_x))
            lane_confidence = self._clamp01(1.0 - (center_disp / max(0.001, side_threshold * 1.2)))
        jump_confidence = self._jump_confidence(
            left_wrist_y,
            right_wrist_y,
            left_shoulder_y,
            right_shoulder_y,
            (0.01 / sensitivity) * scale * arm_scale,
        )
        duck_confidence = self._duck_confidence(
            nose_y,
            shoulder_mid_y,
            (0.098 / sensitivity) * scale,
            self.baseline_shoulder_y,
            (0.038 / sensitivity) * scale,
        )

        gesture = "CENTER"
        confidence = lane_confidence
        if lane == 0:
            gesture = "LEFT"
        elif lane == 2:
            gesture = "RIGHT"

        if jump:
            gesture = "JUMP"
            confidence = jump_confidence
        elif duck:
            gesture = "DUCK"
            confidence = duck_confidence

        return MovementState(
            lane=lane,
            jump=jump,
            duck=duck,
            tracked=True,
            message="Elderly Mode: both hands left/right to move, slow raise above head to jump.",
            gesture=gesture,
            confidence=confidence,
        )

    def _handle_disabled_hand_profile(self, landmarks) -> MovementState:
        left_shoulder = landmarks[self.pose_landmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.pose_landmark.RIGHT_SHOULDER.value]
        left_hip = landmarks[self.pose_landmark.LEFT_HIP.value]
        right_hip = landmarks[self.pose_landmark.RIGHT_HIP.value]

        left_shoulder_x, left_shoulder_y = self._smooth_xy(left_shoulder, self._f_left_shoulder_x, self._f_left_shoulder_y)
        right_shoulder_x, right_shoulder_y = self._smooth_xy(right_shoulder, self._f_right_shoulder_x, self._f_right_shoulder_y)
        left_hip_x, left_hip_y = self._smooth_xy(left_hip, self._f_left_hip_x, self._f_left_hip_y)
        right_hip_x, right_hip_y = self._smooth_xy(right_hip, self._f_right_hip_x, self._f_right_hip_y)

        shoulders_visible = left_shoulder.visibility > 0.30 and right_shoulder.visibility > 0.30
        hips_visible = left_hip.visibility > 0.18 and right_hip.visibility > 0.18
        if not shoulders_visible:
            self.disabled_hand_jump_hold_frames = 0
            return MovementState(
                message="Disabled Hand Mode: keep shoulders visible to control movement.",
                gesture="NONE",
                confidence=0.0,
                confidence_reason="Shoulders not visible. Keep upper body in view.",
            )

        sensitivity = max(0.55, self.mode_config.movement_sensitivity)
        scale = self._body_scale_factor()
        shoulder_mid_x = (left_shoulder_x + right_shoulder_x) * 0.5
        shoulder_mid_y = (left_shoulder_y + right_shoulder_y) * 0.5
        if hips_visible:
            hip_mid_x = (left_hip_x + right_hip_x) * 0.5
            torso_mid_x = (shoulder_mid_x + hip_mid_x) * 0.5
        else:
            torso_mid_x = shoulder_mid_x
        shoulder_width = max(0.08, abs(right_shoulder_x - left_shoulder_x))
        tilt = left_shoulder_y - right_shoulder_y

        if self.baseline_torso_x is None:
            self.baseline_torso_x = torso_mid_x
        if self.baseline_shoulder_y is None:
            self.baseline_shoulder_y = shoulder_mid_y

        # No wrist landmarks: combine shoulder tilt + torso lean for lane selection.
        tilt_norm = tilt / shoulder_width
        torso_lean = torso_mid_x - self.baseline_torso_x
        lateral_signal = torso_lean - (tilt_norm * 0.08)
        lateral_threshold = (0.030 / sensitivity) * scale
        target_lane = 1
        if lateral_signal < -lateral_threshold:
            target_lane = 0
        elif lateral_signal > lateral_threshold:
            target_lane = 2
        lane = self._smooth_lane(target_lane, smoothing=max(0.24, self.mode_config.lane_smoothing))

        # Vertical shoulder baseline movement maps to jump (up) and duck (down).
        upward_shift = self.baseline_shoulder_y - shoulder_mid_y
        downward_shift = shoulder_mid_y - self.baseline_shoulder_y
        jump_threshold = (0.040 / sensitivity) * scale
        jump_release_threshold = jump_threshold * 0.45
        duck_threshold = (0.052 / sensitivity) * scale

        jump_pose = upward_shift > jump_threshold
        if jump_pose and self.disabled_hand_jump_armed:
            self.disabled_hand_jump_hold_frames += 1
        elif not jump_pose:
            self.disabled_hand_jump_hold_frames = 0

        jump = False
        if self.disabled_hand_jump_hold_frames >= 3 and self._trigger_jump():
            jump = True
            self.disabled_hand_jump_armed = False
            self.disabled_hand_jump_hold_frames = 0

        if upward_shift < jump_release_threshold:
            self.disabled_hand_jump_armed = True

        duck_pose = downward_shift > duck_threshold
        self.disabled_hand_duck_filter = (self.disabled_hand_duck_filter * 0.80) + (0.20 if duck_pose else 0.0)
        duck = self.disabled_hand_duck_filter > 0.50

        neutral_window = abs(upward_shift) < (jump_threshold * 0.55) and downward_shift < (duck_threshold * 0.55)
        if neutral_window:
            self.baseline_shoulder_y = (self.baseline_shoulder_y * 0.95) + (shoulder_mid_y * 0.05)
            self.baseline_torso_x = (self.baseline_torso_x * 0.94) + (torso_mid_x * 0.06)

        if lane == 1:
            lane_confidence = self._clamp01(1.0 - (abs(lateral_signal) / max(0.001, lateral_threshold * 1.2)))
        else:
            lane_confidence = self._clamp01(abs(lateral_signal) / max(0.001, lateral_threshold * 1.5))
        jump_confidence = self._clamp01((upward_shift - jump_threshold) / max(0.001, jump_threshold * 1.5))
        duck_confidence = self._clamp01((downward_shift - duck_threshold) / max(0.001, duck_threshold * 1.5))

        gesture = "CENTER"
        confidence = lane_confidence
        if lane == 0:
            gesture = "LEFT"
        elif lane == 2:
            gesture = "RIGHT"

        if jump:
            gesture = "JUMP"
            confidence = jump_confidence
        elif duck:
            gesture = "DUCK"
            confidence = duck_confidence

        return MovementState(
            lane=lane,
            jump=jump,
            duck=duck,
            tracked=True,
            message="Disabled Hand Mode: lean torso/tilt shoulders to move, rise body to jump, small squat to duck.",
            gesture=gesture,
            confidence=confidence,
        )
