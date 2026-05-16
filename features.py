from __future__ import annotations

from typing import Any

NOSE_INDEX = 0
LEFT_SHOULDER_INDEX = 11
RIGHT_SHOULDER_INDEX = 12
LEFT_WRIST_INDEX = 15
RIGHT_WRIST_INDEX = 16
LEFT_HIP_INDEX = 23
RIGHT_HIP_INDEX = 24
LEFT_ANKLE_INDEX = 27
RIGHT_ANKLE_INDEX = 28

POSE_FEATURE_INDICES = (
    NOSE_INDEX,
    LEFT_SHOULDER_INDEX,
    RIGHT_SHOULDER_INDEX,
    LEFT_WRIST_INDEX,
    RIGHT_WRIST_INDEX,
    LEFT_HIP_INDEX,
    RIGHT_HIP_INDEX,
    LEFT_ANKLE_INDEX,
    RIGHT_ANKLE_INDEX,
)


def _visibility(landmark: Any) -> float:
    return float(getattr(landmark, "visibility", 1.0))


def extract_pose_motion_features(landmarks) -> list[float] | None:
    if landmarks is None or len(landmarks) <= RIGHT_ANKLE_INDEX:
        return None

    left_shoulder = landmarks[LEFT_SHOULDER_INDEX]
    right_shoulder = landmarks[RIGHT_SHOULDER_INDEX]
    left_hip = landmarks[LEFT_HIP_INDEX]
    right_hip = landmarks[RIGHT_HIP_INDEX]
    left_wrist = landmarks[LEFT_WRIST_INDEX]
    right_wrist = landmarks[RIGHT_WRIST_INDEX]
    left_ankle = landmarks[LEFT_ANKLE_INDEX]
    right_ankle = landmarks[RIGHT_ANKLE_INDEX]

    shoulder_mid_x = (float(left_shoulder.x) + float(right_shoulder.x)) * 0.5
    shoulder_mid_y = (float(left_shoulder.y) + float(right_shoulder.y)) * 0.5
    hip_mid_x = (float(left_hip.x) + float(right_hip.x)) * 0.5
    hip_mid_y = (float(left_hip.y) + float(right_hip.y)) * 0.5
    torso_mid_x = (shoulder_mid_x + hip_mid_x) * 0.5
    torso_mid_y = (shoulder_mid_y + hip_mid_y) * 0.5

    shoulder_width = abs(float(right_shoulder.x) - float(left_shoulder.x))
    torso_length = abs(hip_mid_y - shoulder_mid_y)
    scale = max(0.001, shoulder_width, torso_length)

    features: list[float] = []
    for index in POSE_FEATURE_INDICES:
        point = landmarks[index]
        features.extend(
            [
                (float(point.x) - torso_mid_x) / scale,
                (float(point.y) - torso_mid_y) / scale,
                _visibility(point),
            ]
        )

    features.extend(
        [
            shoulder_width / scale,
            torso_length / scale,
            (torso_mid_x - 0.5) / scale,
            (float(left_shoulder.y) - float(right_shoulder.y)) / scale,
            (shoulder_mid_y - float(left_wrist.y)) / scale,
            (shoulder_mid_y - float(right_wrist.y)) / scale,
            (float(right_ankle.x) - float(left_ankle.x)) / scale,
        ]
    )
    return features


def _hand_summary_vector(hand: Any, rest_y: float | None) -> list[float]:
    if hand is None:
        return [0.0] * 7

    wrist_x = float(getattr(hand, "wrist_x", 0.0))
    wrist_y = float(getattr(hand, "wrist_y", 0.0))
    extended_count = float(getattr(hand, "extended_count", 0.0))
    is_open = float(bool(getattr(hand, "is_open", False)))
    is_fist = float(bool(getattr(hand, "is_fist", False)))
    rest_delta = 0.0 if rest_y is None else wrist_y - float(rest_y)

    return [
        1.0,
        wrist_x - 0.5,
        wrist_y,
        extended_count / 5.0,
        is_open,
        is_fist,
        rest_delta,
    ]


def extract_hand_motion_features(
    left_hand: Any,
    right_hand: Any,
    pose_results: Any,
    left_hand_rest_y: float | None = None,
    right_hand_rest_y: float | None = None,
) -> list[float]:
    pose_landmarks = getattr(getattr(pose_results, "pose_landmarks", None), "landmark", None)

    pose_present = 0.0
    shoulder_mid_x = 0.0
    shoulder_mid_y = 0.0
    left_shoulder_y = 0.0
    right_shoulder_y = 0.0
    nose_y = 0.0

    if pose_landmarks is not None and len(pose_landmarks) > RIGHT_SHOULDER_INDEX:
        pose_present = 1.0
        left_shoulder = pose_landmarks[LEFT_SHOULDER_INDEX]
        right_shoulder = pose_landmarks[RIGHT_SHOULDER_INDEX]
        nose = pose_landmarks[NOSE_INDEX]
        shoulder_mid_x = ((float(left_shoulder.x) + float(right_shoulder.x)) * 0.5) - 0.5
        left_shoulder_y = float(left_shoulder.y)
        right_shoulder_y = float(right_shoulder.y)
        shoulder_mid_y = (left_shoulder_y + right_shoulder_y) * 0.5
        nose_y = float(nose.y)

    left_above_shoulder = 0.0
    right_above_shoulder = 0.0
    if pose_present:
        if left_hand is not None:
            left_above_shoulder = float(float(getattr(left_hand, "wrist_y", 1.0)) < left_shoulder_y)
        if right_hand is not None:
            right_above_shoulder = float(float(getattr(right_hand, "wrist_y", 1.0)) < right_shoulder_y)

    features = _hand_summary_vector(left_hand, left_hand_rest_y)
    features.extend(_hand_summary_vector(right_hand, right_hand_rest_y))
    features.extend(
        [
            pose_present,
            shoulder_mid_x,
            shoulder_mid_y,
            left_shoulder_y,
            right_shoulder_y,
            nose_y,
            left_above_shoulder,
            right_above_shoulder,
        ]
    )
    return features
