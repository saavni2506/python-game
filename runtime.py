from __future__ import annotations

import importlib.util
import json
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np

from gesture_ml.config import GestureMLConfig

DEFAULT_GESTURE_LABELS = ("CENTER", "LEFT", "RIGHT", "JUMP", "DUCK", "NONE")


@dataclass(frozen=True, slots=True)
class GesturePrediction:
    label: str
    confidence: float
    source: str = "ml"
    reason: str = ""


class MotionSequenceBuffer:
    def __init__(self, sequence_length: int) -> None:
        self.sequence_length = max(1, int(sequence_length))
        self.feature_dim: int | None = None
        self._values: deque[np.ndarray] = deque(maxlen=self.sequence_length)

    def append(self, features: Sequence[float] | None) -> None:
        if features is None:
            return

        vector = np.asarray(features, dtype=np.float32).reshape(-1)
        if vector.size == 0:
            return

        if self.feature_dim is None:
            self.feature_dim = int(vector.size)
        elif int(vector.size) != self.feature_dim:
            self.clear()
            self.feature_dim = int(vector.size)

        self._values.append(vector)

    def clear(self) -> None:
        self._values.clear()

    def ready(self, minimum_frames: int | None = None) -> bool:
        required = minimum_frames or self.sequence_length
        return len(self._values) >= required

    def snapshot(self, target_length: int | None = None) -> np.ndarray | None:
        if self.feature_dim is None:
            return None

        length = max(1, int(target_length or self.sequence_length))
        output = np.zeros((length, self.feature_dim), dtype=np.float32)
        if not self._values:
            return output

        values = list(self._values)[-length:]
        output[-len(values) :, :] = np.vstack(values)
        return output

    def __len__(self) -> int:
        return len(self._values)


class GestureMLRuntime:
    def __init__(self, config: GestureMLConfig) -> None:
        self.config = config
        self.buffer = MotionSequenceBuffer(config.sequence_length)
        self.model_path = config.model_path
        self.labels = self._load_labels(self.model_path)
        self.status = "disabled"
        self._backend_name: str | None = None
        self._tf = None
        self._interpreter = None
        self._input_details = None
        self._output_details = None
        self._keras_model = None
        self._load_backend()

    @classmethod
    def for_stream(cls, stream_name: str) -> "GestureMLRuntime":
        return cls(GestureMLConfig.from_env(stream_name))

    def latest_sequence(self) -> np.ndarray | None:
        return self.buffer.snapshot(self.required_sequence_length())

    def predict(self, features: Sequence[float] | None) -> GesturePrediction | None:
        self.buffer.append(features)
        if self._backend_name is None:
            return None

        required_length = self.required_sequence_length()
        if not self.buffer.ready(required_length):
            return None

        sequence = self.buffer.snapshot(required_length)
        if sequence is None:
            return None

        raw_scores = self._run_inference(sequence)
        if raw_scores is None:
            return None

        scores = self._normalize_scores(raw_scores)
        if scores is None or scores.size == 0:
            return None

        label_count = min(scores.size, len(self.labels))
        if label_count <= 0:
            return None

        scores = scores[:label_count]
        labels = self.labels[:label_count]
        best_index = int(np.argmax(scores))
        confidence = float(scores[best_index])
        if confidence < self.config.min_confidence:
            return None

        label = labels[best_index]
        reason = f"{self._backend_name} predicted {label.lower()} at {confidence:.0%}"
        return GesturePrediction(
            label=label,
            confidence=confidence,
            source=self._backend_name or "ml",
            reason=reason,
        )

    def required_sequence_length(self) -> int:
        default_length = self.config.sequence_length
        if self._backend_name == "keras" and self._keras_model is not None:
            input_shape = getattr(self._keras_model, "input_shape", None)
            if isinstance(input_shape, (list, tuple)) and len(input_shape) >= 3:
                length = input_shape[1]
                if isinstance(length, int) and length > 0:
                    return length

        if self._backend_name == "tflite" and self._input_details:
            details = self._input_details[0]
            shape = details.get("shape_signature") or details.get("shape")
            if shape is not None and len(shape) >= 3:
                length = int(shape[1])
                if length > 0:
                    return length

        return default_length

    def _load_backend(self) -> None:
        if not self.config.enabled:
            self.status = "disabled by config"
            return
        if self.model_path is None:
            self.status = "enabled, waiting for a model path"
            return
        if not self.model_path.exists():
            self.status = f"model not found: {self.model_path}"
            return
        if importlib.util.find_spec("tensorflow") is None:
            self.status = "TensorFlow is not installed"
            return

        try:
            import tensorflow as tf
        except Exception as exc:  # pragma: no cover - environment-dependent
            self.status = f"TensorFlow import failed: {exc}"
            return

        self._tf = tf

        try:
            if self.model_path.suffix.lower() == ".tflite":
                interpreter = tf.lite.Interpreter(model_path=str(self.model_path))
                interpreter.allocate_tensors()
                self._interpreter = interpreter
                self._input_details = interpreter.get_input_details()
                self._output_details = interpreter.get_output_details()
                self._backend_name = "tflite"
                self.status = f"loaded {self.model_path.name}"
                return

            self._keras_model = tf.keras.models.load_model(self.model_path)
            self._backend_name = "keras"
            self.status = f"loaded {self.model_path.name}"
        except Exception as exc:  # pragma: no cover - environment-dependent
            self.status = f"model load failed: {exc}"

    def _run_inference(self, sequence: np.ndarray) -> np.ndarray | None:
        batch = np.expand_dims(sequence, axis=0).astype(np.float32)

        if self._backend_name == "keras" and self._keras_model is not None:
            outputs = self._keras_model(batch, training=False)
            if hasattr(outputs, "numpy"):
                return np.asarray(outputs.numpy(), dtype=np.float32)
            return np.asarray(outputs, dtype=np.float32)

        if self._backend_name == "tflite" and self._interpreter is not None and self._input_details:
            input_details = self._input_details[0]
            expected_shape = tuple(int(dim) for dim in input_details.get("shape", batch.shape))
            shape_signature = tuple(int(dim) for dim in input_details.get("shape_signature", expected_shape))
            if batch.shape != expected_shape and -1 in shape_signature:
                self._interpreter.resize_tensor_input(
                    int(input_details["index"]),
                    list(batch.shape),
                    strict=False,
                )
                self._interpreter.allocate_tensors()
                self._input_details = self._interpreter.get_input_details()
                self._output_details = self._interpreter.get_output_details()
                input_details = self._input_details[0]

            dtype = input_details.get("dtype", np.float32)
            self._interpreter.set_tensor(int(input_details["index"]), batch.astype(dtype))
            self._interpreter.invoke()
            output_details = self._output_details[0]
            return np.asarray(
                self._interpreter.get_tensor(int(output_details["index"])),
                dtype=np.float32,
            )

        return None

    @staticmethod
    def _normalize_scores(raw_scores: np.ndarray) -> np.ndarray | None:
        scores = np.asarray(raw_scores, dtype=np.float32).reshape(-1)
        if scores.size == 0:
            return None

        total = float(scores.sum())
        within_bounds = bool(np.all(scores >= 0.0) and np.all(scores <= 1.0))
        if within_bounds and abs(total - 1.0) <= 0.15:
            return scores

        shifted = scores - float(np.max(scores))
        exp_scores = np.exp(shifted)
        denom = float(exp_scores.sum())
        if denom <= 0.0:
            return None
        return exp_scores / denom

    @staticmethod
    def _load_labels(model_path: Path | None) -> tuple[str, ...]:
        if model_path is None:
            return DEFAULT_GESTURE_LABELS

        candidates = (
            model_path.with_suffix(".labels.json"),
            model_path.parent / "labels.json",
        )
        for path in candidates:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(data, list) and data and all(isinstance(item, str) for item in data):
                return tuple(data)
        return DEFAULT_GESTURE_LABELS


def apply_prediction_to_state(base_state, prediction: GesturePrediction | None, override_gameplay: bool = False):
    if prediction is None:
        return base_state

    state = replace(base_state)
    state.confidence = max(float(getattr(state, "confidence", 0.0)), prediction.confidence)

    if getattr(state, "confidence_reason", ""):
        state.confidence_reason = f"{state.confidence_reason} | {prediction.reason}"
    else:
        state.confidence_reason = prediction.reason

    if not override_gameplay:
        return state

    label = prediction.label.upper()
    state.jump = False
    state.duck = False
    state.lane = 1
    state.tracked = label != "NONE"
    state.gesture = label

    if label == "LEFT":
        state.lane = 0
        state.message = "ML LEFT detected."
    elif label == "RIGHT":
        state.lane = 2
        state.message = "ML RIGHT detected."
    elif label == "JUMP":
        state.jump = True
        state.message = "ML JUMP detected."
    elif label == "DUCK":
        state.duck = True
        state.message = "ML DUCK detected."
    elif label == "CENTER":
        state.message = "ML CENTER detected."
    else:
        state.tracked = False
        state.gesture = "NONE"
        state.message = "ML could not confirm a gesture yet."

    return state
