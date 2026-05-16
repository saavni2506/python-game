from __future__ import annotations

import json
import time

import numpy as np

from gesture_ml.config import GestureMLConfig


class GestureSampleLogger:
    def __init__(self, config: GestureMLConfig) -> None:
        self.config = config
        self.enabled = bool(config.record_samples and config.record_dir is not None)
        self.path = (
            config.record_dir / f"{config.stream_name}_samples.jsonl"
            if self.enabled and config.record_dir is not None
            else None
        )
        self._last_label: str | None = None
        self._last_write_ts = 0.0

    def maybe_record(self, sequence, movement_state, metadata: dict[str, object] | None = None) -> None:
        if not self.enabled or self.path is None or sequence is None:
            return

        label = str(getattr(movement_state, "gesture", "NONE") or "NONE")
        tracked = bool(getattr(movement_state, "tracked", False))
        if not tracked and label == "NONE":
            return

        now = time.time()
        if self._last_label == label and (now - self._last_write_ts) < self.config.record_interval_seconds:
            return

        payload = {
            "timestamp": now,
            "stream": self.config.stream_name,
            "label": label,
            "tracked": tracked,
            "lane": int(getattr(movement_state, "lane", 1)),
            "jump": bool(getattr(movement_state, "jump", False)),
            "duck": bool(getattr(movement_state, "duck", False)),
            "confidence": float(getattr(movement_state, "confidence", 0.0)),
            "sequence": np.asarray(sequence, dtype=np.float32).round(6).tolist(),
        }
        if metadata:
            payload["metadata"] = metadata

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload))
            handle.write("\n")

        self._last_label = label
        self._last_write_ts = now
