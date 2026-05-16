from __future__ import annotations

import cv2


class LightingNormalizer:
    def __init__(
        self,
        target_luma: float = 0.52,
        min_gain: float = 0.70,
        max_gain: float = 1.45,
    ) -> None:
        self._target_luma = target_luma
        self._min_gain = min_gain
        self._max_gain = max_gain
        self._ema_gain = 1.0
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def apply(self, frame_bgr):
        if frame_bgr is None:
            return frame_bgr

        ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
        y_channel = ycrcb[:, :, 0]
        mean_luma = float(y_channel.mean()) / 255.0
        if mean_luma > 0.01:
            gain = self._target_luma / mean_luma
            gain = max(self._min_gain, min(self._max_gain, gain))
            self._ema_gain = (self._ema_gain * 0.85) + (gain * 0.15)

        balanced = cv2.convertScaleAbs(frame_bgr, alpha=self._ema_gain, beta=0)
        ycrcb = cv2.cvtColor(balanced, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = self._clahe.apply(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
