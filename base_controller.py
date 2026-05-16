from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pygame

from config.modes import ModeConfig


@dataclass(slots=True)
class MovementState:
    lane: int = 1
    jump: bool = False
    duck: bool = False
    tracked: bool = False
    message: str = "No gesture detected. Keep your body visible."
    gesture: str = "NONE"
    confidence: float = 0.0
    confidence_reason: str = ""


class BaseController(ABC):
    def __init__(self, mode_config: ModeConfig, camera_index: int = 0) -> None:
        self.mode_config = mode_config
        self.camera_index = camera_index

    @abstractmethod
    def get_movement(self) -> tuple[MovementState, pygame.Surface | None]:
        """Return movement state and a camera preview frame (if available)."""

    @abstractmethod
    def release_resources(self) -> None:
        """Release camera and MediaPipe resources."""

    def get_calibration_sample(self) -> tuple[dict[str, float] | None, str, pygame.Surface | None]:
        """Return one neutral-pose sample with preview frame for calibration."""
        return None, "Calibration sampling not implemented for this controller.", None

    def apply_calibration(self, calibration_data: dict[str, float] | None) -> None:
        """Apply stored calibration values to controller thresholds/baselines."""
        _ = calibration_data
