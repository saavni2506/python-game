from __future__ import annotations

from dataclasses import dataclass


def _clamp01(value: float) -> float:
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return value


@dataclass(slots=True)
class SessionMetrics:
    combo: int = 0
    calories: float = 0.0
    intensity: float = 0.0
    elapsed_seconds: float = 0.0
    progress: float = 0.0
    _combo_decay_timer: float = 0.0


@dataclass(frozen=True, slots=True)
class SessionHistorySample:
    elapsed_seconds: float
    calories: float
    intensity: float
    progress: float


class GameManager:
    """Tracks gameplay metrics used by the futuristic HUD and summary panel."""

    def __init__(
        self,
        session_target_seconds: float = 180.0,
        history_sample_interval: float = 0.5,
    ) -> None:
        self.session_target_seconds = max(30.0, float(session_target_seconds))
        self.history_sample_interval = max(0.2, float(history_sample_interval))
        self.metrics = SessionMetrics()
        self.session_history: list[SessionHistorySample] = []
        self._history_timer = 0.0
        self.reset_session()

    def reset_session(self) -> None:
        self.metrics = SessionMetrics()
        self.session_history = []
        self._history_timer = 0.0
        self._record_history_sample()

    def update_metrics(
        self,
        dt: float,
        speed: float,
        tracked: bool,
        lane_changed: bool,
        jumped: bool,
        duck_hold: bool,
        coins_gained: int,
    ) -> None:
        metrics = self.metrics
        metrics.elapsed_seconds += dt
        metrics.progress = _clamp01(metrics.elapsed_seconds / self.session_target_seconds)

        # Intensity decays passively and rises from body movement events.
        metrics.intensity = _clamp01(metrics.intensity - (dt * 0.22))
        if tracked:
            metrics.intensity = _clamp01(metrics.intensity + (0.06 * dt))
        if lane_changed:
            metrics.intensity = _clamp01(metrics.intensity + 0.18)
        if jumped:
            metrics.intensity = _clamp01(metrics.intensity + 0.22)
        if duck_hold:
            metrics.intensity = _clamp01(metrics.intensity + (0.08 * dt))

        if coins_gained > 0:
            metrics.combo = min(999, metrics.combo + coins_gained)
            metrics._combo_decay_timer = 0.0
            metrics.calories += coins_gained * 0.22
        else:
            metrics._combo_decay_timer += dt
            if metrics._combo_decay_timer > 2.2:
                metrics.combo = max(0, metrics.combo - 1)
                metrics._combo_decay_timer = 1.7

        # Passive calorie burn scales with movement intensity and pace.
        pace_factor = min(1.0, speed / 16.0)
        burn_rate = 0.06 + (metrics.intensity * 0.22) + (pace_factor * 0.05)
        metrics.calories += burn_rate * dt

        self._history_timer += dt
        if self._history_timer >= self.history_sample_interval or metrics.progress >= 1.0:
            self._record_history_sample()

    def reset_combo(self) -> None:
        self.metrics.combo = 0
        self.metrics._combo_decay_timer = 0.0

    def history_points(self) -> list[SessionHistorySample]:
        points = list(self.session_history)
        current = self._current_history_sample()
        if not points or points[-1] != current:
            points.append(current)
        return points

    def formatted_timer(self) -> str:
        total = max(0, int(self.metrics.elapsed_seconds))
        minutes = total // 60
        seconds = total % 60
        return f"{minutes:02}:{seconds:02}"

    def _current_history_sample(self) -> SessionHistorySample:
        return SessionHistorySample(
            elapsed_seconds=self.metrics.elapsed_seconds,
            calories=self.metrics.calories,
            intensity=self.metrics.intensity,
            progress=self.metrics.progress,
        )

    def _record_history_sample(self) -> None:
        sample = self._current_history_sample()
        if self.session_history and self.session_history[-1] == sample:
            self._history_timer = 0.0
            return
        self.session_history.append(sample)
        self._history_timer = 0.0
