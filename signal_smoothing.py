from __future__ import annotations

import math


def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + (tau / dt))


class LowPassFilter:
    def __init__(self, alpha: float) -> None:
        self._alpha = alpha
        self._initialized = False
        self._value = 0.0

    def apply(self, x: float, alpha: float | None = None) -> float:
        if alpha is not None:
            self._alpha = alpha
        if not self._initialized:
            self._value = x
            self._initialized = True
            return x
        self._value = (self._alpha * x) + ((1.0 - self._alpha) * self._value)
        return self._value


class OneEuroFilter:
    def __init__(self, min_cutoff: float = 1.2, beta: float = 0.6, d_cutoff: float = 1.0) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x = LowPassFilter(alpha=1.0)
        self._dx = LowPassFilter(alpha=1.0)
        self._last_x: float | None = None

    def apply(self, x: float, dt: float) -> float:
        if dt <= 0.0:
            return x
        if self._last_x is None:
            self._last_x = x
            return self._x.apply(x, alpha=1.0)

        dx = (x - self._last_x) / dt
        self._last_x = x
        edx = self._dx.apply(dx, _alpha(self.d_cutoff, dt))
        cutoff = self.min_cutoff + (self.beta * abs(edx))
        return self._x.apply(x, _alpha(cutoff, dt))
