from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


class CalibrationStore:
    def __init__(self, path: Path | None = None) -> None:
        default_path = Path(__file__).resolve().parents[1] / "config" / "user_calibration.json"
        self.path = path or default_path
        self._cache = self._load()

    def get_mode(self, mode_key: str) -> dict[str, float]:
        mode_data = self._cache.get("modes", {}).get(mode_key, {})
        return {
            key: float(value)
            for key, value in mode_data.items()
            if isinstance(value, (int, float))
        }

    def save_mode(self, mode_key: str, calibration_data: dict[str, float]) -> None:
        payload = {
            key: float(value)
            for key, value in calibration_data.items()
            if isinstance(value, (int, float))
        }
        payload["updated_epoch"] = datetime.now(timezone.utc).timestamp()

        self._cache.setdefault("modes", {})
        self._cache["modes"][mode_key] = payload
        self._write()

    def _load(self) -> dict:
        if not self.path.exists():
            return {"version": 1, "modes": {}}

        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {"version": 1, "modes": {}}

        if not isinstance(data, dict):
            return {"version": 1, "modes": {}}
        data.setdefault("version", 1)
        data.setdefault("modes", {})
        if not isinstance(data["modes"], dict):
            data["modes"] = {}
        return data

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(self._cache, indent=2)
        self.path.write_text(text, encoding="utf-8")

