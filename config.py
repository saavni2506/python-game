from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

_TRUTHY_VALUES = {"1", "true", "yes", "on"}


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in _TRUTHY_VALUES


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _resolve_path(value: str | None, root: Path) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def _read_json_file(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _merge_dicts(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_project_settings(root: Path) -> dict:
    config_dir = root / "config"
    shared_path = config_dir / "gesture_ml.json"
    local_path = config_dir / "gesture_ml.local.json"

    settings: dict = {}
    if shared_path.exists():
        settings = _merge_dicts(settings, _read_json_file(shared_path))
    if local_path.exists():
        settings = _merge_dicts(settings, _read_json_file(local_path))
    return settings


def _as_dict(value) -> dict:
    return value if isinstance(value, dict) else {}


def _config_value(global_settings: dict, stream_settings: dict, key: str, default):
    if key in stream_settings:
        return stream_settings[key]
    if key in global_settings:
        return global_settings[key]
    return default


def _bool_from_value(value, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in _TRUTHY_VALUES
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def _float_from_value(value, default: float) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _int_from_value(value, default: int) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


@dataclass(frozen=True, slots=True)
class GestureMLConfig:
    stream_name: str
    enabled: bool = False
    override_gameplay: bool = False
    min_confidence: float = 0.80
    sequence_length: int = 20
    record_samples: bool = False
    record_interval_seconds: float = 0.40
    model_path: Path | None = None
    record_dir: Path | None = None

    @classmethod
    def from_env(
        cls,
        stream_name: str,
        project_root: Path | None = None,
    ) -> "GestureMLConfig":
        root = project_root or Path(__file__).resolve().parents[1]
        prefix = f"GBG_ML_{stream_name.upper()}_"
        settings = _load_project_settings(root)
        global_settings = _as_dict(settings.get("global"))
        stream_settings = _as_dict(_as_dict(settings.get("streams")).get(stream_name))

        model_path = _resolve_path(
            os.getenv(f"{prefix}MODEL")
            or os.getenv("GBG_ML_MODEL")
            or _config_value(global_settings, stream_settings, "model_path", None),
            root,
        )
        record_dir_value = (
            os.getenv(f"{prefix}RECORD_DIR")
            or os.getenv("GBG_ML_RECORD_DIR")
            or _config_value(global_settings, stream_settings, "record_dir", None)
        )
        record_dir = _resolve_path(record_dir_value, root)

        enabled_default = _bool_from_value(
            _config_value(global_settings, stream_settings, "enabled", False),
            False,
        )
        enabled = _get_bool(f"{prefix}ENABLED", _get_bool("GBG_ML_ENABLED", enabled_default))
        enabled = enabled or model_path is not None

        record_default = _bool_from_value(
            _config_value(global_settings, stream_settings, "record_samples", False),
            False,
        )
        record_samples = _get_bool(
            f"{prefix}RECORD_SAMPLES",
            _get_bool("GBG_ML_RECORD_SAMPLES", record_default),
        )
        record_samples = record_samples or record_dir_value is not None
        if record_samples and record_dir is None:
            record_dir = root / "ml_data"

        override_default = _bool_from_value(
            _config_value(global_settings, stream_settings, "override_gameplay", False),
            False,
        )
        override_gameplay = _get_bool(
            f"{prefix}OVERRIDE",
            _get_bool("GBG_ML_OVERRIDE", override_default),
        )
        min_confidence_default = _float_from_value(
            _config_value(global_settings, stream_settings, "min_confidence", 0.80),
            0.80,
        )
        min_confidence = max(
            0.0,
            min(
                1.0,
                _get_float(
                    f"{prefix}MIN_CONFIDENCE",
                    _get_float("GBG_ML_MIN_CONFIDENCE", min_confidence_default),
                ),
            ),
        )
        sequence_length_default = _int_from_value(
            _config_value(global_settings, stream_settings, "sequence_length", 20),
            20,
        )
        sequence_length = max(
            4,
            _get_int(
                f"{prefix}SEQUENCE_LENGTH",
                _get_int("GBG_ML_SEQUENCE_LENGTH", sequence_length_default),
            ),
        )
        record_interval_default = _float_from_value(
            _config_value(global_settings, stream_settings, "record_interval_seconds", 0.40),
            0.40,
        )
        record_interval_seconds = max(
            0.10,
            _get_float(
                f"{prefix}RECORD_INTERVAL",
                _get_float("GBG_ML_RECORD_INTERVAL", record_interval_default),
            ),
        )

        return cls(
            stream_name=stream_name,
            enabled=enabled,
            override_gameplay=override_gameplay,
            min_confidence=min_confidence,
            sequence_length=sequence_length,
            record_samples=record_samples,
            record_interval_seconds=record_interval_seconds,
            model_path=model_path,
            record_dir=record_dir,
        )
