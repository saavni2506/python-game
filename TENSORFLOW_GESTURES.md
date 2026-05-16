# TensorFlow Gesture Scaffold

This repository now has an optional `gesture_ml` layer that can sit on top of the existing MediaPipe controllers.

What it adds:

- `gesture_ml/features.py`: converts live pose and hand signals into fixed-length feature vectors.
- `gesture_ml/runtime.py`: buffers feature sequences and runs optional TensorFlow or TFLite inference.
- `gesture_ml/logger.py`: records heuristic gameplay gestures into JSONL datasets for later training.
- `tools/train_gesture_model.py`: trains a small sequence model and exports both `.keras` and `.tflite` artifacts.

Default behavior is unchanged. If you do nothing, the game still runs on the current heuristic rules.

## 1. Record training samples

The easiest path now is just:

```powershell
py main.py
```

This workspace includes a local override at `config/gesture_ml.local.json` that enables recording for both pose and hand streams.

If you want to change that behavior later:

- Edit `config/gesture_ml.local.json` for your personal machine settings.
- Edit `config/gesture_ml.json` for project-wide defaults.

Environment variables still work and override the JSON files when needed.

You can also enable recording manually in PowerShell:

```powershell
$env:GBG_ML_POSE_RECORD_SAMPLES = "1"
$env:GBG_ML_HAND_RECORD_SAMPLES = "1"
$env:GBG_ML_RECORD_DIR = "ml_data"
py main.py
```

This writes files like:

- `ml_data/pose_samples.jsonl`
- `ml_data/hand_samples.jsonl`

The labels come from your current controllers, so the ML model starts by learning the same gestures that already work.

## 2. Train a model

Make sure TensorFlow is installed in the same environment you use for development, then run:

```powershell
py tools/train_gesture_model.py --stream pose
py tools/train_gesture_model.py --stream hand
```

Artifacts are exported to `model_artifacts/` by default.

## 3. Enable inference

To use a trained TFLite model:

```powershell
$env:GBG_ML_POSE_ENABLED = "1"
$env:GBG_ML_POSE_MODEL = "model_artifacts/pose_gesture_model.tflite"
py main.py
```

To let ML override gameplay decisions instead of only observing:

```powershell
$env:GBG_ML_POSE_OVERRIDE = "1"
$env:GBG_ML_HAND_OVERRIDE = "1"
py main.py
```

Helpful env vars:

- `GBG_ML_MIN_CONFIDENCE`
- `GBG_ML_SEQUENCE_LENGTH`
- `GBG_ML_RECORD_INTERVAL`
- `GBG_ML_POSE_MODEL`
- `GBG_ML_HAND_MODEL`

## Recommended rollout

1. Record data with the current heuristic controllers.
2. Train a pose model first, because it covers more of the existing gameplay.
3. Run inference without override and inspect feel/confidence.
4. Turn on override only after the model behaves better than the heuristic path.
