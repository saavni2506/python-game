from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gesture_ml.runtime import DEFAULT_GESTURE_LABELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a small TensorFlow gesture classifier from recorded JSONL sequences.",
    )
    parser.add_argument("--stream", choices=("pose", "hand"), required=True)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("model_artifacts"))
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def load_samples(path: Path) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    if not path.exists():
        raise SystemExit(f"Sample file not found: {path}")

    labels = DEFAULT_GESTURE_LABELS
    label_to_index = {label: index for index, label in enumerate(labels)}
    sequences: list[np.ndarray] = []
    targets: list[int] = []
    expected_shape: tuple[int, int] | None = None

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        sample = json.loads(line)
        label = str(sample.get("label", "NONE")).upper()
        if label not in label_to_index:
            continue

        sequence = np.asarray(sample.get("sequence"), dtype=np.float32)
        if sequence.ndim != 2 or sequence.size == 0:
            continue

        shape = (int(sequence.shape[0]), int(sequence.shape[1]))
        if expected_shape is None:
            expected_shape = shape
        elif shape != expected_shape:
            continue

        sequences.append(sequence)
        targets.append(label_to_index[label])

    if not sequences:
        raise SystemExit("No valid samples were found in the dataset.")

    return np.stack(sequences), np.asarray(targets, dtype=np.int32), labels


def split_dataset(features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    indices = np.arange(features.shape[0])
    rng.shuffle(indices)

    features = features[indices]
    labels = labels[indices]
    if features.shape[0] < 8:
        return (
            features,
            labels,
            np.empty((0,) + features.shape[1:], dtype=features.dtype),
            np.empty(0, dtype=labels.dtype),
        )

    split_index = max(1, int(features.shape[0] * 0.2))
    validation_features = features[:split_index]
    validation_labels = labels[:split_index]
    train_features = features[split_index:]
    train_labels = labels[split_index:]
    return train_features, train_labels, validation_features, validation_labels


def build_model(tf, sequence_length: int, feature_dim: int, class_count: int):
    inputs = tf.keras.Input(shape=(sequence_length, feature_dim), name="gesture_sequence")
    x = tf.keras.layers.Masking(mask_value=0.0)(inputs)
    x = tf.keras.layers.Conv1D(48, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(48))(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(class_count, activation="softmax", name="gesture_logits")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_artifacts(tf, model, output_dir: Path, stream: str, labels: tuple[str, ...]) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    keras_path = output_dir / f"{stream}_gesture_model.keras"
    tflite_path = output_dir / f"{stream}_gesture_model.tflite"
    labels_path = tflite_path.with_suffix(".labels.json")

    model.save(keras_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    tflite_path.write_bytes(tflite_model)
    labels_path.write_text(json.dumps(list(labels), indent=2), encoding="utf-8")
    return keras_path, tflite_path


def main() -> None:
    args = parse_args()
    sample_path = args.input or Path("ml_data") / f"{args.stream}_samples.jsonl"
    features, labels, gesture_labels = load_samples(sample_path)

    try:
        import tensorflow as tf
    except Exception as exc:  # pragma: no cover - environment-dependent
        raise SystemExit(
            "TensorFlow is required to train a gesture model. "
            f"Import failed with: {exc}"
        ) from exc

    train_x, train_y, val_x, val_y = split_dataset(features, labels)
    model = build_model(
        tf=tf,
        sequence_length=int(train_x.shape[1]),
        feature_dim=int(train_x.shape[2]),
        class_count=len(gesture_labels),
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=4,
            restore_best_weights=True,
        )
    ]
    validation_data = (val_x, val_y) if len(val_x) > 0 else None
    if validation_data is None:
        callbacks = []

    model.fit(
        train_x,
        train_y,
        validation_data=validation_data,
        epochs=max(1, args.epochs),
        batch_size=max(1, args.batch_size),
        verbose=2,
        callbacks=callbacks,
    )

    if validation_data is not None:
        loss, accuracy = model.evaluate(val_x, val_y, verbose=0)
        print(f"Validation loss: {loss:.4f}")
        print(f"Validation accuracy: {accuracy:.4f}")

    keras_path, tflite_path = save_artifacts(
        tf=tf,
        model=model,
        output_dir=args.output_dir,
        stream=args.stream,
        labels=gesture_labels,
    )
    print(f"Saved Keras model to {keras_path}")
    print(f"Saved TFLite model to {tflite_path}")


if __name__ == "__main__":
    main()
