"""
train.py

Trains an organ-classification model using transfer learning on top of
EfficientNetB0 (pretrained on ImageNet). Two-phase training:
  Phase 1: base frozen, train only the new classification head.
  Phase 2: unfreeze the top of the base and fine-tune at a low learning rate.

Usage:
    python train.py --data_dir ../data/processed --model_out ../model \
        --img_size 224 --batch_size 32 --epochs_head 10 --epochs_finetune 10

Expects data_dir to contain train/, val/, (test/) subfolders, each with one
folder per organ class, e.g.:
    data/processed/train/brain/*.png
    data/processed/train/lungs/*.png
    ...
"""

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.utils.class_weight import compute_class_weight


def build_datasets(data_dir: Path, img_size: int, batch_size: int):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir / "train", image_size=(img_size, img_size),
        batch_size=batch_size, label_mode="int", color_mode="rgb", shuffle=True, seed=42,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir / "val", image_size=(img_size, img_size),
        batch_size=batch_size, label_mode="int", color_mode="rgb", shuffle=False,
    )
    class_names = train_ds.class_names

    # Compute class weights to counter imbalance (e.g. pancreas having far fewer images)
    train_labels = np.concatenate([y.numpy() for _, y in train_ds])
    class_weights_arr = compute_class_weight(
        class_weight="balanced", classes=np.arange(len(class_names)), y=train_labels
    )
    class_weights = {i: w for i, w in enumerate(class_weights_arr)}

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, class_names, class_weights


def build_model(num_classes: int, img_size: int):
    data_augmentation = models.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ], name="augmentation")

    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=(img_size, img_size, 3)
    )
    base_model.trainable = False

    inputs = layers.Input(shape=(img_size, img_size, 3))
    x = data_augmentation(inputs)
    x = tf.keras.applications.efficientnet.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    return model, base_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_out", type=str, required=True)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs_head", type=int, default=10)
    parser.add_argument("--epochs_finetune", type=int, default=10)
    parser.add_argument("--finetune_lr", type=float, default=1e-5)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    model_out = Path(args.model_out)
    model_out.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, class_names, class_weights = build_datasets(
        data_dir, args.img_size, args.batch_size
    )
    print(f"Classes ({len(class_names)}): {class_names}")
    print(f"Class weights: {class_weights}")

    model, base_model = build_model(len(class_names), args.img_size)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(model_out / "best_model.keras"), save_best_only=True, monitor="val_accuracy"
        ),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
    ]

    # --- Phase 1: train the classification head only ---
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    print("\n--- Phase 1: training classification head (base frozen) ---")
    history_head = model.fit(
        train_ds, validation_data=val_ds, epochs=args.epochs_head,
        class_weight=class_weights, callbacks=callbacks,
    )

    # --- Phase 2: unfreeze top of base model and fine-tune ---
    base_model.trainable = True
    fine_tune_at = len(base_model.layers) - 30  # unfreeze last 30 layers only
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.finetune_lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    print("\n--- Phase 2: fine-tuning top layers of base model ---")
    history_finetune = model.fit(
        train_ds, validation_data=val_ds, epochs=args.epochs_finetune,
        class_weight=class_weights, callbacks=callbacks,
    )

    # Save final model, class names, and combined training history
    model.save(model_out / "organ_classifier.keras")
    with open(model_out / "class_names.json", "w") as f:
        json.dump(class_names, f, indent=2)
    with open(model_out / "model_config.json", "w") as f:
        json.dump({"img_size": args.img_size, "class_names": class_names}, f, indent=2)

    combined_history = {}
    for key in history_head.history:
        combined_history[key] = history_head.history[key] + history_finetune.history[key]
    with open(model_out / "training_history.json", "w") as f:
        json.dump(combined_history, f, indent=2)

    print(f"\nModel saved to {model_out / 'organ_classifier.keras'}")
    print(f"Class mapping saved to {model_out / 'class_names.json'}")


if __name__ == "__main__":
    main()
