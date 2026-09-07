"""
evaluate.py

Evaluates a trained organ classifier on the held-out test set: overall
accuracy, per-class precision/recall/F1, and a confusion matrix (saved as
a PNG so you can see exactly which organs get confused with each other).

Usage:
    python evaluate.py --data_dir ../data/processed --model_path ../model/organ_classifier.keras \
        --class_names ../model/class_names.json --img_size 224
"""

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--class_names", type=str, required=True)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--out_dir", type=str, default="../model")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)

    with open(args.class_names) as f:
        class_names = json.load(f)

    test_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir / "test", image_size=(args.img_size, args.img_size),
        batch_size=args.batch_size, label_mode="int", color_mode="rgb", shuffle=False,
    )

    model = tf.keras.models.load_model(args.model_path)

    y_true = np.concatenate([y.numpy() for _, y in test_ds])
    y_pred_probs = model.predict(test_ds)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("\n=== Classification Report ===")
    report = classification_report(y_true, y_pred, target_names=class_names, digits=3)
    print(report)

    overall_accuracy = (y_true == y_pred).mean()
    print(f"Overall test accuracy: {overall_accuracy * 100:.2f}%")

    with open(out_dir / "evaluation_report.txt", "w") as f:
        f.write(f"Overall test accuracy: {overall_accuracy * 100:.2f}%\n\n")
        f.write(report)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix — Overall Accuracy: {overall_accuracy * 100:.2f}%")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=150)
    print(f"\nConfusion matrix saved to {out_dir / 'confusion_matrix.png'}")
    print(f"Full report saved to {out_dir / 'evaluation_report.txt'}")


if __name__ == "__main__":
    main()
