
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from skimage.feature import hog
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

IMAGE_ROOT = Path(r"D:\Telegram\Re-TypeAware")
CATEGORIES = {"top", "pants", "dress", "skirt", "shoes", "outwear"}
MAX_PER_CLASS = 150


def load_manifest(csv_path):
    df = pd.read_csv(csv_path)

    path_col = next(
        (c for c in df.columns if "path" in c.lower()), None
    )
    label_col = next(
        (c for c in df.columns
         if c.lower() in {"label", "category", "class",
                          "clothing_category"}), None
    )

    if path_col is None:
        raise ValueError(
            f"No image path column found. Columns: {df.columns.tolist()}"
        )

    if label_col is None:
        raise ValueError(
            f"No category column found. Columns: {df.columns.tolist()}"
        )

    df["image_path"] = df[path_col].astype(str)
    df["label"] = df[label_col].astype(str).str.lower().str.strip()
    df = df[df["label"].isin(CATEGORIES)].copy()

    def resolve_path(value):
        path = Path(value)
        if path.is_file():
            return str(path)

        parts = path.parts
        for i, part in enumerate(parts):
            if part.lower() in CATEGORIES:
                candidate = IMAGE_ROOT.joinpath(*parts[i:])
                if candidate.is_file():
                    return str(candidate)

        return str(path)

    df["image_path"] = df["image_path"].apply(resolve_path)
    df = df[df["image_path"].apply(lambda p: Path(p).is_file())]

    return df


def extract_features(image_path):
    with Image.open(image_path) as image:
        image = image.convert("RGB").resize((64, 64))
        rgb = np.asarray(image, dtype=np.float32) / 255.0

    gray = np.dot(rgb, [0.299, 0.587, 0.114])

    texture = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        feature_vector=True
    )

    color_features = []

    for channel in range(3):
        hist, _ = np.histogram(
            rgb[:, :, channel],
            bins=16,
            range=(0, 1),
            density=True
        )
        color_features.extend(hist)

    return np.concatenate(
        [texture, np.asarray(color_features)]
    ).astype(np.float32)


def prepare_features(df):
    features = []
    labels = []
    counts = {}

    for row in df.itertuples(index=False):
        label = row.label

        if counts.get(label, 0) >= MAX_PER_CLASS:
            continue

        try:
            features.append(extract_features(row.image_path))
            labels.append(label)
            counts[label] = counts.get(label, 0) + 1
        except Exception as error:
            print(f"Skipping {row.image_path}: {error}")

    if not features:
        raise ValueError(
            "No images were processed. Check CSV paths and dataset folder."
        )

    print("Processed images by category:", counts)

    return np.asarray(features), np.asarray(labels)


def main():
    train_df = load_manifest(DATA_DIR / "train.csv")
    valid_df = load_manifest(DATA_DIR / "validation.csv")

    print("Training rows available:", len(train_df))
    print("Validation rows available:", len(valid_df))

    X_train, y_train = prepare_features(train_df)
    X_valid, y_valid = prepare_features(valid_df)

    if len(set(y_train)) < 2:
        raise ValueError("Training data must contain at least two categories.")

    model = make_pipeline(
        StandardScaler(),
        LinearSVC(class_weight="balanced", random_state=42)
    )

    print("\nTraining classifier...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_valid)
    accuracy = accuracy_score(y_valid, predictions)

    print(f"\nValidation accuracy: {accuracy:.4f}")
    print("\nClassification report:")
    print(classification_report(
        y_valid,
        predictions,
        zero_division=0
    ))

    output_path = MODEL_DIR / "clothing_classifier.joblib"
    joblib.dump(model, output_path)

    print(f"\nModel saved to: {output_path}")


if __name__ == "__main__":
    main()
