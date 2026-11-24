from pathlib import Path

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline


BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "SMSSpamCollection"
EXTRA_DATA_PATH = BASE_DIR / "extra_train.csv"
MODEL_PATH = BASE_DIR / "model.joblib"

RANDOM_STATE = 42
TEST_SIZE = 0.20
EXTRA_SAMPLE_WEIGHT = 5.0


def load_data():
    """Load and validate the original and additional SMS datasets."""

    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Run python get_data.py first."
        )

    if not EXTRA_DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Additional training data not found: {EXTRA_DATA_PATH}"
        )

    # Load original UCI dataset
    original = pd.read_csv(
        DATA_PATH,
        sep="\t",
        header=None,
        names=["label", "text"],
    )

    original["is_extra"] = False

    # Load additional hand-labelled examples
    extra = pd.read_csv(EXTRA_DATA_PATH)

    required_columns = {"label", "text"}

    if not required_columns.issubset(extra.columns):
        raise ValueError(
            "extra_train.csv must contain 'label' and 'text' columns."
        )

    extra = extra[["label", "text"]].copy()
    extra["is_extra"] = True

    # Combine datasets
    df = pd.concat(
        [original, extra],
        ignore_index=True,
    )

    # Normalize labels
    df["label"] = (
        df["label"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    valid_labels = {"ham", "spam"}

    invalid_labels = set(
        df["label"].dropna().unique()
    ) - valid_labels

    if df["label"].isna().any() or invalid_labels:
        raise ValueError(
            f"Invalid or missing labels found: {invalid_labels}"
        )

    # Clean message text
    df["text"] = (
        df["text"]
        .astype("string")
        .str.strip()
    )

    before_cleaning = len(df)

    df = df.dropna(subset=["text"])
    df = df[df["text"] != ""].copy()

    removed = before_cleaning - len(df)

    if df.empty:
        raise ValueError("No valid SMS messages found.")

    # Convert labels into binary targets
    df["y"] = (df["label"] == "spam").astype(int)

    # Normalize messages for duplicate detection
    df["normalized_text"] = (
        df["text"]
        .str.lower()
        .str.split()
        .str.join(" ")
    )

    # Detect conflicting labels for identical messages
    conflicts = (
        df.groupby("normalized_text")["y"]
        .nunique()
    )

    if (conflicts > 1).any():
        raise ValueError(
            "Identical SMS messages have conflicting labels. "
            "Review the training CSV files."
        )

    # Remove duplicate messages to reduce evaluation leakage.
    # Prefer hand-labelled examples when duplicates exist.
    before_deduplication = len(df)

    df = (
        df.sort_values("is_extra", ascending=False)
        .drop_duplicates(
            subset=["normalized_text"],
            keep="first",
        )
        .reset_index(drop=True)
    )

    duplicates_removed = before_deduplication - len(df)

    # Check whether both classes have sufficient samples
    counts = df["y"].value_counts()

    if len(counts) < 2 or counts.min() < 2:
        raise ValueError(
            "Both ham and spam classes need at least "
            "two valid examples."
        )

    print("\n=== DATASET SUMMARY ===")
    print(f"Total valid messages: {len(df)}")
    print(f"Additional training messages: {df['is_extra'].sum()}")
    print(f"Empty messages removed: {removed}")
    print(f"Duplicate messages removed: {duplicates_removed}")
    print("\nClass distribution:")
    print(df["label"].value_counts())

    return df.drop(columns=["normalized_text"])


def build_model():
    """Create the TF-IDF and Logistic Regression pipeline."""

    return make_pipeline(
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
        ),
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    )


def get_sample_weights(extra_flags):
    """Give additional hand-labelled examples higher training weight."""

    return (
        extra_flags.astype(bool)
        .map({
            True: EXTRA_SAMPLE_WEIGHT,
            False: 1.0,
        })
        .to_numpy(dtype=float)
    )


def main():

    df = load_data()

    # Split data while preserving class proportions
    X_train, X_test, y_train, y_test, extra_train, _ = (
        train_test_split(
            df["text"],
            df["y"],
            df["is_extra"],
            test_size=TEST_SIZE,
            stratify=df["y"],
            random_state=RANDOM_STATE,
        )
    )

    # Train evaluation model
    evaluation_model = build_model()

    evaluation_model.fit(
        X_train,
        y_train,
        logisticregression__sample_weight=(
            get_sample_weights(extra_train)
        ),
    )

    # Evaluate on held-out data
    predictions = evaluation_model.predict(X_test)

    print("\n=== MODEL EVALUATION ===")

    print(
        f"Accuracy: {accuracy_score(y_test, predictions):.4f}"
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["ham", "spam"],
            zero_division=0,
        )
    )

    print("Confusion Matrix (rows=true, columns=predicted):")

    print(
        confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1],
        )
    )

    # Refit final production model using all available training data
    print("\nTraining final model on complete dataset...")

    final_model = build_model()

    final_model.fit(
        df["text"],
        df["y"],
        logisticregression__sample_weight=(
            get_sample_weights(df["is_extra"])
        ),
    )

    # Save model safely
    temporary_path = MODEL_PATH.with_name(
        "model.joblib.tmp"
    )

    joblib.dump(final_model, temporary_path)
    temporary_path.replace(MODEL_PATH)

    print(f"\nFinal model saved to: {MODEL_PATH}")
    print("Training completed successfully.")


if __name__ == "__main__":
    try:
        main()

    except (FileNotFoundError, ValueError) as error:
        print(f"\nTraining error: {error}")

    except Exception as error:
        print(f"\nUnexpected training failure: {error}")
        raise
