import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

DATA_PATH = "data/SMSSpamCollection"
MODEL_PATH = "model.joblib"


def load_data():
    df = pd.read_csv(DATA_PATH, sep="\t", header=None, names=["label", "text"])
    df["is_extra"] = False
    extra = pd.read_csv("extra_train.csv")
    extra["is_extra"] = True
    df = pd.concat([df, extra], ignore_index=True)
    df["y"] = (df["label"] == "spam").astype(int)
    return df


def build_model():
    return make_pipeline(
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),   # single words and word pairs
            min_df=2,             # ignore words that appear only once
            sublinear_tf=True,    # dampen very frequent terms
        ),
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",  # ham outnumbers spam ~6:1
        ),
    )


def main():
    df = load_data()
    X_train, X_test, y_train, y_test, w_train, _ = train_test_split(
        df["text"], df["y"], df["is_extra"],
        test_size=0.2, stratify=df["y"], random_state=42,
    )

    # Your hand-written rows count 5x more than the original dataset rows
    sample_weight = w_train.map({True: 5.0, False: 1.0})

    model = build_model()
    model.fit(X_train, y_train, logisticregression__sample_weight=sample_weight)

    preds = model.predict(X_test)
    print("\n=== Classification report ===")
    print(classification_report(y_test, preds, target_names=["ham", "spam"]))
    print("=== Confusion matrix (rows=true, cols=pred) ===")
    print(confusion_matrix(y_test, preds))

    joblib.dump(model, MODEL_PATH)
    print(f"\nSaved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()