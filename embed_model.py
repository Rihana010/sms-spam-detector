import joblib
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

DATA_PATH = "data/SMSSpamCollection"


def load_data():
    df = pd.read_csv(DATA_PATH, sep="\t", header=None, names=["label", "text"])
    extra = pd.read_csv("extra_train.csv")
    df = pd.concat([df, extra], ignore_index=True)
    df["y"] = (df["label"] == "spam").astype(int)
    return df


def main():
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    df = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["y"], test_size=0.2, stratify=df["y"], random_state=42
    )

    emb_train = encoder.encode(X_train.tolist(), show_progress_bar=True)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(emb_train, y_train)

    print(classification_report(y_test, clf.predict(encoder.encode(X_test.tolist())),
                                target_names=["ham", "spam"]))

    fresh = pd.read_csv("fresh_test.csv")
    fresh["y"] = (fresh["label"] == "spam").astype(int)
    fresh_pred = clf.predict(encoder.encode(fresh["text"].tolist()))
    print("=== Fresh set ===")
    print(classification_report(fresh["y"], fresh_pred, target_names=["ham", "spam"]))

    joblib.dump({"encoder_name": "all-MiniLM-L6-v2", "clf": clf}, "embed_model.joblib")


if __name__ == "__main__":
    main()