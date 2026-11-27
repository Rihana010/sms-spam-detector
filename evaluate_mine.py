import sys

import joblib
import pandas as pd
from sklearn.metrics import classification_report

model = joblib.load("model.joblib")
path = sys.argv[1] if len(sys.argv) > 1 else "fresh_test.csv"
df = pd.read_csv(path)
df["y"] = (df["label"] == "spam").astype(int)
df["pred"] = model.predict(df["text"])
df["spam_prob"] = model.predict_proba(df["text"])[:, 1].round(2)

print(df[["label", "pred", "spam_prob", "text"]].to_string())
print(classification_report(df["y"], df["pred"], target_names=["ham", "spam"]))