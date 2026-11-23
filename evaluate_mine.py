import joblib
import pandas as pd
from sklearn.metrics import classification_report

model = joblib.load("model.joblib")
df = pd.read_csv("fresh_test.csv")
df["y"] = (df["label"] == "spam").astype(int)
df["pred"] = model.predict(df["text"])
df["spam_prob"] = model.predict_proba(df["text"])[:, 1].round(2)

print(df[["label", "pred", "spam_prob", "text"]].to_string())
print(classification_report(df["y"], df["pred"], target_names=["ham", "spam"]))