import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
DATA_DIR = Path("data")

def main():
    DATA_DIR.mkdir(exist_ok=True)
    print("Downloading dataset...")
    r = requests.get(URL, timeout=30)
    r.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        z.extractall(DATA_DIR)

    df = pd.read_csv(
        DATA_DIR / "SMSSpamCollection",
        sep = "\t",
        header = None,
        names=["label", "text"],
    )
    print(f"Loaded {len(df)} messages")
    print(df["label"].value_counts())


if __name__ == "__main__":
    main()
    