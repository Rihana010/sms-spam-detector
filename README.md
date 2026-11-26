# SMS Spam Detector


A spam classifier for SMS messages, built to find out where keyword-based models break down on modern scam wording.

## Overview

The model is trained on the UCI SMS Spam Collection (5,572 labeled messages, mostly English, from around 2012) plus about 70 hand-written examples covering UPI, KYC, OTP, and bank-alert scams. It is served through a Streamlit web app that shows the top words driving each prediction.

## Approach

1. **Features:** TF-IDF on unigrams and bigrams (`min_df=2`, `sublinear_tf=True`).
2. **Classifier:** Logistic Regression with `class_weight="balanced"`, since spam is about 13% of the data.
3. **Extra data:** Hand-labeled rows in `extra_train.csv`, weighted 5x during training.
4. **Evaluation:** A separate `my_tests.csv` (26 messages) and a `fresh_test.csv` (10 messages, never used for training).

## Results

| Test set | Accuracy | Notes |
|---|---|---|
| Original held-out split (20%) | (paste from train.py) | Standard benchmark |
| my_tests.csv (26 rows, seen during tuning) | 0.85 | Optimistic: phrasing overlaps with training extras |
| fresh_test.csv (10 rows, unseen) | 0.50 | The honest number |

The gap between the 0.85 and the 0.50 is the main finding. The model learns surface keywords such as "OTP", "account", and "parcel", so it flags legitimate bank and delivery alerts and misses scams that use plain language, like "share the OTP to receive your refund."

## Known failures

- **Missed scams:** "Share OTP to disburse," "Update your PIN to continue," "Work from home, register with a security deposit."
- **False alarms:** "Rs 2,000 debited via NEFT," "Your Swiggy order is out for delivery," "Your account statement is available."

## Setup

```bash
pip install -r requirements.txt
python get_data.py        # downloads the dataset into data/
python train.py           # trains and saves model.joblib
streamlit run app.py      # opens the web app at localhost:8501
```

To run the evaluation on your own messages:

```bash
python evaluate_mine.py
```

## Project structure

```
get_data.py        dataset download
train.py           training, evaluation, model saving
app.py             Streamlit UI with per-prediction word contributions
evaluate_mine.py   scores the model on custom CSV test sets
extra_train.csv    hand-labeled training examples
my_tests.csv       test set used during tuning
fresh_test.csv     unseen test set
embed_model.py     sentence-embedding experiment (see Next steps)
```

## Next steps

- Sentence embeddings (`all-MiniLM-L6-v2`) to capture meaning instead of keywords. The script is written but was not run locally; it needs a PyTorch-capable environment.
- Grow the hand-labeled set and test whether the fresh-set score improves with more diverse wording.
- Compare against Multinomial Naive Bayes on the same splits.

## Limitations

- The dataset is English-only and dated. Hinglish and Kannada-English messages are not represented.
- The hand-written test sets are small (10 to 26 messages), so scores have wide uncertainty.
- The model is not suitable for real filtering decisions.
