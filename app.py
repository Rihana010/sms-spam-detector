from pathlib import Path

import joblib
import streamlit as st

MODEL_PATH = Path(__file__).resolve().parent / "model.joblib"

st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="🛡️",
    layout="centered"
)


@st.cache_resource
def load_model(model_path):
    path = Path(model_path)

    if not path.is_file():
        raise FileNotFoundError("Trained model not found.")

    return joblib.load(path)


# Application header
st.title("SMS Spam Detector")
st.write(
    "Analyze suspicious SMS messages using "
    "machine learning."
)

st.info(
    "Paste an SMS message below to check whether "
    "it looks like spam."
)

# Load trained model safely
try:
    model = load_model(str(MODEL_PATH))

except FileNotFoundError:
    st.error("The trained model is missing.")

    st.markdown(
        """
        ### How to fix this

        Open your project terminal and run:

        ```bash
        python get_data.py
        python train.py
        ```

        This downloads the dataset and trains the model.
        """
    )

    st.stop()
