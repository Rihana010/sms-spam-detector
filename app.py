import joblib
import streamlit as st

@st.cache_resource
def load_model():
    return joblib.load("model.joblib")

model = load_model()

st.title("SMS Spam Detector")
st.write("Paste a message and check whether it looks like spam.")

message = st.text_area("Message", height=150)

if st.button("Check") and message.strip():
    p_spam = model.predict_proba([message])[0][1]
    if p_spam >= 0.5:
        st.error(f"Likely SPAM (confidence {p_spam:.0%})")
    else:
        st.success(f"Likely NOT spam (confidence {1 - p_spam:.0%})")

    with st.expander("Why? Top words that pushed the score"):
        vec = model.named_steps["tfidfvectorizer"]
        clf = model.named_steps["logisticregression"]
        X = vec.transform([message])
        contributions = X.multiply(clf.coef_[0]).toarray()[0]
        names = vec.get_feature_names_out()
        top = contributions.argsort()[::-1][:8]
        for i in top:
            if contributions[i] != 0:
                st.write(f"`{names[i]}`: {contributions[i]:+.2f}")