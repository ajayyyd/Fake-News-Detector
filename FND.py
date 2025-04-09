import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def clean_text(text):
    text = re.sub(r'\W', ' ', str(text))
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    text = re.sub(r'\^[a-zA-Z]\s+', ' ', text)
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    text = re.sub(r'^b\s+', '', text)
    return text.lower()


@st.cache_data
def load_data():
    fake = pd.read_csv("Fake.csv")
    real = pd.read_csv("True.csv")
    fake['label'] = 0
    real['label'] = 1
    df = pd.concat([fake, real])
    df = df.sample(frac=1).reset_index(drop=True)
    df['text'] = df['text'].apply(clean_text)
    return df

df = load_data()


tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
X = tfidf.fit_transform(df['text']).toarray()
y = df['label']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
accuracy = accuracy_score(y_test, model.predict(X_test))


st.title("Fake News Detector")
st.markdown("Built using Logistic Regression + TF-IDF")

st.write(f"Model Accuracy: **{accuracy * 100:.2f}%**")

user_input = st.text_area("Enter a news article or headline to check if it's real or fake:")

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Enter something first.")
    else:
        cleaned_input = clean_text(user_input)
        vector_input = tfidf.transform([cleaned_input]).toarray()
        prediction = model.predict(vector_input)[0]
        if prediction == 0:
            st.error("This news is FAKE.")
        else:
            st.success("This news is REAL.")

st.caption("Project by Ajay Dhanesh | Powered by scikit-learn + Streamlit")
