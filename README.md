# Fake-News-Detector

A simple machine learning app that detects whether news content is real or fake using TF-IDF and Logistic Regression.

## How It Works

- Dataset: `Fake.csv` + `True.csv` (Kaggle)
- Cleaned text with regex
- Vectorized using TF-IDF (top 5000 words)
- Model: Logistic Regression
- Accuracy: ~90%
- Deployed with **Streamlit**

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
