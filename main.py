import streamlit as st
import joblib
from app.utils.text_cleaning import clean_text

# Load saved model and vectorizer
model = joblib.load('app/model/fake_news_model.pkl')
vectorizer = joblib.load('app/model/tfidf_vectorizer.pkl')

st.title("📰 Fake News Detection App")
st.subheader("Paste a news article and check if it's real or fake")

user_input = st.text_area("Enter news article text:")

if st.button("Check"):
    cleaned_text = clean_text(user_input)
    vectorized_text = vectorizer.transform([cleaned_text])
    prediction = model.predict(vectorized_text)[0]
    
    if prediction == 1:
        st.success("✅ Real News")
    else:
        st.error("🚫 Fake News")
col1, col2 = st.columns(2)

with col1:
    st.header("Input News")
    user_input = st.text_area("Paste here")

with col2:
    st.header("Prediction Result")
    if user_input:
        result = predict_fake_news(user_input)
        st.write(result)
