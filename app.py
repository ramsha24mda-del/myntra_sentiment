import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from transformers import pipeline

# ---------------------------
# Page Title
# ---------------------------
st.set_page_config(page_title="Myntra Sentiment Analysis", layout="wide")
st.title("🛍️ Myntra Customer Reviews - Sentiment Analysis")

# ---------------------------
# Upload Dataset
# ---------------------------
st.subheader("📂 Upload Dataset")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Dataset Overview
    st.subheader("📊 Dataset Overview")
    st.write(df.head())
    st.write("Dataset Shape:", df.shape)
    st.write("Sentiment Distribution:")
    st.bar_chart(df['sentiment'].value_counts())

    # ---------------------------
    # WordCloud - Positive & Negative
    # ---------------------------
    st.subheader("☁️ WordClouds")

    # Positive Reviews
    positive_reviews = df[df['sentiment'] == 5]['review']
    positive_text = " ".join(positive_reviews.astype(str))

    wc_pos = WordCloud(width=800, height=400, background_color='white',
                       colormap='Greens', stopwords='english').generate(positive_text)

    st.markdown("**Positive Reviews WordCloud**")
    fig1, ax1 = plt.subplots(figsize=(10,5))
    ax1.imshow(wc_pos, interpolation='bilinear')
    ax1.axis("off")
    st.pyplot(fig1)

    # Negative Reviews
    negative_reviews = df[df['sentiment'] == 1]['review']
    negative_text = " ".join(negative_reviews.astype(str))

    wc_neg = WordCloud(width=800, height=400, background_color='white',
                       colormap='Reds', stopwords='english').generate(negative_text)

    st.markdown("**Negative Reviews WordCloud**")
    fig2, ax2 = plt.subplots(figsize=(10,5))
    ax2.imshow(wc_neg, interpolation='bilinear')
    ax2.axis("off")
    st.pyplot(fig2)

    # ---------------------------
    # Sentiment Prediction
    # ---------------------------
    st.subheader("🤖 Try Your Own Review")

    @st.cache_resource
    def load_model():
        return pipeline("sentiment-analysis")

    sentiment_model = load_model()

    user_input = st.text_area("✍️ Enter a review to analyze:")

    if st.button("Predict"):
        if user_input.strip():
            prediction = sentiment_model(user_input)[0]
            label = prediction['label']
            score = prediction['score']
            st.success(f"**Prediction:** {label} (Confidence: {score:.2f})")
        else:
            st.warning("⚠️ Please enter a review text before predicting.")
else:
    st.info("Please upload a dataset to continue.")
