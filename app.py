import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from transformers import pipeline

# -----------------------
# Sentiment Model
# -----------------------
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis")

sentiment_model = load_model()

# -----------------------
# WordCloud helper inline
# -----------------------
def generate_wordcloud(text, title, colormap):
    if not text.strip():
        st.info(f"No data for {title}")
        return
    wc = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap=colormap,
        stopwords=STOPWORDS
    ).generate(text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt)

def show_wordclouds(df):
    st.header("☁️ Wordclouds")

    # ✅ Positive reviews
    positive_reviews = df[df['sentiment'] == 'POSITIVE']['review'].astype(str)
    positive_text = " ".join(positive_reviews)
    st.subheader("Positive Reviews")
    generate_wordcloud(positive_text, "Positive Reviews", "Greens")

    # ✅ Negative reviews
    negative_reviews = df[df['sentiment'] == 'NEGATIVE']['review'].astype(str)
    negative_text = " ".join(negative_reviews)
    st.subheader("Negative Reviews")
    generate_wordcloud(negative_text, "Negative Reviews", "Reds")


# -----------------------
# Main Streamlit App
# -----------------------
st.title("📊 Sentiment Analysis with WordClouds & Prediction")

# File upload
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Show data preview
    st.write("### Data Preview", df.head())

    # Check required columns
    if 'sentiment' not in df.columns or 'review' not in df.columns:
        st.error("CSV must contain 'sentiment' and 'review' columns.")
    else:
        # Show sentiment distribution
        st.subheader("Sentiment Distribution")
        st.bar_chart(df['sentiment'].value_counts())

        # Show WordClouds (positive + negative only)
        show_wordclouds(df)

# -----------------------
# Prediction Section
# -----------------------
st.header("🔮 Predict Sentiment for a New Review")

user_input = st.text_area("Enter a review to analyze:")

if st.button("Predict"):
    if user_input.strip():
        prediction = sentiment_model(user_input)[0]
        label = prediction['label']
        score = prediction['score']
        st.success(f"**Prediction:** {label} (Confidence: {score:.2f})")
    else:
        st.warning("Please enter a review text before predicting.")
