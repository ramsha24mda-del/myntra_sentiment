import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob

# -----------------------
# Load Data Function
# -----------------------
@st.cache_data
def load_data():
    # File ko app.py ke sath rakho
    df = pd.read_csv("Copy of Myntra.csv")
    return df

# -----------------------
# Sentiment Function
# -----------------------
def get_sentiment(text):
    analysis = TextBlob(str(text))
    polarity = analysis.sentiment.polarity
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

# -----------------------
# WordCloud Function
# -----------------------
def generate_wordcloud(text, title):
    wc = WordCloud(
        width=800,
        height=400,
        background_color="white",
        colormap="viridis",
        collocations=False
    ).generate(" ".join(text))
    plt.figure(figsize=(8, 4))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title, fontsize=16)
    st.pyplot(plt)

# -----------------------
# Streamlit App
# -----------------------
st.set_page_config(page_title="Myntra Sentiment Analysis", layout="wide")
st.title("🛍️ Myntra Customer Reviews - Sentiment Analysis")

# Load Data
df = load_data()

# Column names ko lowercase me convert karte hain
df.columns = [col.lower() for col in df.columns]

# Reviews aur sentiments check
if "review" not in df.columns:
    st.error("⚠️ Column `review` dataset me nahi mila. Kripya check karo.")
else:
    # Sentiment Analysis
    df["sentiment"] = df["review"].apply(get_sentiment)

    # Show dataframe
    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head())

    # Sentiment Distribution
    st.subheader("📊 Sentiment Distribution")
    sentiment_counts = df["sentiment"].value_counts()

    fig, ax = plt.subplots()
    sentiment_counts.plot(kind="bar", ax=ax, color=["green", "red", "gray"])
    ax.set_title("Sentiment Counts")
    ax.set_ylabel("Number of Reviews")
    st.pyplot(fig)

    # Word Clouds
    st.subheader("☁️ Word Clouds")
    positive_reviews = df[df["sentiment"] == "Positive"]["review"]
    negative_reviews = df[df["sentiment"] == "Negative"]["review"]

    col1, col2 = st.columns(2)
    with col1:
        generate_wordcloud(positive_reviews, "Positive Reviews WordCloud")
    with col2:
        generate_wordcloud(negative_reviews, "Negative Reviews WordCloud")

    # Prediction Section
    st.subheader("🔮 Sentiment Prediction")
    user_input = st.text_area("Type a review to analyze its sentiment:")

    if st.button("Predict Sentiment"):
        if user_input.strip() == "":
            st.warning("⚠️ Please enter some text first!")
        else:
            prediction = get_sentiment(user_input)
            if prediction == "Positive":
                st.success(f"😊 Sentiment: **{prediction}**")
            elif prediction == "Negative":
                st.error(f"😡 Sentiment: **{prediction}**")
            else:
                st.info(f"😐 Sentiment: **{prediction}**")

