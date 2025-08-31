import streamlit as st
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# --------------------------
# Load Data
# --------------------------
st.set_page_config(page_title="Wordclouds", layout="wide")
st.title("☁️ Wordclouds")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Show unique sentiment values
    st.subheader("Unique Sentiment Values:")
    st.dataframe(pd.DataFrame(df["sentiment"].unique(), columns=["value"]))

    # Sentiment mapping (convert 1–5 to POSITIVE/NEGATIVE/NEUTRAL)
    sentiment_map = {
        1: "NEGATIVE",
        2: "NEGATIVE",
        3: "NEUTRAL",
        4: "POSITIVE",
        5: "POSITIVE"
    }
    df["sentiment_std"] = df["sentiment"].map(sentiment_map)

    # --------------------------
    # WordClouds
    # --------------------------
    col1, col2 = st.columns(2)

    # Positive Reviews
    with col1:
        st.subheader("Positive Reviews")
        positive_text = " ".join(df[df["sentiment_std"] == "POSITIVE"]["review"].astype(str))
        if positive_text.strip():
            wc_pos = WordCloud(width=800, height=400, background_color="black", colormap="Greens").generate(positive_text)
            st.image(wc_pos.to_array())
        else:
            st.info("No Positive Reviews found")

    # Negative Reviews
    with col2:
        st.subheader("Negative Reviews")
        negative_text = " ".join(df[df["sentiment_std"] == "NEGATIVE"]["review"].astype(str))
        if negative_text.strip():
            wc_neg = WordCloud(width=800, height=400, background_color="black", colormap="Reds").generate(negative_text)
            st.image(wc_neg.to_array())
        else:
            st.info("No Negative Reviews found")
