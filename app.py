import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# ===============================
#   Load BERT Sentiment Pipeline
# ===============================
@st.cache_resource(show_spinner=True)
def load_pipeline():
    from transformers import pipeline
    return pipeline(
        "sentiment-analysis", 
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

@st.cache_data(show_spinner=False)
def load_csv(path: str):
    return pd.read_csv(path)

# ===============================
#   Normalize Sentiment Labels
# ===============================
def normalize_sentiment_column(df: pd.DataFrame) -> pd.DataFrame:
    if "sentiment" not in df.columns:
        return df

    def _map(v):
        if pd.isna(v):
            return np.nan
        if isinstance(v, (int, float, np.integer, np.floating)):
            if v in [0, 1, 2]:
                return {0: "NEGATIVE", 1: "POSITIVE", 2: "NEUTRAL"}[int(v)]
            if v in [-1, 0, 1]:
                return {-1: "NEGATIVE", 0: "NEUTRAL", 1: "POSITIVE"}[int(v)]
        s = str(v).strip().lower()
        if s in ["neg", "negative", "bad"]:
            return "NEGATIVE"
        if s in ["pos", "positive", "good"]:
            return "POSITIVE"
        if s in ["neu", "neutral"]:
            return "NEUTRAL"
        return s.upper()

    df = df.copy()
    df["sentiment_std"] = df["sentiment"].map(_map)
    return df

# ===============================
#   Wordcloud Helper
# ===============================
def make_wordcloud(series, title, cmap="Blues"):
    if series.empty:
        st.info(f"No data for {title}")
        return
    text = " ".join(series.dropna().astype(str))
    wc = WordCloud(
        width=900, height=500, background_color="black",
        colormap=cmap, max_words=100
    ).generate(text)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.subheader(title)
    st.pyplot(fig)

# ===============================
#   Predict One Review
# ===============================
def predict_one(pipe, text: str, neutral_margin: float = 0.05):
    if not text or not text.strip():
        return {"label": None, "score": None}
    res = pipe(text)[0]
    label = res["label"].upper()
    score = float(res["score"])
    p_pos = score if label == "POSITIVE" else 1.0 - score
    if 0.5 - neutral_margin <= p_pos <= 0.5 + neutral_margin:
        final = "NEUTRAL"
    else:
        final = "POSITIVE" if p_pos > 0.5 else "NEGATIVE"
    return {"label": final, "score": round(p_pos, 4)}

# ===============================
#   Streamlit App
# ===============================
st.set_page_config(page_title="Myntra Reviews — BERT Sentiment", page_icon="👜", layout="wide")
st.title("👜 Myntra Reviews — BERT Sentiment Analysis")

st.markdown(
    """
    Upload or use the default **Myntra.csv** dataset to explore reviews, 
    generate **Positive & Negative WordClouds**, 
    and run **BERT-powered sentiment analysis**.
    """
)

# ---- Sidebar / Controls ----
left, right = st.columns([2, 1])
with right:
    st.subheader("Data source")
    uploaded = st.file_uploader("Upload a CSV (optional)", type=["csv"])
    default_path = "Myntra.csv"
    use_default = st.toggle("Use default `Myntra.csv`", value=uploaded is None)
    neutral_band = st.slider("Neutral band (±)", 0.0, 0.25, 0.05, 0.01)

with left:
    st.subheader("Model")
    with st.spinner("Loading BERT pipeline..."):
        pipe = load_pipeline()
    st.success("Model ready.")

# ---- Load Data ----
df = None
if uploaded is not None and not use_default:
    df = pd.read_csv(uploaded)
elif use_default:
    try:
        df = load_csv(default_path)
    except Exception:
        st.warning("Couldn't find `Myntra.csv` in repo. Please upload a CSV.")

# ---- Dataset Overview ----
if df is not None:
    st.header("📊 Dataset Overview")
    if "review" not in df.columns:
        st.error("CSV must contain a 'review' column.")
    else:
        df = normalize_sentiment_column(df)
        st.write(f"Total rows: **{len(df)}** | Missing reviews: **{df['review'].isna().sum()}**")
        st.dataframe(df.head(20), use_container_width=True)

        if "sentiment_std" in df.columns:
            st.subheader("Sentiment distribution (dataset)")
            counts = df["sentiment_std"].value_counts(dropna=False).sort_index()
            fig = plt.figure(figsize=(6, 4))
            counts.plot(kind="bar")
            plt.title("Sentiment Distribution (dataset)")
            plt.xlabel("Sentiment")
            plt.ylabel("Count")
            st.pyplot(fig)

        # ---- WordClouds (ONLY POSITIVE & NEGATIVE) ----
        st.header("☁️ Wordclouds")
        st.write("Unique Sentiment Values:", df["sentiment"].unique())

        col1, col2 = st.columns(2)
        with col1:
            make_wordcloud(df[df["sentiment_std"] == "POSITIVE"]["review"], "Positive Reviews", cmap="Greens")
        with col2:
            make_wordcloud(df[df["sentiment_std"] == "NEGATIVE"]["review"], "Negative Reviews", cmap="Reds")

# ---- Single Prediction ----
st.header("🧪 Try a Review")
example = st.text_area("Type/paste a review:", value="Loved the fabric and fit, delivery was quick!")
if st.button("Predict sentiment"):
    out = predict_one(pipe, example, neutral_margin=neutral_band)
    if out["label"]:
        st.success(f"Prediction: **{out['label']}** (P(Positive)={out['score']})")
    else:
        st.info("Please enter some text.")

# ---- Batch Prediction ----
if df is not None and "review" in df.columns:
    st.header("📦 Batch Prediction")
    n_sample = st.slider("Sample size", 20, min(200, len(df)), min(100, len(df)))
    sample = df.sample(n=n_sample, random_state=42)["review"].astype(str).tolist()

    if st.button("Run batch prediction"):
        with st.spinner("Running predictions..."):
            preds = [predict_one(pipe, t, neutral_margin=neutral_band) for t in sample]
        pred_labels = [p["label"] for p in preds]
        pred_series = pd.Series(pred_labels).value_counts().sort_index()
        fig2 = plt.figure(figsize=(6, 4))
        pred_series.plot(kind="bar")
        plt.title("Predicted Sentiment Distribution (sample)")
        plt.xlabel("Sentiment")
        plt.ylabel("Count")
        st.pyplot(fig2)
        st.dataframe(pd.DataFrame({"review": sample, "predicted": pred_labels}).head(30), use_container_width=True)

st.caption("Built with Streamlit + HuggingFace DistilBERT (SST-2).")
