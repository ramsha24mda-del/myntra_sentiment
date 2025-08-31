
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from io import StringIO
from functools import lru_cache

# ---- Inference (BERT via HF Transformers) ----
@st.cache_resource(show_spinner=True)
def load_pipeline():
    from transformers import pipeline
    # DistilBERT fine-tuned on SST-2 (POSITIVE/NEGATIVE)
    clf = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    return clf

@st.cache_data(show_spinner=False)
def load_csv(path: str):
    return pd.read_csv(path)

def normalize_sentiment_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Try to standardize sentiment labels to {NEGATIVE, NEUTRAL, POSITIVE}
    Supported input formats:
      - 0/1 or 0/1/2 (assume 0=NEGATIVE, 1=POSITIVE, 2=NEUTRAL)
      - -1/0/1 (assume -1=NEGATIVE, 0=NEUTRAL, 1=POSITIVE)
      - strings like "neg", "negative", "pos", "positive", "neutral"
    """
    if "sentiment" not in df.columns:
        return df

    ser = df["sentiment"]

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
    df["sentiment_std"] = ser.map(_map)
    return df

def make_wordcloud(texts, title):
    text_joined = " ".join(texts.astype(str).tolist())
    if not text_joined.strip():
        st.info(f"No text available for {title}.")
        return
    wc = WordCloud(width=1000, height=500, stopwords=STOPWORDS).generate(text_joined)
    fig = plt.figure(figsize=(8, 4))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.title(title)
    st.pyplot(fig)

def predict_one(pipe, text: str, neutral_margin: float = 0.05):
    """
    Convert BERT POSITIVE/NEGATIVE into 3-class with a small neutrality band.
    If model confidence is near 0.5 (+/- margin), label as NEUTRAL.
    """
    if not text or not text.strip():
        return {"label": None, "score": None}
    res = pipe(text)[0]
    label = res["label"].upper()
    score = float(res["score"])
    # Convert to probability of POSITIVE
    p_pos = score if label == "POSITIVE" else 1.0 - score
    if 0.5 - neutral_margin <= p_pos <= 0.5 + neutral_margin:
        final = "NEUTRAL"
    else:
        final = "POSITIVE" if p_pos > 0.5 else "NEGATIVE"
    return {"label": final, "score": round(p_pos, 4)}

st.set_page_config(page_title="Myntra Reviews — BERT Sentiment", page_icon="👜", layout="wide")
st.title("👜 Myntra Reviews — BERT Sentiment Analysis (Deploy-ready)")

st.markdown(
    """
    This app loads reviews from **Myntra.csv** (or a file you upload), shows quick EDA, 
    generates wordclouds, and uses a pre-trained **DistilBERT** model to predict sentiment.  
    """
)

# ---- Data Input ----
left, right = st.columns([2, 1])
with right:
    st.subheader("Data source")
    uploaded = st.file_uploader("Upload a CSV (optional)", type=["csv"], accept_multiple_files=False)
    default_path = "Myntra.csv"
    use_default = st.toggle("Use default `Myntra.csv` from repo", value=True if uploaded is None else False)
    neutral_band = st.slider("Neutral band (± around 0.5)", 0.0, 0.25, 0.05, 0.01)

with left:
    st.subheader("Model")
    with st.spinner("Loading BERT sentiment pipeline..."):
        pipe = load_pipeline()
    st.success("Model ready.")

# Load data
df = None
if uploaded is not None and not use_default:
    df = pd.read_csv(uploaded)
elif use_default:
    try:
        df = load_csv(default_path)
    except Exception as e:
        st.warning("Couldn't find `Myntra.csv` in the repository. Upload a CSV to continue.")

# ---- EDA ----
if df is not None:
    st.header("📊 Dataset Overview")
    if "review" not in df.columns:
        st.error("CSV must contain a 'review' column.")
    else:
        df = normalize_sentiment_column(df)
        n_rows = len(df)
        n_missing = df["review"].isna().sum()
        st.write(f"Total rows: **{n_rows}** | Missing reviews: **{n_missing}**")

        # Show head
        st.dataframe(df.head(20), use_container_width=True)

        # Sentiment distribution if available
        if "sentiment_std" in df.columns:
            st.subheader("Sentiment distribution (from dataset)")
            counts = df["sentiment_std"].value_counts(dropna=False).sort_index()
            fig = plt.figure(figsize=(6,4))
            counts.plot(kind="bar")
            plt.title("Sentiment Distribution (dataset)")
            plt.xlabel("Sentiment")
            plt.ylabel("Count")
            st.pyplot(fig)

        # Wordclouds
        st.header("☁️ Wordclouds")
        col1, col2, col3 = st.columns(3)
        with col1:
            make_wordcloud(df["review"], "All Reviews")
        if "sentiment_std" in df.columns:
            with col2:
                make_wordcloud(df[df["sentiment_std"] == "POSITIVE"]["review"], "Positive Reviews")
            with col3:
                make_wordcloud(df[df["sentiment_std"] == "NEGATIVE"]["review"], "Negative Reviews")

# ---- Inference on single text ----
st.header("🧪 Try a Review")
example = st.text_area("Type/paste a review:", value="Loved the fabric and fit, delivery was quick!")
if st.button("Predict sentiment"):
    out = predict_one(pipe, example, neutral_margin=neutral_band)
    if out["label"]:
        st.success(f"Prediction: **{out['label']}** (P(Positive)={out['score']})")
    else:
        st.info("Please enter some text.")

# ---- Batch inference (optional) ----
if df is not None and "review" in df.columns:
    st.header("📦 Batch Prediction on Sample")
    n_sample = st.slider("Sample size", 20, min(200, len(df)), min(100, len(df)))
    sample = df.sample(n=n_sample, random_state=42)["review"].astype(str).tolist()

    if st.button("Run batch prediction"):
        with st.spinner("Running predictions..."):
            preds = [predict_one(pipe, t, neutral_margin=neutral_band) for t in sample]
        pred_labels = [p["label"] for p in preds]
        pred_series = pd.Series(pred_labels).value_counts().sort_index()
        fig2 = plt.figure(figsize=(6,4))
        pred_series.plot(kind="bar")
        plt.title("Predicted Sentiment Distribution (sample)")
        plt.xlabel("Sentiment")
        plt.ylabel("Count")
        st.pyplot(fig2)
        st.dataframe(pd.DataFrame({"review": sample, "predicted": pred_labels}).head(30), use_container_width=True)

st.caption("Built with Streamlit + HuggingFace DistilBERT (SST-2).")
