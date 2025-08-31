import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
import re
import string
import nltk
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# ========================
# DOWNLOAD STOPWORDS
# ========================
nltk.download('stopwords')
from nltk.corpus import stopwords

st.set_page_config(page_title="Sentiment Analysis App", layout="wide")

# ========================
# TEXT CLEANING FUNCTION
# ========================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www.\S+', '', text)  # remove links
    text = re.sub(r'[^a-z\s]', '', text)        # keep only alphabets
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = [w for w in text.split() if w not in stopwords.words('english')]
    return " ".join(words)

# ========================
# LOAD DATA
# ========================
st.title("🛍️ Customer Review Sentiment Analysis")

uploaded_file = st.file_uploader("📂 Upload CSV file with Reviews", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Check required columns
    if "Review" not in df.columns or "Sentiment" not in df.columns:
        st.error("CSV must contain 'Review' and 'Sentiment' columns!")
    else:
        df['cleaned'] = df['Review'].apply(clean_text)

        # ========================
        # TRAIN ML MODEL
        # ========================
        X = df['cleaned']
        y = df['Sentiment']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        vectorizer = TfidfVectorizer(max_features=5000)
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        model = LogisticRegression(max_iter=200)
        model.fit(X_train_vec, y_train)

        # Save (optional)
        joblib.dump(model, "sentiment_model.pkl")
        joblib.dump(vectorizer, "vectorizer.pkl")

        # Predict
        df['Predicted_Sentiment'] = model.predict(vectorizer.transform(df['cleaned']))

        # ========================
        # SHOW DATA
        # ========================
        st.subheader("📄 Sample Data")
        st.write(df.head(10))

        # ========================
        # WORDCLOUD
        # ========================
        st.subheader("☁ Word Clouds")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Positive Reviews Word Cloud")
            pos_text = " ".join(df[df['Predicted_Sentiment'] == 'positive']['cleaned'])
            if pos_text.strip():
                wc = WordCloud(width=500, height=400, background_color="white").generate(pos_text)
                fig, ax = plt.subplots()
                ax.imshow(wc, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig)

        with col2:
            st.markdown("### Negative Reviews Word Cloud")
            neg_text = " ".join(df[df['Predicted_Sentiment'] == 'negative']['cleaned'])
            if neg_text.strip():
                wc = WordCloud(width=500, height=400, background_color="black", colormap="Reds").generate(neg_text)
                fig, ax = plt.subplots()
                ax.imshow(wc, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig)

        # ========================
        # MODEL EVALUATION
        # ========================
        st.subheader("📊 Model Evaluation")

        y_pred = model.predict(X_test_vec)
        acc = accuracy_score(y_test, y_pred)
        st.write(f"✅ Accuracy: **{acc:.2f}**")

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=model.classes_, yticklabels=model.classes_)
        st.pyplot(fig)

        # ========================
        # EXTRA VISUALIZATIONS
        # ========================
        st.subheader("📈 Sentiment Distribution")
        fig, ax = plt.subplots()
        sns.countplot(x='Predicted_Sentiment', data=df, palette="Set2", ax=ax)
        st.pyplot(fig)

        st.subheader("📅 Monthly Trend (if Date column available)")
        if "Date" in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df['Month'] = df['Date'].dt.to_period('M')
            monthly_trend = df.groupby(['Month', 'Predicted_Sentiment']).size().unstack().fillna(0)
            fig, ax = plt.subplots(figsize=(10,5))
            monthly_trend.plot(ax=ax)
            plt.xticks(rotation=45)
            st.pyplot(fig)

        # ========================
        # USER INPUT PREDICTION
        # ========================
        st.subheader("✍ Predict Your Own Review")
        user_review = st.text_area("Type a review here...")
        if st.button("Predict Sentiment"):
            cleaned = clean_text(user_review)
            vec = vectorizer.transform([cleaned])
            pred = model.predict(vec)[0]
            st.success(f"Predicted Sentiment: **{pred.upper()}**")

else:
    st.info("👆 Please upload a CSV file to continue.")
