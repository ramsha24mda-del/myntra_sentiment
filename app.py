import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# -------------------------------
# Load Dataset
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("myntra_reviews.csv")   # apna dataset ka naam yahan rakho
    df.columns = [col.lower() for col in df.columns]  # ✅ make all columns lowercase
    return df

# -------------------------------
# Train Model
# -------------------------------
@st.cache_resource
def train_model(df):
    X = df['review']
    y = df['sentiment']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    vectorizer = CountVectorizer(stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)

    return model, vectorizer, acc

# -------------------------------
# Generate WordCloud
# -------------------------------
def plot_wordcloud(text, title):
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(text))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(title, fontsize=16)
    st.pyplot(plt)

# -------------------------------
# Streamlit App
# -------------------------------
st.title("🛍️ Myntra Customer Reviews - Sentiment Analysis")

# Load data
df = load_data()

# Sidebar
st.sidebar.header("Dataset Preview")
if st.sidebar.checkbox("Show Raw Data"):
    st.write(df.head())

# Train model
model, vectorizer, accuracy = train_model(df)
st.sidebar.success(f"Model trained with accuracy: {accuracy:.2f}")

# -------------------------------
# Word Clouds
# -------------------------------
st.subheader("Positive Reviews WordCloud")
positive_reviews = df[df['sentiment'] == "positive"]['review']
plot_wordcloud(positive_reviews, "Positive Reviews")

st.subheader("Negative Reviews WordCloud")
negative_reviews = df[df['sentiment'] == "negative"]['review']
plot_wordcloud(negative_reviews, "Negative Reviews")

# -------------------------------
# Prediction
# -------------------------------
st.subheader("🔮 Predict Sentiment of Your Review")
user_input = st.text_area("Enter your review here:")

if st.button("Predict Sentiment"):
    if user_input.strip():
        user_vec = vectorizer.transform([user_input])
        prediction = model.predict(user_vec)[0]
        st.success(f"Predicted sentiment: **{prediction.lower()}**")  # ✅ lowercase output
    else:
        st.warning("Please enter a review text.")
