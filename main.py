import streamlit as st
import joblib
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

# Download NLTK stopwords (if not already downloaded)
nltk.download('stopwords', quiet=True)

# Load the model and vectorizer
model = joblib.load('model.joblib')
vectorizer = joblib.load('vectorizer.joblib')

def predict_sentiment(text):
    """Predicts the sentiment of a given text."""
    stemmer = PorterStemmer()
    stopwords_set = set(stopwords.words("english"))
    text_vec = text.lower().translate(str.maketrans('', '', string.punctuation)).split()
    text_vec = [stemmer.stem(word) for word in text_vec if word not in stopwords_set]
    corpus = [" ".join(text_vec)]
    x_senti = vectorizer.transform(corpus)
    x_senti_dense = x_senti.toarray()
    y_predicted = model.predict(x_senti_dense)
    if y_predicted == 0:
        return "The user is happy"
    else:
        return "The user is not happy"

# Streamlit UI
st.title("Sentiment Analysis")
st.write("Enter text to analyze its sentiment.")

user_input = st.text_input("Enter text here:")

if user_input:
    prediction = predict_sentiment(user_input)
    st.write(f"**Prediction:** {prediction}")