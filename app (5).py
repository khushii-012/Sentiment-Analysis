
from flask import Flask, request, jsonify
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

app = Flask(__name__)

# --- NLTK Downloads and Initialization ---
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# --- Preprocessing Functions (same as in the notebook) ---
def clean_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_stopwords(text):
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered_tokens)

def apply_lemmatization(text):
    tokens = word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized_tokens)

def apply_tokenization(text):
    return word_tokenize(text)

# --- Load Model and Vectorizer ---
try:
    best_model_filename = 'trained_models/tuned_logistic_regression_model.pkl'
    best_log_reg_model = joblib.load(best_model_filename)

    vectorizer_filename = 'trained_models/count_vectorizer.pkl'
    count_vectorizer_loaded = joblib.load(vectorizer_filename)
    print("Model and Vectorizer loaded successfully.")
except Exception as e:
    print(f"Error loading model or vectorizer: {e}")
    best_log_reg_model = None
    count_vectorizer_loaded = None

# --- Prediction Function ---
def predict_sentiment(new_text, vectorizer, model):
    if not model or not vectorizer:
        return "Error: Model or vectorizer not loaded/fitted properly."

    # 1. Preprocess the text
    cleaned = clean_text(new_text)
    stopwords_removed = remove_stopwords(cleaned)
    lemmatized = apply_lemmatization(stopwords_removed)
    tokenized_list = apply_tokenization(lemmatized)
    processed_text = ' '.join(tokenized_list)

    # 2. Vectorize the text using the *loaded* vectorizer
    text_vectorized = vectorizer.transform([processed_text])

    # 3. Make prediction
    prediction = model.predict(text_vectorized)
    
    return prediction[0]

# --- API Endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    text = data.get('text')

    if not text:
        return jsonify({"error": "'text' field is required"}), 400

    sentiment = predict_sentiment(text, count_vectorizer_loaded, best_log_reg_model)
    return jsonify({"sentiment": sentiment})

if __name__ == '__main__':
    print("Flask app 'app.py' created. To run locally, save this file and execute `python app.py`.")
    print("To expose it publicly, consider using tools like ngrok or deploying to a cloud platform.")
