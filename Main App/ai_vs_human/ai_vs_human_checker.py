import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ============================
# 1. Load Model Assets
# ============================

# Load the TfidfVectorizer
try:
    with open('vectorizer.pkl', 'rb') as f:
        loaded_vectorizer = pickle.load(f)
    print("✅ TfidfVectorizer loaded successfully.")
except FileNotFoundError:
    print("❌ Error: 'vectorizer.pkl' not found. Ensure the training script was run.")
    exit()

# Load the Logistic Regression Model
try:
    with open('model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)
    print("✅ LogisticRegression model loaded successfully.")
except FileNotFoundError:
    print("❌ Error: 'model.pkl' not found. Ensure the training script was run.")
    exit()

# ============================
# 2. Sample Text and Prediction
# ============================

# Sample texts to test
new_texts = [
    "The quick brown fox jumps over the lazy dog, a short sentence demonstrating human typing.",
    "Artificial intelligence detection methods often rely on statistical analysis of linguistic features such as perplexity and burstiness to differentiate between machine-generated and naturally occurring human text."
]

print("\n--- Predictions ---")

for text in new_texts:
    # 1. Transform the new text using the loaded vectorizer
    # The text must be passed as a list, even if it's a single item.
    text_vec = loaded_vectorizer.transform([text])

    # 2. Make a prediction using the loaded model
    # Get the probability of the text being AI (class 1)
    probabilities = loaded_model.predict_proba(text_vec)
    ai_confidence = probabilities[0][1] * 100 # Probability of class 1 (AI)

    # 3. Determine the final label
    prediction = loaded_model.predict(text_vec)[0] # 1 for AI, 0 for Human
    label = "AI" if prediction == 1 else "Human"

    print(f"\nText: \"{text[:50]}...\"")
    print(f"-> Predicted Label: {label}")
    print(f"-> AI Confidence: {ai_confidence:.2f}%")