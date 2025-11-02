import pickle
import os

# --- Build robust paths to the model files within this directory ---
script_dir = os.path.dirname(os.path.abspath(__file__))
VECTORIZER_PATH = os.path.join(script_dir, 'vectorizer.pkl')
MODEL_PATH = os.path.join(script_dir, 'model.pkl')

def load_ai_detection_models():
    """Loads the TF-IDF vectorizer and Logistic Regression model."""
    try:
        with open(VECTORIZER_PATH, 'rb') as f:
            vectorizer = pickle.load(f)
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        print("✅ AI Detection models (vectorizer, model) loaded successfully.")
        return {"vectorizer": vectorizer, "model": model}
    except FileNotFoundError as e:
        print(f"❌ Error loading AI Detection model: {e}. Ensure 'vectorizer.pkl' and 'model.pkl' are in the 'ai_vs_human' directory.")
        return None

def predict_ai_generated_text(text, models):
    """Predicts if text is AI-generated or human."""
    if not models:
        return {"error": "AI Detection models are not loaded."}
        
    try:
        # Transform the text using the loaded vectorizer
        text_vec = models["vectorizer"].transform([text])

        # Get prediction and probabilities
        prediction = models["model"].predict(text_vec)[0]
        probabilities = models["model"].predict_proba(text_vec)
        
        # --- FIX ---
        # Explicitly cast the prediction to a standard integer to ensure it's a valid index.
        prediction_index = int(prediction)
        
        # Confidence is the probability of the predicted class
        confidence = probabilities[0][prediction_index] * 100
        label = "AI-Generated" if prediction_index == 1 else "Human-Written"

        return {
            "Predicted_Label": label,
            "Confidence": round(confidence, 2)
        }
    except Exception as e:
        print(f"Error during AI detection prediction: {e}")
        return {"error": "Prediction failed."}

