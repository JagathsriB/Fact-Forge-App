from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
from language_analyzer import LanguageAnalyzer
from checker.checker import (
    load_or_create_models,
    extract_keywords,
    search_duckduckgo,
    compare_similarity,
    calculate_trust_score,
    fact_check_text,
    run_nli_check,
)
import requests


# --- NEW: Import from your new detector module ---
from ai_vs_human.detector import load_ai_detection_models, predict_ai_generated_text

# Load environment variables
load_dotenv()

# --- API KEYS ---
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
GOOGLE_SAFE_KEY = os.getenv("GOOGLE_SAFE_BROWSING_KEY")
GOOGLE_FACT_KEY = os.getenv("GOOGLE_FACT_CHECK_KEY")

# --- MODEL PATH ---
LOCAL_MODEL_PATH = "Vivy-0507/deberta-v3-large-bias-detection"

# Initialize Flask and enable CORS
app = Flask(__name__)
CORS(app, origins=["https://fact-forge-app.onrender.com"])

# Initialize bias analyzer
analyzer = LanguageAnalyzer(local_model_path=LOCAL_MODEL_PATH, gemini_api_key=GEMINI_KEY)

# Load AI detection models on startup
ai_detection_models = load_ai_detection_models()


def unified_plagiarism_check(input_text):
    """Run plagiarism, similarity, trust score, NLI, and fact-check pipeline."""
    models = load_or_create_models()
    keywords = extract_keywords(input_text, models["kw_model"])
    query = " ".join(keywords)

    urls = search_duckduckgo(query, num_results=5)
    results = []

    for u in urls:
        url = u["url"]
        try:
            res = requests.get(url, timeout=10)
            if res.status_code != 200:
                continue
            text_content = res.text
            sim = compare_similarity(input_text, text_content, models["model"])
            trust = calculate_trust_score(url, GOOGLE_SAFE_KEY, models)
            weighted = sim * trust["trust_score"]
            results.append({
                "url": url,
                "similarity": sim,
                "trust_score": trust["trust_score"],
                "weighted_score": weighted,
                "content": text_content
            })
        except Exception:
            continue

    if not results:
        return {
            "Plagiarism": {
                "Top_Source": None, "Similarity": 0, "Trust_Score": 0,
                "Weighted_Score": 0, "Verdict": "No valid sources found"
            },
            "NLI_Check": {"Label": "Unknown", "Confidence": 0},
            "Fact_Check": []
        }

    results.sort(key=lambda x: x["weighted_score"], reverse=True)
    top = results[0]

    try:
        label, score = run_nli_check(input_text, top["content"], models["nli_model"])
    except Exception:
        label, score = "Unknown", 0

    try:
        facts = fact_check_text(input_text, GOOGLE_FACT_KEY) if GOOGLE_FACT_KEY else []
    except Exception:
        facts = []

    if top["weighted_score"] > 0.6:
        verdict = "Likely Plagiarized"
    elif top["weighted_score"] > 0.4:
        verdict = "Possibly Paraphrased"
    else:
        verdict = "Likely Original"

    return {
        "Plagiarism": {
            "Top_Source": top["url"],
            "Similarity": round(top["similarity"], 3),
            "Trust_Score": round(top["trust_score"], 3),
            "Weighted_Score": round(top["weighted_score"], 3),
            "Verdict": verdict
        },
        "NLI_Check": {"Label": label, "Confidence": round(score, 2)},
        "Fact_Check": facts
    }


@app.route("/analyze", methods=["POST"])
def analyze_text():
    data = request.json
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        plagiarism_res = unified_plagiarism_check(text)
        local_res = analyzer.analyze(text, backend="local")
        gemini_res = analyzer.analyze(text, backend="gemini")
        
        # --- MODIFIED SECTION ---
        # Check if the models were loaded successfully before trying to predict.
        if ai_detection_models:
            ai_detection_res = predict_ai_generated_text(text, ai_detection_models)
        else:
            # If models failed to load at startup, provide a clear error message.
            ai_detection_res = {
                "error": "AI Detection module is not available. Check server logs for model loading errors."
            }
        # --- END MODIFIED SECTION ---

        final_output = {
            "Plagiarism_and_Fact_Checking": plagiarism_res,
            "Bias_Detection": {
                "Local_Model": local_res,
                "Gemini_Model": gemini_res
            },
            "AI_Generated_Content_Detection": ai_detection_res
        }

        return jsonify(final_output)

    except Exception as e:
        # Log the full exception for debugging on the server
        print(f"An unexpected error occurred in /analyze: {e}")
        return jsonify({"error": "An unexpected server error occurred."}), 500


if __name__ == "__main__":
    app.run(debug=True)

