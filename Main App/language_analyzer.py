# File: language_analyzer.py

import os
import json
import spacy
import torch
import torch.nn.functional as F
import google.generativeai as genai
from nrclex import NRCLex
from pysentimiento import create_analyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class LanguageAnalyzer:
    """
    Analyzer for loaded language and implicit bias,
    with two backends: local fine-tuned model and Gemini API.
    """

    def __init__(self, local_model_path: str, gemini_api_key: str = None):
        print("Initializing LanguageAnalyzer...")
        self._load_local_models(local_model_path)
        self._load_gemini_model(gemini_api_key)
        self.euphemism_dict = {
            "involuntary staff reduction", "rightsizing", "synergy", "streamlining",
            "optimizing", "strategic realignment", "downsizing"
        }
        print("✅ Analyzer ready.")

    def _load_local_models(self, model_path):
        """Load local models."""
        print("Loading local models (Pysentimiento, spaCy, Custom Bias Model)...")
        self.pys_emotion = create_analyzer(task="emotion", lang="en")
        self.pys_sentiment = create_analyzer(task="sentiment", lang="en")
        self.pys_irony = create_analyzer(task="irony", lang="en")
        self.nlp = spacy.load("en_core_web_sm")

        try:
            self.local_tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.local_model = AutoModelForSequenceClassification.from_pretrained(model_path)
            if torch.cuda.is_available():
                self.local_model.to("cuda")
        except Exception as e:
            print(f"⚠️ Warning: Could not load local bias model. Error: {e}")
            self.local_model, self.local_tokenizer = None, None

    def _load_gemini_model(self, api_key):
        """Initialize Gemini API client."""
        self.gemini_model = None
        if api_key:
            try:
                genai.configure(api_key=api_key)
                self.gemini_model = genai.GenerativeModel("gemini-pro-latest") # or self.gemini_model = genai.GenerativeModel("gemini-2.5-pro")

                print("✅ Gemini API model loaded successfully.")
            except Exception as e:
                print(f"⚠️ Warning: Could not configure Gemini API. Error: {e}")
        else:
            print("INFO: No Gemini API key provided. Gemini backend unavailable.")

    def _get_heuristic_reasons(self, text: str):
        """Run heuristic checks for loaded language."""
        reasons = []

        # 1. NRCLex emotions
        emotion_object = NRCLex(text)
        top_emotions = emotion_object.top_emotions
        if top_emotions:
            top_emotion_name, top_emotion_score = top_emotions[0]
            if top_emotion_score > 0:
                reasons.append(f"Dominant emotion: {top_emotion_name.capitalize()}")

        # 2. Irony detection
        irony_result = self.pys_irony.predict(text)
        if irony_result.output == "irony" and irony_result.probas["irony"] > 0.6:
            reasons.append(f"High irony ({irony_result.probas['irony']:.2f})")

        # 3. High-arousal emotions
        emotion_result = self.pys_emotion.predict(text)
        for emotion, prob in emotion_result.probas.items():
            if emotion in {"anger", "fear", "disgust", "surprise"} and prob > 0.3:
                reasons.append(f"High {emotion} ({prob:.2f})")

        # 4. Leading question
        doc = self.nlp(text)
        if doc and doc[-1].text == "?" and any(
            phrase in text.lower() for phrase in ["is it any surprise", "shouldn't we"]
        ):
            reasons.append("Leading question")

        # 5. Euphemisms
        if any(euphemism in text.lower() for euphemism in self.euphemism_dict):
            reasons.append("Contains corporate jargon/euphemism")

        return sorted(list(set(reasons)))

    def _analyze_with_local_model(self, text: str, threshold=0.35):
        """Analyze with local fine-tuned model."""
        if not self.local_model:
            return {"error": "Local model not available."}

        reasons = self._get_heuristic_reasons(text)

        inputs = self.local_tokenizer(
            text, return_tensors="pt", truncation=True, padding=True
        ).to(self.local_model.device)

        with torch.no_grad():
            outputs = self.local_model(**inputs)
            probabilities = F.softmax(outputs.logits, dim=-1)
            bias_probability = probabilities[0][1].item()

        bias_verdict = "Biased" if bias_probability > threshold else "Not-detected"
        if bias_verdict == "Biased":
            reasons.append(f"Flagged by custom bias model (Confidence: {bias_probability:.2%})")

        loaded_language_verdict = "Yes" if reasons else "No"

        return {
            "Bias": bias_verdict,
            "Loaded Language": loaded_language_verdict,
            "Reason": ", ".join(reasons) if reasons else "Neutral language.",
        }

    def _analyze_with_gemini(self, text: str):
        """Analyze using Gemini API with robust parsing."""
        if not self.gemini_model:
            return {"error": "Gemini backend unavailable. Provide an API key."}

        prompt = f"""
        Analyze the following text for bias and loaded language.
        Return a JSON object with fields:
        - Bias (Biased/Not-detected)
        - Loaded Language (Yes/No)
        - Reason (short explanation)

        Text: "{text}"
        """

        try:
            response = self.gemini_model.generate_content(prompt)

            # Safely extract text
            if hasattr(response, "candidates") and response.candidates:
                raw_text = response.candidates[0].content.parts[0].text.strip()
            else:
                return {"error": "No valid response from Gemini."}

            # Clean potential markdown wrappers
            raw_text = raw_text.replace("```json", "").replace("```", "").strip()

            # Parse as JSON if possible
            try:
                return json.loads(raw_text)
            except json.JSONDecodeError:
                return {"error": "Gemini response was not valid JSON", "raw": raw_text}

        except Exception as e:
            return {"error": f"Gemini API call failed. Details: {e}"}

    def analyze(self, text: str, backend: str = "local"):
        """Main entry point."""
        if backend == "local":
            return self._analyze_with_local_model(text)
        elif backend == "gemini":
            return self._analyze_with_gemini(text)
        else:
            return {"error": "Invalid backend. Use 'local' or 'gemini'."}
