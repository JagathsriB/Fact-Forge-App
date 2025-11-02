# checker.py
import os
import pickle
import json
import requests
import re
import torch
import pandas as pd
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from keybert import KeyBERT
from nltk.tokenize import sent_tokenize
from serpapi import GoogleSearch  # <-- NEW SEARCH ENGINE

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

### ---------------- MODEL CACHE ---------------- ###
def load_or_create_models():
    """Loads heavy models from pickle or initializes and caches them."""
    cache_file = "model_cache.pkl"

    if os.path.exists(cache_file):
        print("🔁 Loading cached models from model_cache.pkl ...")
        with open(cache_file, "rb") as f:
            data = pickle.load(f)
        return data

    print("🚀 Initializing models for the first time...")
    kw_model = KeyBERT()
    model = SentenceTransformer("all-MiniLM-L12-v2")
    nli_model = pipeline("text-classification", model="roberta-large-mnli")

    url_df = pd.read_csv("url.csv")
    url_df.columns = url_df.columns.str.lower()
    if "url" not in url_df.columns or "type" not in url_df.columns:
        raise ValueError("CSV must contain 'url' and 'type' columns")

    X = url_df["url"]
    y = url_df["type"]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    vectorizer = CountVectorizer()
    X_vectorized = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_vectorized, y_encoded, test_size=0.3, random_state=42
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    ml_accuracy = accuracy_score(y_test, y_pred)
    print(f"✅ URL Classification Model Accuracy: {ml_accuracy:.4f}")

    data = {
        "kw_model": kw_model,
        "model": model,
        "nli_model": nli_model,
        "vectorizer": vectorizer,
        "label_encoder": label_encoder,
        "clf": clf,
    }

    '''with open(cache_file, "wb") as f:
        pickle.dump(data, f)
        print("💾 Models cached to model_cache.pkl")
    '''

    return data


### ---------------- SEARCH (via SerpAPI) ---------------- #### ---------------- SEARCH (DuckDuckGo) ---------------- #
from ddgs import DDGS

def search_duckduckgo(query, num_results=5):
    """Search DuckDuckGo and return a list of result dicts (url, title, snippet)."""
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=num_results):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", "")
                })
    except Exception as e:
        print(f"❌ DuckDuckGo search failed: {e}")
    return results

### ---------------- FACT CHECKING ---------------- ###
def extract_claims(text):
    """Extract simple factual sentences from text."""
    sents = sent_tokenize(text)
    claims = []
    patterns = [
        r"\b(is|are|was|were|has been|have been)\b",
        r"\b(according to|studies show|research indicates)\b",
        r"\b(percent|percentage|%)",
        r"\b(million|billion|trillion)\b",
        r"\d{4}",
    ]
    for s in sents:
        if len(s.split()) < 5 or s.strip().endswith("?"):
            continue
        if any(re.search(p, s, re.IGNORECASE) for p in patterns):
            claims.append(s)
    return claims[:3]


def check_google_fact_check(claim, api_key):
    """Query the Google Fact Check Tools API."""
    endpoint = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    params = {"key": api_key, "query": claim[:100], "languageCode": "en"}
    try:
        res = requests.get(endpoint, params=params)
        if res.status_code == 200:
            data = res.json()
            if "claims" in data and data["claims"]:
                return data["claims"]
    except Exception as e:
        print(f"Error in fact check API: {e}")
    return None


def fact_check_text(text, google_fact_api_key):
    """Perform fact-checking using Google API."""
    claims = extract_claims(text)
    results = []
    for c in claims:
        fc = check_google_fact_check(c, google_fact_api_key)
        if fc:
            for r in fc:
                if "claimReview" in r and r["claimReview"]:
                    rev = r["claimReview"][0]
                    results.append(
                        {
                            "claim": c,
                            "rating": rev.get("textualRating", "Unknown"),
                            "publisher": rev.get("publisher", {}).get("name", "Unknown"),
                            "url": rev.get("url", ""),
                        }
                    )
                    break
        else:
            results.append({"claim": c, "rating": "Not Found", "publisher": "N/A", "url": "N/A"})
    return results


def run_nli_check(input_text, reference_text, nli_model):
    """Natural Language Inference check using RoBERTa."""
    hypothesis = input_text.strip()
    premise = reference_text.strip()
    result = nli_model(f"{hypothesis} </s></s> {premise}", truncation=True)
    return result[0]["label"], result[0]["score"]


### ---------------- TRUST SCORE + SIMILARITY ---------------- ###
def check_google_safe_browsing(api_key, url):
    endpoint = f"https://safebrowsing.googleapis.com/v4/threatMatches:find?key={api_key}"
    payload = {
        "client": {"clientId": "plag-checker", "clientVersion": "1.0"},
        "threatInfo": {
            "threatTypes": ["MALWARE", "SOCIAL_ENGINEERING", "UNWANTED_SOFTWARE"],
            "platformTypes": ["ANY_PLATFORM"],
            "threatEntryTypes": ["URL"],
            "threatEntries": [{"url": url}],
        },
    }
    headers = {"Content-Type": "application/json"}
    try:
        res = requests.post(endpoint, headers=headers, data=json.dumps(payload))
        if res.status_code == 200 and "matches" in res.json():
            return "Blacklisted", 0.0
        return "Safe", 1.0
    except:
        return "Error", 0.5


def check_phishtank(url_to_check, phishtank_csv="verified_online.csv"):
    try:
        df = pd.read_csv(phishtank_csv)
        if "url" not in df.columns:
            return "CSV missing url", 0.5
        domain = urlparse(url_to_check).netloc.replace("www.", "")
        for u in df["url"].dropna():
            if domain in urlparse(u).netloc.replace("www.", ""):
                return "Phishing", 0.0
        return "Safe", 1.0
    except:
        return "Error", 0.5


def predict_url_type(url, vectorizer, clf, label_encoder):
    x_new = vectorizer.transform([url])
    pred = clf.predict(x_new)
    proba = clf.predict_proba(x_new)
    confidence = max(proba[0])
    return label_encoder.inverse_transform(pred)[0], confidence


def calculate_trust_score(url, google_api_key, models):
    ml_pred, ml_conf = predict_url_type(url, models["vectorizer"], models["clf"], models["label_encoder"])
    google_status, google_score = check_google_safe_browsing(google_api_key, url)
    phish_status, phish_score = check_phishtank(url)

    weights = {"ml": 0.6, "google": 0.3, "phish": 0.1}
    ml_score = 1.0 if ml_pred == "benign" else 0.0
    trust = (weights["ml"] * ml_score * ml_conf) + (weights["google"] * google_score) + (weights["phish"] * phish_score)
    return {"trust_score": trust, "ml_pred": ml_pred, "ml_conf": ml_conf,
            "google_status": google_status, "phish_status": phish_status}


def extract_keywords(text, kw_model, top_n=5):
    return [kw for kw, _ in kw_model.extract_keywords(text, top_n=top_n)]


def compare_similarity(text1, text2, model):
    sents1, sents2 = sent_tokenize(text1), sent_tokenize(text2)
    emb1, emb2 = model.encode(sents1, convert_to_tensor=True), model.encode(sents2, convert_to_tensor=True)
    sim_matrix = util.cos_sim(emb1, emb2)
    avg = torch.max(sim_matrix, dim=1).values.mean().item()
    return avg
