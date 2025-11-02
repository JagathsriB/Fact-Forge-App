import React, { useState } from "react";
import "./App.css";

import PlagiarismCard from "./components/PlagiarismCard";
import NLICard from "./components/NLICard";
import BiasCard from "./components/BiasCard";
import AIDetectionCard from "./components/AIDetectionCard"; // <-- NEW IMPORT
import Conclusion from "./components/Conclusion";
import SidePanel from "./components/SidePanel";


import { analyzeTextAPI } from "./services/api";

function App() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [expanded, setExpanded] = useState(false);

  const analyzeText = async () => {
    if (!text.trim()) return;
    setLoading(true);
    setResult(null);
    try {
      const data = await analyzeTextAPI(text);
      setResult(data);
    } catch (err) {
      console.error("Error:", err);
      setResult({ error: "Failed to analyze text" });
    }
    setLoading(false);
  };

  return (
    <div className={`app-container ${expanded ? "app-expanded" : ""}`}>
      <SidePanel onExpandChange={setExpanded} />
      <div className="main-content">
        <div className="container">
          <h1 className="gradient-text">Fact Forge</h1>

          <textarea
            rows="6"
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Enter text to analyze..."
            className="input-textarea"
          />

          <button
            onClick={analyzeText}
            className="analyze-button"
            disabled={loading}
          >
            {loading ? "Analyzing..." : "Analyze →"}
          </button>

          {result && !result.error && (
            <div className="results">
              <PlagiarismCard data={result?.Plagiarism_and_Fact_Checking} />
              <NLICard data={result?.Plagiarism_and_Fact_Checking?.NLI_Check} />
              <BiasCard data={result?.Bias_Detection} />
              {/* --- NEW CARD ADDED HERE --- */}
              <AIDetectionCard data={result?.AI_Generated_Content_Detection} />
            </div>
          )}

          {result?.error && <p className="error-text">{result.error}</p>}

          {result && (
            <div style={{ marginTop: "30px", textAlign: "left" }}>
              {/* Existing module cards here... */}

              {/* ✅ Add Conclusion */}
              <Conclusion result={result} />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
