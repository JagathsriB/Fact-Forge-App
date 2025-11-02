import React from "react";

const BiasCard = ({ data }) => {
  const localBias = data?.Local_Model || {};
  const geminiBias = data?.Gemini_Model || {};

  return (
    <div className="card">
      <h2 className="gradient-text">★ Bias Detection</h2>

      <h3>Local Model</h3>
      <p><strong>Bias →</strong> {localBias?.Bias || "Unknown"}</p>
      <p><strong>Loaded Language →</strong> {localBias?.["Loaded Language"] || "Unknown"}</p>
      <p><strong>Reason →</strong> {localBias?.Reason || "Unknown"}</p>

      <h3>Gemini Model</h3>
      <p><strong>Bias →</strong> {geminiBias?.Bias || "Unknown"}</p>
      <p><strong>Loaded Language →</strong> {geminiBias?.["Loaded Language"] || "Unknown"}</p>
      <p><strong>Reason →</strong> {geminiBias?.Reason || "Unknown"}</p>
    </div>
  );
};

export default BiasCard;
