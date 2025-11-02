import React from "react";

const AIDetectionCard = ({ data }) => {
  const detectionData = data || {};

  return (
    <div className="card">
      <h2 className="gradient-text">★ AI Content Detection</h2>
      <p>
        <strong>Verdict →</strong> {detectionData.Predicted_Label || "Unknown"}
      </p>
      <p>
        <strong>Confidence →</strong>{" "}
        {detectionData.Confidence !== undefined
          ? `${detectionData.Confidence}%`
          : "N/A"}
      </p>
    </div>
  );
};

export default AIDetectionCard;
