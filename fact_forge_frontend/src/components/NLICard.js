import React from "react";

const NLICard = ({ data }) => (
  <div className="card">
    <h2 className="gradient-text">★ NLI Consistency Check</h2>
    <p><strong>Label →</strong> {data?.Label || "Unknown"}</p>
    <p><strong>Confidence →</strong> {data?.Confidence ?? 0}</p>
  </div>
);

export default NLICard;
