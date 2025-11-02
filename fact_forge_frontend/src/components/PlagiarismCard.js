import React from "react";

const PlagiarismCard = ({ data }) => {
  const plagiarism = data?.Plagiarism;
  const factCheck = data?.Fact_Check || [];

  return (
    <div className="card">
      <h2 className="gradient-text">★ Plagiarism Check</h2>
      {plagiarism?.Top_Source ? (
        <>
          <p>
            <strong>Top Source →</strong>{" "}
            <a
              href={plagiarism.Top_Source}
              target="_blank"
              rel="noopener noreferrer"
              style={{ color: "#ffcc00", textDecoration: "underline" }}
            >
              {plagiarism.Top_Source}
            </a>
          </p>
          <p><strong>Similarity →</strong> {plagiarism.Similarity}</p>
          <p><strong>Trust Score →</strong> {plagiarism.Trust_Score}</p>
          <p><strong>Weighted Score →</strong> {plagiarism.Weighted_Score}</p>
          <p><strong>Verdict →</strong> {plagiarism.Verdict}</p>
        </>
      ) : (
        <p>No sources found</p>
      )}

      <h2 className="gradient-text">★ Fact Check</h2>
      {factCheck.length > 0 ? (
        factCheck.map((f, idx) => (
          <div key={idx} style={{ marginBottom: "10px" }}>
            <p><strong>Claim →</strong> {f.claim}</p>
            <p><strong>Rating →</strong> {f.rating}</p>
            <p><strong>Publisher →</strong> {f.publisher}</p>
            <p>
              <strong>More Info →</strong>{" "}
              <a href={f.url} target="_blank" rel="noopener noreferrer">{f.url}</a>
            </p>
          </div>
        ))
      ) : (
        <p>No fact-check results found</p>
      )}
    </div>
  );
};

export default PlagiarismCard;
