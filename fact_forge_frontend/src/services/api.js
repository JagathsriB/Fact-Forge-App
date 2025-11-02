// --- Configuration ---

// Use this for production (when your app is hosted)
const API_URL = "https://fact-forge-api.onrender.com/analyze";

// Use this for local testing
// const API_URL = "http://127.0.0.1:5000/analyze";

// ---------------------

export const analyzeTextAPI = async (text) => {
  const res = await fetch(API_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });

  if (!res.ok) {
    const errData = await res.json();
    throw new Error(errData.error || "API error");
  }

  return await res.json();
};
