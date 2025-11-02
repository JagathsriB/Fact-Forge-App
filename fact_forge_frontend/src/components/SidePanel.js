import React, { useState, useEffect } from "react";
import ReactMarkdown from 'react-markdown';
import {
  FiBookOpen,
  FiCheckCircle,
  FiAlertTriangle,
  FiShield,
  FiCpu,
  FiChevronLeft,
  FiChevronRight,
} from "react-icons/fi";
import "./SidePanel.css";

const explanations = {
  plagiarism: "/resources/plagiarism.txt",
  nli: "/resources/nli.txt",
  bias: "/resources/bias.txt",
  trust: "/resources/trust.txt",
  ai_vs_human: "/resources/ai_vs_human.txt",
};

// --- Create a data structure for the icons to ensure consistency ---
const icons = [
  { name: "plagiarism", Component: FiBookOpen },
  { name: "nli", Component: FiCheckCircle },
  { name: "bias", Component: FiAlertTriangle },
  { name: "trust", Component: FiShield },
  { name: "ai_vs_human", Component: FiCpu },
];

function SidePanel({ onExpandChange }) {
  const [expanded, setExpanded] = useState(false);
  const [active, setActive] = useState(null);
  const [content, setContent] = useState("");

  useEffect(() => {
    if (active) {
      fetch(explanations[active])
        .then((res) => res.text())
        .then((text) => setContent(text))
        .catch(() => setContent("⚠ Failed to load explanation."));
    }
  }, [active]);

  useEffect(() => {
    if (onExpandChange) onExpandChange(expanded);
  }, [expanded, onExpandChange]);

  const handleIconClick = (iconName) => {
    setExpanded(true);
    setActive(iconName);
  };

  const handleToggleSidebar = () => {
    setExpanded(!expanded);
    if (!expanded && !active) {
      setActive("plagiarism");
    }
  };

  return (
    <div className={`sidepanel ${expanded ? "expanded" : ""}`}>
      {/* Bar 1: Components text, expands to show icons */}
      <div className="components-bar">
        <span className="components-label title-gradient">Components</span>
        <div className="icon-list">
          {/* --- Map over the icons array to render each button --- */}
          {icons.map(({ name, Component }) => (
            // Wrap the icon in a div to create a consistent clickable area
            <div
              key={name}
              className={`icon ${active === name ? "active" : ""}`}
              onClick={() => handleIconClick(name)}
            >
              <Component />
            </div>
          ))}
        </div>
      </div>

      {/* Bar 2: The yellow toggle button, always visible */}
      <div className="toggle-bar" onClick={handleToggleSidebar}>
        {expanded ? <FiChevronLeft /> : <FiChevronRight />}
      </div>

      {/* The main content panel */}
      <div className={`panel ${expanded ? "open" : ""}`}>
        <div className="panel-header">
          {/* Replace underscore with space and capitalize for display */}
          <h2>{active?.replace(/_/g, " ").toUpperCase()}</h2>
        </div>
        <div className="panel-content">
          <ReactMarkdown>{content}</ReactMarkdown>
        </div>
      </div>
    </div>
  );
}

export default SidePanel;

