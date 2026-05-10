import { useState, useRef, useEffect } from "react";

const EXAMPLES = [
  "Symptoms of fever in a pneumonia patient",
  "Chest pain with elevated troponin",
  "Polyuria and polydipsia in type 2 diabetes",
  "Stroke symptoms and neurological findings",
  "COPD exacerbation presentation",
  "Heart failure with reduced ejection fraction",
];

interface QueryInputProps {
  onSubmit: (query: string) => void;
  loading: boolean;
  value: string;
  onChange: (v: string) => void;
}

export function QueryInput({ onSubmit, loading, value, onChange }: QueryInputProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
    }
  }, [value]);

  const handleKey = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (value.trim() && !loading) onSubmit(value.trim());
    }
  };

  const handleExample = (ex: string) => {
    onChange(ex);
    onSubmit(ex);
  };

  return (
    <div className="query-input-wrapper">
      <div className="example-chips">
        {EXAMPLES.map((ex) => (
          <button
            key={ex}
            className="chip"
            onClick={() => handleExample(ex)}
            disabled={loading}
          >
            {ex}
          </button>
        ))}
      </div>

      <div className="input-row">
        <textarea
          ref={textareaRef}
          className="query-textarea"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={handleKey}
          placeholder="Ask a clinical question… (Enter to send)"
          rows={1}
          disabled={loading}
        />
        <button
          className="send-btn"
          onClick={() => value.trim() && !loading && onSubmit(value.trim())}
          disabled={loading || !value.trim()}
          aria-label="Send query"
        >
          {loading ? (
            <span className="spinner" />
          ) : (
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="22" y1="2" x2="11" y2="13" />
              <polygon points="22 2 15 22 11 13 2 9 22 2" />
            </svg>
          )}
        </button>
      </div>
    </div>
  );
}
