import { useState } from "react";
import type { RetrievedDoc } from "../types/api";

const METHOD_LABELS: Record<string, string> = {
  hybrid: "hybrid",
  bm25:   "bm25",
  dense:  "dense",
};

function methodStyleKey(method: string): string {
  if (method.includes("→rerank")) return "rerank";
  const base = method.split("→")[0].trim();
  return METHOD_LABELS[base] ? base : "dense";
}

function methodLabel(method: string): string {
  if (method.includes("→rerank")) return "reranked";
  return METHOD_LABELS[method] ?? method;
}

function SourceCard({ doc, index }: { doc: RetrievedDoc; index: number }) {
  const [open, setOpen] = useState(false);
  const mkey = methodStyleKey(doc.method);

  return (
    <div className={`source-card source-${mkey}`}>
      <button className="source-header" onClick={() => setOpen((o) => !o)}>
        <div className="source-header-top">
          <span className="source-index">#{index + 1}</span>
          <span className={`method-tag method-${mkey}`}>
            {methodLabel(doc.method)}
          </span>
          <span className="source-chevron">{open ? "▲" : "▼"}</span>
        </div>
        <div className="source-diagnosis">{doc.metadata.diagnosis}</div>
        <div className="source-meta">
          {doc.metadata.disease_category}
          <span className="source-score">· {doc.score.toFixed(5)}</span>
        </div>
      </button>

      {open && (
        <div className="source-body">
          <p className="source-content">{doc.content}</p>
          <div className="source-file">{doc.metadata.source}</div>
        </div>
      )}
    </div>
  );
}

interface SourcesPanelProps {
  sources: RetrievedDoc[];
}

export function SourcesPanel({ sources }: SourcesPanelProps) {
  return (
    <div className="sources-panel">
      <div className="sources-header">
        <span className="label-tag">sources</span>
        {sources.length > 0 && (
          <span className="sources-count">{sources.length}</span>
        )}
      </div>

      {sources.length === 0 ? (
        <div className="sources-empty">Retrieved documents appear here</div>
      ) : (
        <div className="sources-list">
          {sources.map((doc, i) => (
            <SourceCard key={i} doc={doc} index={i} />
          ))}
        </div>
      )}
    </div>
  );
}
