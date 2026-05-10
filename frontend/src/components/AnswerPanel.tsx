import { useEffect, useRef } from "react";
import type { RetrievalConfidence } from "../types/api";
import { StructuredAnswer } from "./StructuredAnswer";

interface AnswerPanelProps {
  answer: string;
  streaming: boolean;
  loading: boolean;
  error: string | null;
  query: string;
  retrievalConfidence: RetrievalConfidence | null;
}

export function AnswerPanel({
  answer,
  streaming,
  loading,
  error,
  query,
  retrievalConfidence,
}: AnswerPanelProps) {
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [answer]);

  if (!query && !loading) {
    return (
      <div className="answer-panel answer-empty">
        <div className="empty-icon">✚</div>
        <p className="empty-label">Enter a clinical query to begin</p>
        <p className="empty-sub">S-PubMedBERT-MS-MARCO embeddings · hybrid BM25 + FAISS · MS MARCO MiniLM reranker</p>
      </div>
    );
  }

  return (
    <div className="answer-panel">
      {query && (
        <div className="answer-query-label">
          <span className="label-tag">query</span>
          <span className="answer-query-text">{query}</span>
          {retrievalConfidence && (
            <span
              className={`confidence-chip confidence-${retrievalConfidence.label}`}
              title={retrievalConfidence.detail}
            >
              retrieval {retrievalConfidence.label}{" "}
              <span className="confidence-score">
                {Math.round(retrievalConfidence.score * 100)}%
              </span>
            </span>
          )}
        </div>
      )}

      {loading && !answer && (
        <div className="loading-dots">
          <span /><span /><span />
          <span className="loading-text">retrieving context…</span>
        </div>
      )}

      {error && (
        <div className="error-box">
          <span className="error-icon">⚠</span>
          {error}
        </div>
      )}

      {answer && (
        <div className="answer-content">
          <div className="answer-label-row">
            <span className="label-tag">analysis</span>
          </div>
          <div className="answer-text">
            {streaming ? (
              <div className="answer-stream-row">
                <div className="answer-streaming-wrap">
                  <StructuredAnswer text={answer} />
                </div>
                <span className="cursor-blink" aria-hidden="true" />
              </div>
            ) : (
              <StructuredAnswer text={answer} />
            )}
          </div>
        </div>
      )}

      <div ref={endRef} />
    </div>
  );
}
