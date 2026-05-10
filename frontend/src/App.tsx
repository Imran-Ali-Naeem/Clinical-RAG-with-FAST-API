import { useState, useCallback, useRef } from "react";
import { QueryInput } from "./components/QueryInput";
import { AnswerPanel } from "./components/AnswerPanel";
import { SourcesPanel } from "./components/SourcesPanel";
import { StatusBadges } from "./components/StatusBadges";
import { queryNonStream } from "./api/client";
import { queryStream } from "./api/stream";
import type { RetrievedDoc, RetrievalConfidence } from "./types/api";
import "./App.css";

export default function App() {
  const [query, setQuery] = useState("");
  const [submittedQuery, setSubmittedQuery] = useState("");
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState<RetrievedDoc[]>([]);
  const [retrievalConfidence, setRetrievalConfidence] = useState<RetrievalConfidence | null>(null);
  const [provider, setProvider] = useState<"gemini" | "groq" | null>(null);
  const [fallback, setFallback] = useState(false);
  const [loading, setLoading] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [useStream, setUseStream] = useState(true);
  const abortRef = useRef(false);

  const reset = () => {
    setAnswer("");
    setSources([]);
    setRetrievalConfidence(null);
    setProvider(null);
    setFallback(false);
    setError(null);
    abortRef.current = false;
  };

  const handleSubmit = useCallback(async (q: string) => {
    reset();
    setSubmittedQuery(q);
    setLoading(true);

    try {
      if (useStream) {
        setLoading(false);
        setStreaming(true);
        for await (const event of queryStream(q, 5)) {
          if (abortRef.current) break;
          if (event.type === "token") {
            setAnswer((prev) => prev + event.content);
          } else if (event.type === "metadata") {
            setProvider(event.provider_used);
            setFallback(event.fallback_triggered);
            setSources(event.sources ?? []);
            setRetrievalConfidence(event.retrieval_confidence ?? null);
          } else if (event.type === "error") {
            setError(event.content);
          }
        }
        setStreaming(false);
      } else {
        const result = await queryNonStream(q, 5);
        setAnswer(result.answer);
        setSources(result.sources ?? []);
        setRetrievalConfidence(result.retrieval_confidence ?? null);
        setProvider(result.provider_used);
        setFallback(result.fallback_triggered);
        setLoading(false);
      }
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Unknown error");
      setLoading(false);
      setStreaming(false);
    }
  }, [useStream]);

  const isActive = loading || streaming;

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header-left">
          <div className="logo-mark">✚</div>
          <span className="logo-text">Clinical RAG</span>
          <span className="logo-sub">MIMIC-IV · S-PubMedBERT · BM25 + FAISS · cross-encoder rerank</span>
        </div>
        <div className="header-right">
          <StatusBadges provider={provider} fallback={fallback} streaming={streaming} />
          <label className="stream-toggle">
            <span className="toggle-label">stream</span>
            <button
              role="switch"
              aria-checked={useStream}
              className={`toggle-btn ${useStream ? "toggle-on" : ""}`}
              onClick={() => setUseStream((v) => !v)}
            >
              <span className="toggle-thumb" />
            </button>
          </label>
        </div>
      </header>

      {/* Main content */}
      <main className="main">
        {/* Answer — center, takes most space */}
        <section className="center-col">
          <AnswerPanel
            answer={answer}
            streaming={streaming}
            loading={loading}
            error={error}
            query={submittedQuery}
            retrievalConfidence={retrievalConfidence}
          />
        </section>

        {/* Sources — right panel */}
        <aside className="right-col">
          <SourcesPanel sources={sources} />
        </aside>
      </main>

      {/* Input — bottom */}
      <footer className="footer">
        <QueryInput
          onSubmit={handleSubmit}
          loading={isActive}
          value={query}
          onChange={setQuery}
        />
      </footer>
    </div>
  );
}
