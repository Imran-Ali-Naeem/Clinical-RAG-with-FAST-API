export interface RetrievedDoc {
  content: string;
  metadata: {
    source: string;
    disease_category: string;
    pdd_category: string;
    diagnosis: string;
  };
  score: number;
  method: string;
}

export interface RetrievalConfidence {
  score: number;
  label: "low" | "medium" | "high";
  basis: "rerank" | "rrf_fallback" | "none";
  top1_top2_gap: number | null;
  mean_normalized_strength: number;
  num_sources: number;
  detail: string;
}

export interface QueryRequest {
  query: string;
  top_k: number;
}

export interface PipelineResult {
  query: string;
  answer: string;
  sources: RetrievedDoc[];
  retrieval_confidence: RetrievalConfidence;
  provider_used: "gemini" | "groq";
  fallback_triggered: boolean;
  error_summary: string | null;
}

export interface StreamToken {
  type: "token";
  content: string;
}

export interface StreamMetadata {
  type: "metadata";
  provider_used: "gemini" | "groq";
  fallback_triggered: boolean;
  error_summary: string | null;
  sources: RetrievedDoc[];
  retrieval_confidence: RetrievalConfidence;
}

export interface StreamError {
  type: "error";
  content: string;
}

export type StreamEvent = StreamToken | StreamMetadata | StreamError;
