import type { PipelineResult, RetrievedDoc } from "../types/api";

const API_BASE = "http://127.0.0.1:8001/api";

export async function queryNonStream(
  query: string,
  topK = 5
): Promise<PipelineResult> {
  const res = await fetch(`${API_BASE}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, top_k: topK }),
  });
  if (!res.ok) throw new Error(`Server error: HTTP ${res.status}`);
  return res.json();
}

export async function queryRetrieve(
  query: string,
  topK = 5
): Promise<RetrievedDoc[]> {
  const res = await fetch(`${API_BASE}/retrieve`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, top_k: topK }),
  });
  if (!res.ok) throw new Error(`Server error: HTTP ${res.status}`);
  const data = await res.json();
  return data.results;
}

export async function checkHealth(): Promise<{ status: string }> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error("Health check failed");
  return res.json();
}
