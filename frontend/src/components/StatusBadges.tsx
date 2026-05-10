interface StatusBadgesProps {
  provider: "gemini" | "groq" | null;
  fallback: boolean;
  streaming: boolean;
}

export function StatusBadges({ provider, fallback, streaming }: StatusBadgesProps) {
  if (!provider && !streaming) return null;

  return (
    <div className="status-badges">
      {streaming && (
        <span className="badge badge-streaming">
          <span className="pulse-dot" />
          streaming
        </span>
      )}
      {provider && (
        <span className={`badge badge-provider badge-${provider}`}>
          {provider === "gemini" ? "◆ gemini" : "⚡ groq"}
        </span>
      )}
      {fallback && (
        <span className="badge badge-fallback">↩ fallback</span>
      )}
    </div>
  );
}
