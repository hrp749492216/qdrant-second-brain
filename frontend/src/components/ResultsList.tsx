import { useState } from "react";
import type { SearchResult } from "../types";

const PLATFORM_COLORS: Record<string, string> = {
  local: "#16a34a", claude: "#d97706", chatgpt: "#2563eb", gemini: "#7c3aed",
};

function ResultCard({ result }: { result: SearchResult }) {
  const [expanded, setExpanded] = useState(false);
  const color = PLATFORM_COLORS[result.platform] || "#555";
  const date = result.date ? result.date.slice(0, 10) : "";

  return (
    <div style={{ border: "1px solid #e0e0e0", borderRadius: 8, padding: 16, marginBottom: 12 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start",
                    marginBottom: 8 }}>
        <div>
          <span style={{ fontWeight: 600, fontSize: 15 }}>{result.title}</span>
          <span style={{ marginLeft: 10, background: color, color: "#fff", borderRadius: 4,
                         padding: "2px 8px", fontSize: 12 }}>{result.platform}</span>
          {result.category && (
            <span style={{ marginLeft: 6, background: "#e5e7eb", borderRadius: 4,
                           padding: "2px 8px", fontSize: 12 }}>{result.category}</span>
          )}
        </div>
        <span style={{ fontSize: 12, color: "#888" }}>
          {date} · {(result.score * 100).toFixed(0)}%
        </span>
      </div>

      <p style={{ margin: "0 0 8px", fontSize: 14, color: "#333", lineHeight: 1.5 }}>
        {result.chunk_text}
      </p>

      {result.parent_text && result.parent_text !== result.chunk_text && (
        <button onClick={() => setExpanded(!expanded)}
          style={{ fontSize: 12, color: "#2563eb", background: "none", border: "none",
                   cursor: "pointer", padding: 0 }}>
          {expanded ? "Hide context ▲" : "Show context ▼"}
        </button>
      )}
      {expanded && (
        <p style={{ margin: "8px 0 0", fontSize: 13, color: "#555",
                    background: "#f8f9fa", padding: 10, borderRadius: 4, lineHeight: 1.5 }}>
          {result.parent_text}
        </p>
      )}
    </div>
  );
}

interface Props {
  results: SearchResult[];
  loading: boolean;
  error: string | null;
  searched: boolean;
  onLoadMore: () => void;
  hasMore: boolean;
}

export function ResultsList({ results, loading, error, searched, onLoadMore, hasMore }: Props) {
  if (loading) return <div style={{ textAlign: "center", padding: 40, color: "#888" }}>Searching...</div>;
  if (error)   return <div style={{ color: "#dc2626", padding: 16 }}>Error: {error}</div>;
  if (searched && results.length === 0)
    return <div style={{ color: "#888", padding: 40, textAlign: "center" }}>
      No results found. Try broader filters or a different query.
    </div>;

  return (
    <div>
      {results.map((r) => <ResultCard key={`${r.source_path}-${r.chunk_text.slice(0, 20)}`} result={r} />)}
      {hasMore && (
        <button onClick={onLoadMore}
          style={{ display: "block", margin: "0 auto", padding: "10px 24px",
                   background: "#f3f4f6", border: "1px solid #d1d5db", borderRadius: 6,
                   cursor: "pointer", fontSize: 14 }}>
          Load more
        </button>
      )}
    </div>
  );
}
