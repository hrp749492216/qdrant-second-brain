import { useState } from "react";
import { SearchBar } from "./components/SearchBar";
import { FilterPanel } from "./components/FilterPanel";
import { ResultsList } from "./components/ResultsList";
import type { SearchResult, SearchFilters } from "./types";

const PAGE_SIZE = 10;
const API_BASE  = "";

const EMPTY_FILTERS: SearchFilters = {
  source_type: "", platform: "", category: "", date_from: "", date_to: "",
};

export default function App() {
  const [query,    setQuery]    = useState("");
  const [filters,  setFilters]  = useState<SearchFilters>(EMPTY_FILTERS);
  const [results,  setResults]  = useState<SearchResult[]>([]);
  const [loading,  setLoading]  = useState(false);
  const [error,    setError]    = useState<string | null>(null);
  const [searched, setSearched] = useState(false);
  const [offset,   setOffset]   = useState(0);
  const [hasMore,  setHasMore]  = useState(false);

  const doSearch = async (q: string, off = 0, append = false) => {
    setLoading(true);
    setError(null);
    try {
      const params = new URLSearchParams({ query: q, limit: String(PAGE_SIZE), offset: String(off) });
      if (filters.source_type) params.set("source_type", filters.source_type);
      if (filters.platform)    params.set("platform",    filters.platform);
      if (filters.category)    params.set("category",    filters.category);
      if (filters.date_from)   params.set("date_from",   filters.date_from);
      if (filters.date_to)     params.set("date_to",     filters.date_to);

      const res  = await fetch(`${API_BASE}/search?${params}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();

      setResults(prev => append ? [...prev, ...data.results] : data.results);
      setHasMore(data.results.length === PAGE_SIZE);
      setOffset(off + data.results.length);
      setSearched(true);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  const handleSearch = (q: string) => {
    setQuery(q);
    setOffset(0);
    setResults([]);
    doSearch(q, 0, false);
  };

  const handleLoadMore = () => doSearch(query, offset, true);

  return (
    <div style={{ maxWidth: 1200, margin: "0 auto", padding: 24, fontFamily: "system-ui, sans-serif" }}>
      <h1 style={{ fontSize: 24, fontWeight: 700, marginBottom: 4 }}>Second Brain</h1>
      <p style={{ color: "#888", marginBottom: 24, fontSize: 14 }}>
        Semantic search over your documents and AI conversations.
      </p>

      <SearchBar onSearch={handleSearch} loading={loading} />

      <div style={{ display: "grid", gridTemplateColumns: "260px 1fr", gap: 24, marginTop: 24 }}>
        <FilterPanel filters={filters} onChange={f => { setFilters(f); setResults([]); setSearched(false); }} />
        <ResultsList results={results} loading={loading} error={error}
                     searched={searched} onLoadMore={handleLoadMore} hasMore={hasMore} />
      </div>
    </div>
  );
}
