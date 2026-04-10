import type { SearchFilters } from "../types";

interface Props {
  filters: SearchFilters;
  onChange: (f: SearchFilters) => void;
}

const PLATFORMS = ["", "local", "claude", "chatgpt", "gemini"];
const SOURCE_TYPES = ["", "document", "conversation"];

export function FilterPanel({ filters, onChange }: Props) {
  const set = (k: keyof SearchFilters, v: string) =>
    onChange({ ...filters, [k]: v });

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12, padding: 16,
                  background: "#f8f9fa", borderRadius: 8, border: "1px solid #e0e0e0" }}>
      <h3 style={{ margin: 0, fontSize: 14, color: "#555" }}>Filters</h3>

      <label style={{ fontSize: 13 }}>
        Source type
        <select value={filters.source_type} onChange={e => set("source_type", e.target.value)}
          style={{ display: "block", marginTop: 4, width: "100%", padding: "6px 8px" }}>
          {SOURCE_TYPES.map(t => <option key={t} value={t}>{t || "All"}</option>)}
        </select>
      </label>

      <label style={{ fontSize: 13 }}>
        Platform
        <select value={filters.platform} onChange={e => set("platform", e.target.value)}
          style={{ display: "block", marginTop: 4, width: "100%", padding: "6px 8px" }}>
          {PLATFORMS.map(p => <option key={p} value={p}>{p || "All"}</option>)}
        </select>
      </label>

      <label style={{ fontSize: 13 }}>
        Category
        <input value={filters.category} onChange={e => set("category", e.target.value)}
          placeholder="e.g. AI-Research"
          style={{ display: "block", marginTop: 4, width: "100%", padding: "6px 8px" }} />
      </label>

      <label style={{ fontSize: 13 }}>
        From date
        <input type="date" value={filters.date_from}
          onChange={e => set("date_from", e.target.value ? e.target.value + "T00:00:00Z" : "")}
          style={{ display: "block", marginTop: 4, width: "100%", padding: "6px 8px" }} />
      </label>

      <label style={{ fontSize: 13 }}>
        To date
        <input type="date" value={filters.date_to}
          onChange={e => set("date_to", e.target.value ? e.target.value + "T23:59:59Z" : "")}
          style={{ display: "block", marginTop: 4, width: "100%", padding: "6px 8px" }} />
      </label>
    </div>
  );
}
