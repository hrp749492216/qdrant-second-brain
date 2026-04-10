import { useState } from "react";

interface Props {
  onSearch: (query: string) => void;
  loading: boolean;
}

export function SearchBar({ onSearch, loading }: Props) {
  const [value, setValue] = useState("");

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    if (value.trim()) onSearch(value.trim());
  };

  return (
    <form onSubmit={submit} style={{ display: "flex", gap: 8 }}>
      <input
        value={value}
        onChange={e => setValue(e.target.value)}
        placeholder="Search your second brain..."
        style={{ flex: 1, padding: "10px 14px", fontSize: 16, borderRadius: 6,
                 border: "1px solid #ccc" }}
        disabled={loading}
      />
      <button type="submit" disabled={loading || !value.trim()}
        style={{ padding: "10px 20px", fontSize: 16, borderRadius: 6,
                 background: "#2563eb", color: "#fff", border: "none",
                 cursor: "pointer" }}>
        {loading ? "..." : "Search"}
      </button>
    </form>
  );
}
