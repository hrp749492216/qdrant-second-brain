export interface SearchResult {
  score: number;
  title: string;
  platform: string;
  category: string;
  date: string;
  source_path: string;
  chunk_text: string;
  parent_text: string;
  source_type: string;
}

export interface SearchFilters {
  source_type: string;
  platform: string;
  category: string;
  date_from: string;
  date_to: string;
}
