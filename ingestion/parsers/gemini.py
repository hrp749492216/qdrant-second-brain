import json
from pathlib import Path
from typing import List
from .base import Parser, Conversation, Message

class GeminiParser(Parser):
    def parse(self, file_path: Path) -> List[Conversation]:
        data = json.loads(Path(file_path).read_text(encoding="utf-8"))
        raw = data.get("conversations", [])
        results = []
        for item in raw:
            conv_id = item.get("id", "")
            title   = item.get("title", "Untitled")
            replies = item.get("reply", [])
            messages = []
            date = None
            for r in replies:
                author = r.get("author", "")
                role = "user" if author == "user" else "assistant"
                content = r.get("text", "") or ""
                ts = r.get("timestamp")
                if date is None and ts:
                    date = ts[:10]
                messages.append(Message(role=role, content=content, timestamp=ts))
            results.append(Conversation(
                platform="gemini",
                conversation_id=conv_id,
                title=title,
                date=date or "1970-01-01",
                messages=messages,
            ))
        return results
