import json
from pathlib import Path
from typing import List
from .base import Parser, Conversation, Message

class ClaudeParser(Parser):
    def parse(self, file_path: Path) -> List[Conversation]:
        data = json.loads(Path(file_path).read_text(encoding="utf-8"))
        raw = data.get("conversations", [])
        results = []
        for item in raw:
            conv_id = item.get("uuid", "")
            title   = item.get("name", "Untitled")
            msgs_raw = item.get("chat_messages", [])
            messages = []
            date = None
            for m in msgs_raw:
                role = "user" if m.get("sender") == "human" else "assistant"
                content = m.get("text", "") or ""
                ts = m.get("created_at")
                if date is None and ts:
                    date = ts[:10]
                messages.append(Message(role=role, content=content, timestamp=ts))
            results.append(Conversation(
                platform="claude",
                conversation_id=conv_id,
                title=title,
                date=date or "1970-01-01",
                messages=messages,
            ))
        return results
