import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List
from .base import Parser, Conversation, Message

class ChatGPTParser(Parser):
    def parse(self, file_path: Path) -> List[Conversation]:
        data = json.loads(Path(file_path).read_text(encoding="utf-8"))
        results = []
        for item in data:
            conv_id = item.get("conversation_id", "")
            title   = item.get("title", "Untitled")
            mapping = item.get("mapping", {})
            messages = []
            date = None
            # Sort nodes by create_time
            nodes = sorted(
                [v for v in mapping.values() if v.get("message")],
                key=lambda n: n["message"].get("create_time") or 0,
            )
            for node in nodes:
                msg = node["message"]
                role_raw = msg.get("author", {}).get("role", "")
                if role_raw not in ("user", "assistant"):
                    continue
                parts   = msg.get("content", {}).get("parts", [])
                content = " ".join(str(p) for p in parts if isinstance(p, str))
                ts_raw  = msg.get("create_time")
                ts = None
                if ts_raw:
                    ts = datetime.fromtimestamp(ts_raw, tz=timezone.utc).isoformat()
                    if date is None:
                        date = ts[:10]
                messages.append(Message(role=role_raw, content=content, timestamp=ts))
            results.append(Conversation(
                platform="chatgpt",
                conversation_id=conv_id,
                title=title,
                date=date or "1970-01-01",
                messages=messages,
            ))
        return results
