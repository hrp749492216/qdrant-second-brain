from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

@dataclass
class Message:
    role: str              # "user" | "assistant"
    content: str
    timestamp: Optional[str] = None   # ISO 8601 or None

@dataclass
class Conversation:
    platform: str          # "claude" | "chatgpt" | "gemini"
    conversation_id: str   # platform-native stable UUID/ID
    title: str
    date: str              # ISO 8601 date of first message
    messages: List[Message] = field(default_factory=list)

class Parser(ABC):
    @abstractmethod
    def parse(self, file_path: Path) -> List[Conversation]:
        ...

def detect_platform(data) -> Optional[str]:
    """Heuristic platform detection from parsed JSON."""
    if isinstance(data, list):
        # ChatGPT: top-level array with 'mapping' and 'conversation_id' per item
        if data and "mapping" in data[0] and "conversation_id" in data[0]:
            return "chatgpt"
    elif isinstance(data, dict):
        convos = data.get("conversations", [])
        if convos:
            first = convos[0] if isinstance(convos, list) else next(iter(convos.values()), {})
            if "uuid" in first and "chat_messages" in first:
                return "claude"
            if "id" in first and "reply" in first:
                return "gemini"
    return None
