import json, sys, os
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

FIXTURES = Path(__file__).parent / "fixtures"

def test_detect_platform_claude():
    from ingestion.parsers.base import detect_platform
    data = json.loads((FIXTURES / "claude_export.json").read_text())
    assert detect_platform(data) == "claude"

def test_detect_platform_chatgpt():
    from ingestion.parsers.base import detect_platform
    data = json.loads((FIXTURES / "chatgpt_export.json").read_text())
    assert detect_platform(data) == "chatgpt"

def test_detect_platform_gemini():
    from ingestion.parsers.base import detect_platform
    data = json.loads((FIXTURES / "gemini_export.json").read_text())
    assert detect_platform(data) == "gemini"

def test_claude_parser():
    from ingestion.parsers.claude import ClaudeParser
    convos = ClaudeParser().parse(FIXTURES / "claude_export.json")
    assert len(convos) == 1
    c = convos[0]
    assert c.platform == "claude"
    assert c.conversation_id == "aaaaaaaa-0001-0001-0001-000000000001"
    assert c.title == "Test Conversation"
    assert len(c.messages) == 2
    assert c.messages[0].role == "user"
    assert c.messages[1].role == "assistant"
    assert "transformer" in c.messages[0].content.lower()

def test_chatgpt_parser():
    from ingestion.parsers.chatgpt import ChatGPTParser
    convos = ChatGPTParser().parse(FIXTURES / "chatgpt_export.json")
    assert len(convos) == 1
    c = convos[0]
    assert c.platform == "chatgpt"
    assert c.conversation_id == "bbbbbbbb-0001-0001-0001-000000000001"
    assert len(c.messages) >= 2
    roles = {m.role for m in c.messages}
    assert "user" in roles and "assistant" in roles

def test_gemini_parser():
    from ingestion.parsers.gemini import GeminiParser
    convos = GeminiParser().parse(FIXTURES / "gemini_export.json")
    assert len(convos) == 1
    c = convos[0]
    assert c.platform == "gemini"
    assert c.conversation_id == "cccccccc-0001-0001-0001-000000000001"
    assert len(c.messages) == 2
