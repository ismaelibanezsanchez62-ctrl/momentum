import json
import uuid

class UserMessage:
    def __init__(self, text: str):
        self.text = text

class LlmChat:
    def __init__(self, api_key: str, session_id: str, system_message: str):
        self.api_key = api_key
        self.session_id = session_id
        self.system_message = system_message
        self.model_provider = None
        self.model_name = None

    def with_model(self, provider: str, model: str):
        self.model_provider = provider
        self.model_name = model
        return self

    async def send_message(self, message: UserMessage) -> str:
        # Return a safe default JSON task list if the real integration isn't available.
        return json.dumps([
            {"title": "Take a small action", "duration_minutes": 10, "difficulty": 1, "day_offset": 0},
            {"title": "Review your goal and next step", "duration_minutes": 15, "difficulty": 1, "day_offset": 1},
        ])
