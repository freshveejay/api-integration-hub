"""
API Integration Hub
Unified interface for AI and SaaS APIs
"""

import os
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import httpx


class AIClient:
    """Unified AI API client"""

    def __init__(self):
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY")

    def complete(
        self,
        prompt: str,
        provider: str = "openai",
        model: str = None,
        **kwargs
    ) -> str:
        """Get completion from AI provider"""

        if provider == "anthropic":
            return self._anthropic_complete(prompt, model or "claude-3-sonnet-20240229", **kwargs)
        return self._openai_complete(prompt, model or "gpt-4-turbo", **kwargs)

    def _openai_complete(self, prompt: str, model: str, **kwargs) -> str:
        with httpx.Client() as client:
            response = client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.openai_key}"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    **kwargs
                }
            )
            return response.json()["choices"][0]["message"]["content"]

    def _anthropic_complete(self, prompt: str, model: str, **kwargs) -> str:
        with httpx.Client() as client:
            response = client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.anthropic_key,
                    "anthropic-version": "2024-01-01"
                },
                json={
                    "model": model,
                    "max_tokens": 4096,
                    "messages": [{"role": "user", "content": prompt}],
                    **kwargs
                }
            )
            return response.json()["content"][0]["text"]

    def embed(self, text: str, model: str = "text-embedding-3-small") -> List[float]:
        """Generate embeddings"""
        with httpx.Client() as client:
            response = client.post(
                "https://api.openai.com/v1/embeddings",
                headers={"Authorization": f"Bearer {self.openai_key}"},
                json={"model": model, "input": text}
            )
            return response.json()["data"][0]["embedding"]


class SlackClient:
    """Slack API client"""

    def __init__(self):
        self.token = os.getenv("SLACK_BOT_TOKEN")
        self.base_url = "https://slack.com/api"

    def send_message(self, channel: str, text: str, **kwargs) -> Dict:
        with httpx.Client() as client:
            response = client.post(
                f"{self.base_url}/chat.postMessage",
                headers={"Authorization": f"Bearer {self.token}"},
                json={"channel": channel, "text": text, **kwargs}
            )
            return response.json()

    def get_channels(self) -> List[Dict]:
        with httpx.Client() as client:
            response = client.get(
                f"{self.base_url}/conversations.list",
                headers={"Authorization": f"Bearer {self.token}"}
            )
            return response.json().get("channels", [])


class NotionClient:
    """Notion API client"""

    def __init__(self):
        self.token = os.getenv("NOTION_API_KEY")
        self.base_url = "https://api.notion.com/v1"
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Notion-Version": "2022-06-28"
        }

    def create_page(self, database_id: str, properties: Dict) -> Dict:
        with httpx.Client() as client:
            response = client.post(
                f"{self.base_url}/pages",
                headers=self.headers,
                json={
                    "parent": {"database_id": database_id},
                    "properties": properties
                }
            )
            return response.json()

    def query_database(self, database_id: str, filter: Dict = None) -> List[Dict]:
        with httpx.Client() as client:
            response = client.post(
                f"{self.base_url}/databases/{database_id}/query",
                headers=self.headers,
                json={"filter": filter} if filter else {}
            )
            return response.json().get("results", [])


class AirtableClient:
    """Airtable API client"""

    def __init__(self):
        self.token = os.getenv("AIRTABLE_API_KEY")

    def list_records(self, base_id: str, table_name: str) -> List[Dict]:
        with httpx.Client() as client:
            response = client.get(
                f"https://api.airtable.com/v0/{base_id}/{table_name}",
                headers={"Authorization": f"Bearer {self.token}"}
            )
            return response.json().get("records", [])

    def create_record(self, base_id: str, table_name: str, fields: Dict) -> Dict:
        with httpx.Client() as client:
            response = client.post(
                f"https://api.airtable.com/v0/{base_id}/{table_name}",
                headers={"Authorization": f"Bearer {self.token}"},
                json={"fields": fields}
            )
            return response.json()


class APIHub:
    """Main hub providing access to all integrations"""

    def __init__(self):
        self.ai = AIClient()
        self.slack = SlackClient()
        self.notion = NotionClient()
        self.airtable = AirtableClient()


if __name__ == "__main__":
    hub = APIHub()

    # Example: AI completion
    response = hub.ai.complete("Say hello in 3 languages")
    print(f"AI Response: {response}")
