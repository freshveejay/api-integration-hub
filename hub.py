"""
API Integration Hub
Unified interface for AI and SaaS APIs
"""

import os
from typing import Optional, Dict, Any, List
import httpx


class AIClient:
    """Unified AI API client supporting OpenAI and Anthropic"""

    def __init__(
        self,
        openai_key: str = None,
        anthropic_key: str = None,
        timeout: float = 30.0
    ):
        self.openai_key = openai_key or os.getenv("OPENAI_API_KEY")
        self.anthropic_key = anthropic_key or os.getenv("ANTHROPIC_API_KEY")
        self.timeout = timeout

    def complete(
        self,
        prompt: str,
        provider: str = "openai",
        model: str = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Get completion from AI provider.

        Args:
            prompt: The input prompt
            provider: "openai" or "anthropic"
            model: Model name (defaults to gpt-4-turbo or claude-3-sonnet)
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature

        Returns:
            Generated text response
        """
        if provider == "anthropic":
            return self._anthropic_complete(
                prompt, model or "claude-3-sonnet-20240229",
                max_tokens, temperature, **kwargs
            )
        return self._openai_complete(
            prompt, model or "gpt-4-turbo",
            max_tokens, temperature, **kwargs
        )

    def _openai_complete(
        self,
        prompt: str,
        model: str,
        max_tokens: int,
        temperature: float,
        **kwargs
    ) -> str:
        if not self.openai_key:
            raise ValueError("OpenAI API key not configured")

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.openai_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    **kwargs
                }
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]

    def _anthropic_complete(
        self,
        prompt: str,
        model: str,
        max_tokens: int,
        temperature: float,
        **kwargs
    ) -> str:
        if not self.anthropic_key:
            raise ValueError("Anthropic API key not configured")

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.anthropic_key,
                    "anthropic-version": "2024-01-01",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "max_tokens": max_tokens,
                    "messages": [{"role": "user", "content": prompt}],
                    **kwargs
                }
            )
            response.raise_for_status()
            return response.json()["content"][0]["text"]

    def embed(
        self,
        text: str,
        model: str = "text-embedding-3-small"
    ) -> List[float]:
        """Generate embeddings for text"""
        if not self.openai_key:
            raise ValueError("OpenAI API key not configured")

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                "https://api.openai.com/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {self.openai_key}",
                    "Content-Type": "application/json"
                },
                json={"model": model, "input": text}
            )
            response.raise_for_status()
            return response.json()["data"][0]["embedding"]

    def embed_batch(
        self,
        texts: List[str],
        model: str = "text-embedding-3-small"
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        if not self.openai_key:
            raise ValueError("OpenAI API key not configured")

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                "https://api.openai.com/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {self.openai_key}",
                    "Content-Type": "application/json"
                },
                json={"model": model, "input": texts}
            )
            response.raise_for_status()
            return [item["embedding"] for item in response.json()["data"]]


class SlackClient:
    """Slack API client"""

    def __init__(self, token: str = None, timeout: float = 10.0):
        self.token = token or os.getenv("SLACK_BOT_TOKEN")
        self.base_url = "https://slack.com/api"
        self.timeout = timeout

    def _request(self, method: str, endpoint: str, **kwargs) -> Dict:
        if not self.token:
            raise ValueError("Slack token not configured")

        with httpx.Client(timeout=self.timeout) as client:
            response = client.request(
                method,
                f"{self.base_url}/{endpoint}",
                headers={
                    "Authorization": f"Bearer {self.token}",
                    "Content-Type": "application/json"
                },
                **kwargs
            )
            return response.json()

    def send_message(
        self,
        channel: str,
        text: str,
        blocks: List[Dict] = None,
        **kwargs
    ) -> Dict:
        """Send a message to a channel"""
        payload = {"channel": channel, "text": text, **kwargs}
        if blocks:
            payload["blocks"] = blocks
        return self._request("POST", "chat.postMessage", json=payload)

    def get_channels(self, types: str = "public_channel") -> List[Dict]:
        """Get list of channels"""
        result = self._request("GET", "conversations.list", params={"types": types})
        return result.get("channels", [])

    def get_user(self, user_id: str) -> Dict:
        """Get user info"""
        result = self._request("GET", "users.info", params={"user": user_id})
        return result.get("user", {})


class NotionClient:
    """Notion API client"""

    def __init__(self, token: str = None, timeout: float = 10.0):
        self.token = token or os.getenv("NOTION_API_KEY")
        self.base_url = "https://api.notion.com/v1"
        self.timeout = timeout

    def _request(self, method: str, endpoint: str, **kwargs) -> Dict:
        if not self.token:
            raise ValueError("Notion token not configured")

        with httpx.Client(timeout=self.timeout) as client:
            response = client.request(
                method,
                f"{self.base_url}/{endpoint}",
                headers={
                    "Authorization": f"Bearer {self.token}",
                    "Notion-Version": "2022-06-28",
                    "Content-Type": "application/json"
                },
                **kwargs
            )
            return response.json()

    def create_page(
        self,
        database_id: str,
        properties: Dict,
        children: List[Dict] = None
    ) -> Dict:
        """Create a page in a database"""
        payload = {
            "parent": {"database_id": database_id},
            "properties": properties
        }
        if children:
            payload["children"] = children
        return self._request("POST", "pages", json=payload)

    def query_database(
        self,
        database_id: str,
        filter: Dict = None,
        sorts: List[Dict] = None,
        page_size: int = 100
    ) -> List[Dict]:
        """Query a database"""
        payload = {"page_size": page_size}
        if filter:
            payload["filter"] = filter
        if sorts:
            payload["sorts"] = sorts

        result = self._request("POST", f"databases/{database_id}/query", json=payload)
        return result.get("results", [])

    def get_page(self, page_id: str) -> Dict:
        """Get a page by ID"""
        return self._request("GET", f"pages/{page_id}")


class AirtableClient:
    """Airtable API client"""

    def __init__(self, token: str = None, timeout: float = 10.0):
        self.token = token or os.getenv("AIRTABLE_API_KEY")
        self.base_url = "https://api.airtable.com/v0"
        self.timeout = timeout

    def _request(self, method: str, path: str, **kwargs) -> Dict:
        if not self.token:
            raise ValueError("Airtable token not configured")

        with httpx.Client(timeout=self.timeout) as client:
            response = client.request(
                method,
                f"{self.base_url}/{path}",
                headers={
                    "Authorization": f"Bearer {self.token}",
                    "Content-Type": "application/json"
                },
                **kwargs
            )
            return response.json()

    def list_records(
        self,
        base_id: str,
        table_name: str,
        view: str = None,
        max_records: int = 100
    ) -> List[Dict]:
        """List records in a table"""
        params = {"maxRecords": max_records}
        if view:
            params["view"] = view

        result = self._request("GET", f"{base_id}/{table_name}", params=params)
        return result.get("records", [])

    def create_record(
        self,
        base_id: str,
        table_name: str,
        fields: Dict
    ) -> Dict:
        """Create a record"""
        return self._request(
            "POST",
            f"{base_id}/{table_name}",
            json={"fields": fields}
        )

    def update_record(
        self,
        base_id: str,
        table_name: str,
        record_id: str,
        fields: Dict
    ) -> Dict:
        """Update a record"""
        return self._request(
            "PATCH",
            f"{base_id}/{table_name}/{record_id}",
            json={"fields": fields}
        )

    def delete_record(
        self,
        base_id: str,
        table_name: str,
        record_id: str
    ) -> Dict:
        """Delete a record"""
        return self._request("DELETE", f"{base_id}/{table_name}/{record_id}")


class APIHub:
    """Main hub providing access to all integrations"""

    def __init__(self):
        self.ai = AIClient()
        self.slack = SlackClient()
        self.notion = NotionClient()
        self.airtable = AirtableClient()

    def check_connections(self) -> Dict[str, bool]:
        """Check which services have credentials configured"""
        return {
            "openai": bool(os.getenv("OPENAI_API_KEY")),
            "anthropic": bool(os.getenv("ANTHROPIC_API_KEY")),
            "slack": bool(os.getenv("SLACK_BOT_TOKEN")),
            "notion": bool(os.getenv("NOTION_API_KEY")),
            "airtable": bool(os.getenv("AIRTABLE_API_KEY"))
        }


if __name__ == "__main__":
    hub = APIHub()

    print("API Integration Hub")
    print("=" * 40)
    print("\nConfigured services:")
    for service, configured in hub.check_connections().items():
        status = "✓" if configured else "✗"
        print(f"  {status} {service}")

    # Example: AI completion (requires OPENAI_API_KEY)
    if os.getenv("OPENAI_API_KEY"):
        print("\n" + "=" * 40)
        print("Testing AI completion...")
        try:
            response = hub.ai.complete("Say hello in 3 languages", max_tokens=100)
            print(f"Response: {response}")
        except Exception as e:
            print(f"Error: {e}")
