# API Integration Hub

Unified interface for connecting to multiple AI and SaaS APIs.

## Supported Integrations

### AI/ML
- OpenAI (GPT-4, Embeddings, DALL-E)
- Anthropic (Claude)
- Replicate (Open source models)

### Automation
- n8n
- Make.com (Integromat)
- Zapier

### Data
- Airtable
- Notion
- Google Sheets

### Communication
- Slack
- Discord
- Email (SendGrid, Resend)

## Quick Start

```bash
pip install -r requirements.txt
python main.py
```

## Usage

```python
from hub import APIHub

hub = APIHub()

# AI completions
response = hub.ai.complete(
    provider="openai",
    prompt="Hello!",
    model="gpt-4"
)

# Send Slack message
hub.slack.send_message(
    channel="#general",
    text="Hello from the hub!"
)

# Create Notion page
hub.notion.create_page(
    database_id="...",
    properties={"Name": "New Page"}
)
```

## Configuration

```env
# AI
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Communication
SLACK_BOT_TOKEN=xoxb-...
DISCORD_BOT_TOKEN=...

# Data
NOTION_API_KEY=secret_...
AIRTABLE_API_KEY=pat...
```

## Architecture

```
┌────────────────────────────────────┐
│            API Hub                 │
├──────────┬──────────┬──────────────┤
│    AI    │   Data   │    Comms     │
├──────────┼──────────┼──────────────┤
│ OpenAI   │ Notion   │ Slack        │
│ Claude   │ Airtable │ Discord      │
│ Replicate│ Sheets   │ Email        │
└──────────┴──────────┴──────────────┘
```

## License

MIT
