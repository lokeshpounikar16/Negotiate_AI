# NegotiateAI — Agent-to-Agent Negotiation Protocol
## Nebius × NVIDIA Hackathon

A multi-agent system where specialized AI agents with conflicting objectives negotiate a shared plan through a typed protocol — bids, counter-proposals, and consensus — running on Nebius H100s.

---

## Week 1: Foundation

This week builds:
1. The `NegotiationMessage` protocol schema
2. Nebius inference client (OpenAI-compatible)
3. A single Cost Agent that queries Tavily + produces a valid PROPOSAL
4. FastAPI server with Redis message bus
5. Unit tests for schema validation

---

## Project Structure

```
negotiateai/
├── core/
│   ├── schema.py          # NegotiationMessage protocol schema (Pydantic)
│   ├── nebius_client.py   # Nebius LLM inference wrapper
│   ├── message_bus.py     # Redis pub/sub message bus
│   └── round_manager.py   # (Week 2) Negotiation round orchestrator
├── agents/
│   ├── base_agent.py      # Abstract base for all agents
│   └── cost_agent.py      # Cost Agent (Week 1 focus)
├── tests/
│   ├── test_schema.py     # Schema validation tests
│   └── test_cost_agent.py # Cost agent integration test
├── main.py                # FastAPI entrypoint
├── .env.example           # Environment variables template
└── requirements.txt
```

---

## Setup

```bash
# 1. Clone and install
pip install -r requirements.txt

# 2. Copy env file and fill in your keys
cp .env.example .env

# 3. Start Redis (Docker)
docker run -d -p 6379:6379 redis:alpine

# 4. Run tests
pytest tests/ -v

# 5. Start the API
uvicorn main:app --reload
```

---

## Environment Variables

| Variable | Description |
|---|---|
| `NEBIUS_API_KEY` | Your Nebius Token Factory API key |
| `TAVILY_API_KEY` | Tavily search API key |
| `REDIS_URL` | Redis connection string (default: redis://localhost:6379) |
| `NEBIUS_MODEL` | Nebius Token Factory model ID (default: NVIDIA Nemotron 3 Nano; verify/copy the exact ID from the model catalog) |
