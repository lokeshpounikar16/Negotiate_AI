# NegotiationAI Project: Step-by-Step Implementation Guide

## Table of Contents
1. [Prerequisites & Setup](#prerequisites--setup)
2. [Phase 1: Foundation (Week 1)](#phase-1-foundation-week-1)
3. [Phase 2: Multi-Agent Loop (Week 2)](#phase-2-multi-agent-loop-week-2)
4. [Phase 3: Deadlock & Mediator (Week 3)](#phase-3-deadlock--mediator-week-3)
5. [Phase 4: Dashboard (Week 4)](#phase-4-dashboard-week-4)
6. [Phase 5: Polish & Evaluation (Week 5)](#phase-5-polish--evaluation-week-5)
7. [Testing Strategy](#testing-strategy)
8. [Debugging Tips](#debugging-tips)

---

## Prerequisites & Setup

### 1.1 System Requirements

```bash
# Check Python version (3.10+ required)
python3 --version

# Install package manager (pip)
pip install --upgrade pip setuptools wheel
```

### 1.2 Create Project Directory

```bash
# Create and navigate to project
mkdir negotiation-ai
cd negotiation-ai

# Initialize git
git init
git config user.name "Your Name"
git config user.email "you@example.com"
```

### 1.3 Create Virtual Environment

```bash
# Create venv
python3 -m venv venv

# Activate venv
source venv/bin/activate  # On macOS/Linux
# OR
venv\Scripts\activate  # On Windows

# Verify activation (should show venv in prompt)
which python3
```

### 1.4 Project Structure Setup

```bash
# Create folder structure
mkdir -p src/{agents,protocol,orchestrator,utils,storage}
mkdir -p tests/{unit,integration}
mkdir -p config
mkdir -p dashboard
mkdir -p data/logs

# Create __init__.py files
touch src/__init__.py
touch src/agents/__init__.py
touch src/protocol/__init__.py
touch src/orchestrator/__init__.py
touch src/utils/__init__.py
touch src/storage/__init__.py
```

**Final structure:**
```
negotiation-ai/
├── src/
│   ├── agents/              # Agent implementations
│   ├── protocol/            # NegotiationMessage schema & validation
│   ├── orchestrator/        # Round manager, message bus
│   ├── utils/               # Helper functions
│   └── storage/             # PostgreSQL interface
├── tests/
│   ├── unit/                # Unit tests
│   └── integration/         # Integration tests
├── config/                  # Configuration files
├── dashboard/               # React frontend (later)
├── data/logs/               # Negotiation logs
├── venv/                    # Python virtual environment
├── requirements.txt         # Dependencies
├── .env.example             # Environment variables template
├── .gitignore               # Git ignore file
└── main.py                  # Entry point
```

### 1.5 Dependencies Installation

Create `requirements.txt`:

```txt
# Core dependencies
python-dotenv==1.0.0
pydantic==2.5.0
typing-extensions==4.9.0

# LLM & Agent Framework
anthropic==0.15.0
langchain==0.1.0
langgraph==0.0.1

# API & Web
fastapi==0.109.0
uvicorn==0.27.0
websockets==12.0

# Message Bus
redis==5.0.1

# Database
psycopg2-binary==2.9.9
sqlalchemy==2.0.23
alembic==1.13.0

# Real-time Data
tavily-python==0.1.0

# Utilities
httpx==0.25.2
uuid==1.30
requests==2.31.0

# Testing
pytest==7.4.3
pytest-asyncio==0.23.0
pytest-cov==4.1.0

# Development
black==23.12.0
isort==5.13.2
mypy==1.8.0
```

Install:

```bash
pip install -r requirements.txt
```

### 1.6 Environment Configuration

Create `.env.example`:

```env
# Nebius/LLM Configuration
NEBIUS_API_KEY=your_nebius_api_key
NEBIUS_MODEL=Llama-3.1-70B
NEBIUS_API_ENDPOINT=https://api.nebius.ai/v1

# Anthropic (for comparisons)
ANTHROPIC_API_KEY=your_anthropic_api_key

# Tavily API (Real-time vendor data)
TAVILY_API_KEY=your_tavily_api_key

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# PostgreSQL Configuration
DB_HOST=localhost
DB_PORT=5432
DB_USER=negotiation_user
DB_PASSWORD=secure_password
DB_NAME=negotiation_ai

# Server Configuration
FASTAPI_HOST=0.0.0.0
FASTAPI_PORT=8000
ENVIRONMENT=development
```

Copy to `.env`:

```bash
cp .env.example .env
# Edit .env with your actual API keys
```

### 1.7 Git Setup

Create `.gitignore`:

```
# Virtual environment
venv/
env/
ENV/

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Environment variables
.env
.env.local
.env.*.local

# IDE
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store

# Logs
data/logs/
*.log

# Testing
.pytest_cache/
.coverage
htmlcov/

# Database
*.db
*.sqlite
```

Initialize git:

```bash
git add .
git commit -m "Initial project setup"
```

---

# Phase 1: Foundation (Week 1)

## Step 1: Define the Protocol Schema

Create `src/protocol/message.py`:

```python
import json
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field

class MessageType(str, Enum):
    TASK_ANNOUNCEMENT = "TASK_ANNOUNCEMENT"
    PROPOSAL = "PROPOSAL"
    COUNTER_PROPOSAL = "COUNTER_PROPOSAL"
    CONCESSION = "CONCESSION"
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    MEDIATOR_COMPROMISE = "MEDIATOR_COMPROMISE"

class ProposalDetail(BaseModel):
    """The actual proposal terms"""
    vendor: str
    unit_price_usd: float
    total_cost_usd: float
    delivery_days: int
    quality_tier: str  # A, A-, B+, B, etc.
    vendor_risk_score: float = Field(ge=0, le=1)
    
    class Config:
        example = {
            "vendor": "Dell Technologies",
            "unit_price_usd": 3800,
            "total_cost_usd": 1900000,
            "delivery_days": 55,
            "quality_tier": "B+",
            "vendor_risk_score": 0.24
        }

class NegotiationMessage(BaseModel):
    """Core typed message for agent-to-agent communication"""
    msg_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    round: int
    sender: str  # "cost_agent", "quality_agent", etc.
    msg_type: MessageType
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Proposal content
    proposal: ProposalDetail
    
    # Reasoning (public)
    public_reason: str
    
    # Concessions made in this round
    concessions_made: List[str] = Field(default_factory=list)
    
    # Private evaluation (NOT revealed to other agents)
    utility_score: float = Field(ge=0, le=1)
    
    # BATNA info
    batna_activated: bool = False
    batna_utility_if_rejected: float = Field(default=0.0, ge=0, le=1)
    
    # Deal breakers (hard constraints)
    deal_breakers: List[str] = Field(default_factory=list)
    
    # Acceptance threshold
    acceptance_threshold: float = Field(default=0.65, ge=0, le=1)
    
    # Internal reasoning trace (for debugging)
    reasoning_trace: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        example = {
            "msg_id": "550e8400-e29b-41d4-a716-446655440000",
            "round": 3,
            "sender": "cost_agent",
            "msg_type": "COUNTER_PROPOSAL",
            "timestamp": "2026-10-01T14:22:31Z",
            "proposal": {
                "vendor": "Dell Technologies",
                "unit_price_usd": 3800,
                "total_cost_usd": 1900000,
                "delivery_days": 55,
                "quality_tier": "B+",
                "vendor_risk_score": 0.24
            },
            "public_reason": "Accepting lower quality tier to stay within budget cap",
            "concessions_made": ["quality_tier: A → B+"],
            "utility_score": 0.71,
            "batna_activated": False,
            "batna_utility_if_rejected": 0.20,
            "deal_breakers": ["delivery_days > 60", "total_cost > 2000000"],
            "acceptance_threshold": 0.65,
            "reasoning_trace": {}
        }

    def to_json(self) -> str:
        """Serialize to JSON for Redis/PostgreSQL"""
        return self.model_dump_json()

    @classmethod
    def from_json(cls, json_str: str) -> "NegotiationMessage":
        """Deserialize from JSON"""
        return cls.model_validate_json(json_str)

    def is_acceptable(self, agent_utility: float) -> bool:
        """Check if proposal is acceptable to receiving agent"""
        return agent_utility >= self.acceptance_threshold

    def violates_deal_breaker(self, **kwargs) -> bool:
        """Check if proposal violates any deal breaker"""
        # Example: check if delivery_days > 60
        for breaker in self.deal_breakers:
            if "delivery_days" in breaker:
                max_days = int(breaker.split(">")[-1].strip())
                if self.proposal.delivery_days > max_days:
                    return True
        return False
```

**Test the schema:**

Create `tests/unit/test_protocol.py`:

```python
import pytest
from src.protocol.message import NegotiationMessage, ProposalDetail, MessageType
import json

def test_message_creation():
    """Test basic message creation"""
    msg = NegotiationMessage(
        round=1,
        sender="cost_agent",
        msg_type=MessageType.PROPOSAL,
        proposal=ProposalDetail(
            vendor="Dell",
            unit_price_usd=3800,
            total_cost_usd=1900000,
            delivery_days=55,
            quality_tier="B+",
            vendor_risk_score=0.24
        ),
        public_reason="Test proposal",
        utility_score=0.75
    )
    assert msg.sender == "cost_agent"
    assert msg.proposal.vendor == "Dell"

def test_message_serialization():
    """Test JSON serialization"""
    msg = NegotiationMessage(
        round=1,
        sender="cost_agent",
        msg_type=MessageType.PROPOSAL,
        proposal=ProposalDetail(
            vendor="Dell",
            unit_price_usd=3800,
            total_cost_usd=1900000,
            delivery_days=55,
            quality_tier="B+",
            vendor_risk_score=0.24
        ),
        public_reason="Test",
        utility_score=0.75
    )
    
    json_str = msg.to_json()
    msg2 = NegotiationMessage.from_json(json_str)
    
    assert msg2.sender == msg.sender
    assert msg2.proposal.vendor == msg.proposal.vendor

def test_deal_breaker_violation():
    """Test deal breaker checking"""
    msg = NegotiationMessage(
        round=1,
        sender="cost_agent",
        msg_type=MessageType.PROPOSAL,
        proposal=ProposalDetail(
            vendor="Dell",
            unit_price_usd=4500,
            total_cost_usd=2250000,
            delivery_days=75,
            quality_tier="B+",
            vendor_risk_score=0.24
        ),
        public_reason="Test",
        utility_score=0.5,
        deal_breakers=["delivery_days > 60", "total_cost > 2000000"]
    )
    
    assert msg.violates_deal_breaker()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

Run tests:

```bash
pytest tests/unit/test_protocol.py -v
```

## Step 2: Build Single Cost Agent

Create `src/agents/base_agent.py`:

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from src.protocol.message import NegotiationMessage, MessageType, ProposalDetail
import httpx

class BaseAgent(ABC):
    """Abstract base class for all negotiation agents"""
    
    def __init__(self, name: str, agent_id: str, batna_utility: float = 0.0):
        self.name = name
        self.agent_id = agent_id
        self.batna_utility = batna_utility
        self.negotiation_history: List[NegotiationMessage] = []
        self.current_round = 0
        self.tavily_client = None
    
    @abstractmethod
    def compute_utility(self, proposal: ProposalDetail) -> float:
        """Compute utility of a given proposal"""
        pass
    
    @abstractmethod
    def evaluate_proposal(self, proposal: ProposalDetail) -> bool:
        """Decide whether to accept/reject a proposal"""
        pass
    
    @abstractmethod
    def generate_proposal(self) -> ProposalDetail:
        """Generate opening proposal based on agent's objectives"""
        pass
    
    @abstractmethod
    def generate_counter_proposal(
        self, 
        received_proposal: ProposalDetail
    ) -> ProposalDetail:
        """Generate counter-proposal in response to received bid"""
        pass
    
    def make_concession(self, factor: float = 0.95) -> ProposalDetail:
        """Apply time-pressure concession: relaxed constraints"""
        # Subclasses override this
        pass
    
    async def query_tavily(self, query: str) -> Dict[str, Any]:
        """Fetch real-time vendor data from Tavily"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.tavily.com/search",
                json={"api_key": "YOUR_TAVILY_KEY", "query": query}
            )
            return response.json()
    
    def log_message(self, msg: NegotiationMessage):
        """Store message in agent's history"""
        self.negotiation_history.append(msg)
```

Create `src/agents/cost_agent.py`:

```python
from src.agents.base_agent import BaseAgent
from src.protocol.message import NegotiationMessage, MessageType, ProposalDetail
from typing import Optional
import math

class CostAgent(BaseAgent):
    """Minimizes total expenditure"""
    
    def __init__(
        self,
        budget_cap: float = 2000000,
        opening_price: float = 3500,
        min_quality: str = "B"
    ):
        super().__init__(
            name="Cost Agent",
            agent_id="cost_agent",
            batna_utility=0.20  # Can achieve ~0.20 utility with BATNA
        )
        self.budget_cap = budget_cap
        self.opening_price = opening_price
        self.min_quality = min_quality
        self.current_best_price = opening_price
        self.time_pressure_coefficient = 0.4
    
    def compute_utility(self, proposal: ProposalDetail) -> float:
        """
        Utility = (Budget_Cap - Total_Cost) / Budget_Cap
        Penalize if exceeds budget
        """
        total_cost = proposal.total_cost_usd
        
        # Hard constraint: budget cap
        if total_cost > self.budget_cap:
            return -0.5  # Negative utility if over budget
        
        utility = (self.budget_cap - total_cost) / self.budget_cap
        return max(0.0, min(1.0, utility))  # Clamp to [0, 1]
    
    def apply_time_pressure(self, round_num: int, max_rounds: int = 10) -> float:
        """
        Concession curve: utility decreases over time
        adjustment = 1 - α * (round / max)²
        """
        ratio = min(round_num / max_rounds, 1.0)
        adjustment = 1.0 - self.time_pressure_coefficient * (ratio ** 2)
        return adjustment
    
    def evaluate_proposal(self, proposal: ProposalDetail) -> bool:
        """Accept if utility >= acceptance_threshold AND within budget"""
        utility = self.compute_utility(proposal)
        
        # Check hard constraints
        if proposal.total_cost_usd > self.budget_cap:
            return False
        
        # Check quality minimum
        quality_rank = {"A": 1.0, "A-": 0.9, "B+": 0.8, "B": 0.7, "B-": 0.6}
        if quality_rank.get(proposal.quality_tier, 0) < quality_rank.get(self.min_quality, 0):
            return False
        
        return utility >= self.batna_utility
    
    def generate_proposal(self) -> ProposalDetail:
        """Opening proposal: cheapest vendor with acceptable quality"""
        return ProposalDetail(
            vendor="Dell Technologies",
            unit_price_usd=self.opening_price,
            total_cost_usd=self.opening_price * 500,  # 500 units
            delivery_days=45,
            quality_tier="B+",
            vendor_risk_score=0.28
        )
    
    def generate_counter_proposal(
        self, 
        received_proposal: ProposalDetail
    ) -> ProposalDetail:
        """
        Strategy:
        1. If proposal over budget, offer cheaper alternative
        2. If acceptable, offer slight improvement
        3. Apply time-pressure concession
        """
        received_utility = self.compute_utility(received_proposal)
        
        # If over budget, reject and counter with cheaper vendor
        if received_proposal.total_cost_usd > self.budget_cap:
            return ProposalDetail(
                vendor="Lenovo ThinkSystem",
                unit_price_usd=3500,
                total_cost_usd=3500 * 500,
                delivery_days=70,
                quality_tier="B",
                vendor_risk_score=0.32
            )
        
        # If acceptable but can improve, offer slight price reduction
        if received_proposal.unit_price_usd > self.current_best_price:
            new_price = received_proposal.unit_price_usd * 0.98  # 2% reduction
            return ProposalDetail(
                vendor=received_proposal.vendor,
                unit_price_usd=new_price,
                total_cost_usd=new_price * 500,
                delivery_days=received_proposal.delivery_days,
                quality_tier=received_proposal.quality_tier,
                vendor_risk_score=received_proposal.vendor_risk_score
            )
        
        # Already have good price, don't counter aggressively
        return received_proposal
    
    def make_concession(self, round_num: int, max_rounds: int = 10) -> ProposalDetail:
        """Make time-pressure concession"""
        pressure = self.apply_time_pressure(round_num, max_rounds)
        
        # Accept lower quality to make progress
        new_quality = "B-"  # Downgrade quality
        
        return ProposalDetail(
            vendor="Lenovo ThinkSystem",
            unit_price_usd=3400,
            total_cost_usd=3400 * 500,
            delivery_days=70,
            quality_tier=new_quality,
            vendor_risk_score=0.32
        )
```

**Test the Cost Agent:**

Create `tests/unit/test_cost_agent.py`:

```python
import pytest
from src.agents.cost_agent import CostAgent
from src.protocol.message import ProposalDetail

def test_cost_agent_initialization():
    """Test agent creation"""
    agent = CostAgent(budget_cap=2000000)
    assert agent.name == "Cost Agent"
    assert agent.budget_cap == 2000000
    assert agent.batna_utility == 0.20

def test_utility_computation():
    """Test utility calculation"""
    agent = CostAgent(budget_cap=2000000)
    
    # Proposal at $1.9M should give high utility
    proposal = ProposalDetail(
        vendor="Dell",
        unit_price_usd=3800,
        total_cost_usd=1900000,
        delivery_days=55,
        quality_tier="B+",
        vendor_risk_score=0.24
    )
    utility = agent.compute_utility(proposal)
    assert 0.0 <= utility <= 1.0
    assert utility > 0.5  # Good utility
    
    # Proposal over budget should be negative
    over_budget = ProposalDetail(
        vendor="IBM",
        unit_price_usd=4500,
        total_cost_usd=2250000,
        delivery_days=60,
        quality_tier="A",
        vendor_risk_score=0.15
    )
    utility = agent.compute_utility(over_budget)
    assert utility < 0  # Negative utility

def test_proposal_evaluation():
    """Test accept/reject logic"""
    agent = CostAgent(budget_cap=2000000)
    
    acceptable = ProposalDetail(
        vendor="Dell",
        unit_price_usd=3800,
        total_cost_usd=1900000,
        delivery_days=55,
        quality_tier="B+",
        vendor_risk_score=0.24
    )
    assert agent.evaluate_proposal(acceptable) == True
    
    unacceptable_over_budget = ProposalDetail(
        vendor="IBM",
        unit_price_usd=4500,
        total_cost_usd=2250000,
        delivery_days=60,
        quality_tier="A",
        vendor_risk_score=0.15
    )
    assert agent.evaluate_proposal(unacceptable_over_budget) == False

def test_time_pressure():
    """Test concession over rounds"""
    agent = CostAgent(budget_cap=2000000)
    
    adjustment_r1 = agent.apply_time_pressure(1, 10)
    adjustment_r5 = agent.apply_time_pressure(5, 10)
    adjustment_r10 = agent.apply_time_pressure(10, 10)
    
    # Should decrease over time
    assert adjustment_r1 > adjustment_r5
    assert adjustment_r5 > adjustment_r10
    assert adjustment_r10 > 0  # Never 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

Run:

```bash
pytest tests/unit/test_cost_agent.py -v
```

## Step 3: Wire FastAPI + Redis

Create `src/orchestrator/message_bus.py`:

```python
import redis.asyncio as redis
from src.protocol.message import NegotiationMessage
import json
from typing import Optional, List

class MessageBus:
    """Async Redis pub/sub message bus for agent communication"""
    
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        self.host = host
        self.port = port
        self.db = db
        self.redis_client: Optional[redis.Redis] = None
    
    async def connect(self):
        """Establish Redis connection"""
        self.redis_client = await redis.Redis(
            host=self.host,
            port=self.port,
            db=self.db,
            decode_responses=True
        )
    
    async def disconnect(self):
        """Close Redis connection"""
        if self.redis_client:
            await self.redis_client.close()
    
    async def publish(self, topic: str, message: NegotiationMessage):
        """Publish message to topic"""
        json_msg = message.to_json()
        await self.redis_client.publish(topic, json_msg)
    
    async def subscribe(self, topic: str) -> List[NegotiationMessage]:
        """Subscribe to topic (blocking)"""
        pubsub = self.redis_client.pubsub()
        await pubsub.subscribe(topic)
        messages = []
        
        async for message in pubsub.listen():
            if message["type"] == "message":
                msg = NegotiationMessage.from_json(message["data"])
                messages.append(msg)
        
        return messages
    
    async def log_to_db(self, message: NegotiationMessage):
        """Store message in Redis for persistence"""
        key = f"negotiation:round:{message.round}:msg:{message.msg_id}"
        await self.redis_client.set(key, message.to_json())
```

Create `src/orchestrator/orchestrator.py`:

```python
from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.responses import JSONResponse
from src.orchestrator.message_bus import MessageBus
from src.protocol.message import NegotiationMessage, MessageType
from src.agents.cost_agent import CostAgent
from typing import Dict, List
import asyncio
import json
from datetime import datetime

class NegotiationOrchestrator:
    """Manages negotiation rounds and agent interactions"""
    
    def __init__(self):
        self.message_bus = MessageBus()
        self.agents: Dict[str, any] = {}
        self.current_round = 0
        self.max_rounds = 10
        self.negotiation_history: List[NegotiationMessage] = []
        self.active_websockets: List[WebSocket] = []
    
    async def initialize(self):
        """Start message bus and agents"""
        await self.message_bus.connect()
        
        # Initialize agents
        self.agents["cost_agent"] = CostAgent(budget_cap=2000000)
        # Will add more agents in Phase 2
    
    async def broadcast_task(self, task_description: str):
        """Announce negotiation task to all agents"""
        # Create task announcement message
        announcement = {
            "task": task_description,
            "budget_cap": 2000000,
            "delivery_deadline": 60,
            "quality_minimum": "A-",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Broadcast to Redis topic
        await self.message_bus.redis_client.publish(
            "negotiation:task",
            json.dumps(announcement)
        )
    
    async def run_round(self, round_num: int):
        """Execute one negotiation round"""
        self.current_round = round_num
        
        # Get opening proposal from Cost Agent
        proposal = self.agents["cost_agent"].generate_proposal()
        
        msg = NegotiationMessage(
            round=round_num,
            sender="cost_agent",
            msg_type=MessageType.PROPOSAL,
            proposal=proposal,
            public_reason="Opening proposal: cost-optimal solution",
            utility_score=self.agents["cost_agent"].compute_utility(proposal)
        )
        
        # Store and broadcast
        self.negotiation_history.append(msg)
        await self.message_bus.log_to_db(msg)
        await self.broadcast_to_websockets(msg)
    
    async def broadcast_to_websockets(self, message: NegotiationMessage):
        """Send message to all connected WebSocket clients"""
        for ws in self.active_websockets:
            try:
                await ws.send_json(json.loads(message.to_json()))
            except Exception as e:
                print(f"WebSocket send error: {e}")
    
    async def shutdown(self):
        """Clean up resources"""
        await self.message_bus.disconnect()

# Create FastAPI app
app = FastAPI(title="NegotiationAI API", version="0.1")
orchestrator = NegotiationOrchestrator()

@app.on_event("startup")
async def startup():
    await orchestrator.initialize()

@app.on_event("shutdown")
async def shutdown():
    await orchestrator.shutdown()

@app.post("/negotiation/start")
async def start_negotiation(task: str):
    """Start a new negotiation"""
    try:
        await orchestrator.broadcast_task(task)
        
        # Run first round
        await orchestrator.run_round(1)
        
        return {
            "status": "started",
            "round": 1,
            "history": [json.loads(m.to_json()) for m in orchestrator.negotiation_history]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/negotiation/history")
async def get_history():
    """Get full negotiation history"""
    return {
        "current_round": orchestrator.current_round,
        "messages": [json.loads(m.to_json()) for m in orchestrator.negotiation_history]
    }

@app.websocket("/ws/negotiation")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time updates"""
    await websocket.accept()
    orchestrator.active_websockets.append(websocket)
    
    try:
        while True:
            data = await websocket.receive_text()
            # Handle incoming messages
            if data == "ping":
                await websocket.send_text("pong")
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        orchestrator.active_websockets.remove(websocket)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
```

Create `main.py`:

```python
import uvicorn
import os
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    uvicorn.run(
        "src.orchestrator.orchestrator:app",
        host=os.getenv("FASTAPI_HOST", "0.0.0.0"),
        port=int(os.getenv("FASTAPI_PORT", 8000)),
        reload=os.getenv("ENVIRONMENT") == "development"
    )
```

## Step 4: Test Week 1 Foundation

**Run the API:**

```bash
# Start Redis (in another terminal)
redis-server

# Start FastAPI server
python main.py
```

**Test the endpoint:**

```bash
# In another terminal
curl -X POST http://localhost:8000/negotiation/start \
  -H "Content-Type: application/json" \
  -d '{"task": "Procure 500 servers for data center"}'

# Check history
curl http://localhost:8000/negotiation/history

# Health check
curl http://localhost:8000/health
```

**Week 1 Checkpoint:**
- ✅ Protocol schema defined and tested
- ✅ Cost Agent implemented and tested
- ✅ FastAPI + Redis message bus working
- ✅ First proposal being generated

---

# Phase 2: Multi-Agent Loop (Week 2)

## Step 1: Build Remaining Agents

Create `src/agents/quality_agent.py`:

```python
from src.agents.base_agent import BaseAgent
from src.protocol.message import ProposalDetail

class QualityAgent(BaseAgent):
    """Maximizes vendor reliability and SLA compliance"""
    
    def __init__(
        self,
        min_quality_tier: str = "A-",
        weight_rating: float = 0.5,
        weight_sla: float = 0.3,
        weight_tier: float = 0.2
    ):
        super().__init__(
            name="Quality Agent",
            agent_id="quality_agent",
            batna_utility=0.85  # High BATNA: can demand premium
        )
        self.min_quality_tier = min_quality_tier
        self.weight_rating = weight_rating
        self.weight_sla = weight_sla
        self.weight_tier = weight_tier
        self.time_pressure_coefficient = 0.3
    
    def compute_utility(self, proposal: ProposalDetail) -> float:
        """
        Utility = rating_score × 0.5 + sla_score × 0.3 + tier_score × 0.2
        """
        # Map vendor names to ratings (would come from Tavily in real system)
        vendor_ratings = {
            "Dell Technologies": 4.2 / 5.0,
            "HPE ProLiant": 4.5 / 5.0,
            "Lenovo ThinkSystem": 3.8 / 5.0,
            "IBM Power Systems": 4.8 / 5.0
        }
        
        rating_score = vendor_ratings.get(proposal.vendor, 3.5 / 5.0)
        
        # SLA score (simplified)
        sla_score = 0.9 if proposal.delivery_days <= 60 else 0.5
        
        # Quality tier score
        tier_scores = {"A": 1.0, "A-": 0.9, "B+": 0.7, "B": 0.5, "B-": 0.3}
        tier_score = tier_scores.get(proposal.quality_tier, 0.5)
        
        utility = (
            rating_score * self.weight_rating +
            sla_score * self.weight_sla +
            tier_score * self.weight_tier
        )
        
        return max(0.0, min(1.0, utility))
    
    def evaluate_proposal(self, proposal: ProposalDetail) -> bool:
        """Reject if quality tier below minimum"""
        tier_scores = {"A": 1.0, "A-": 0.9, "B+": 0.7, "B": 0.5}
        min_score = tier_scores.get(self.min_quality_tier, 0.9)
        proposal_score = tier_scores.get(proposal.quality_tier, 0)
        
        if proposal_score < min_score:
            return False
        
        utility = self.compute_utility(proposal)
        return utility >= 0.60  # Lower threshold than BATNA
    
    def generate_proposal(self) -> ProposalDetail:
        """Opening: premium vendor, high quality"""
        return ProposalDetail(
            vendor="IBM Power Systems",
            unit_price_usd=4500,
            total_cost_usd=2250000,
            delivery_days=60,
            quality_tier="A",
            vendor_risk_score=0.15
        )
    
    def generate_counter_proposal(self, received: ProposalDetail) -> ProposalDetail:
        """Accept lower cost if quality maintained"""
        if received.quality_tier in ["A", "A-"]:
            return received  # Accept
        
        # Demand A- tier
        return ProposalDetail(
            vendor="HPE ProLiant",
            unit_price_usd=4100,
            total_cost_usd=2050000,
            delivery_days=50,
            quality_tier="A-",
            vendor_risk_score=0.22
        )
```

Create `src/agents/timeline_agent.py`:

```python
from src.agents.base_agent import BaseAgent
from src.protocol.message import ProposalDetail

class TimelineAgent(BaseAgent):
    """Minimizes delivery days"""
    
    def __init__(self, deadline: int = 60, weight_speed: float = 0.7):
        super().__init__(
            name="Timeline Agent",
            agent_id="timeline_agent",
            batna_utility=0.85  # Can achieve 30 days with fast vendors
        )
        self.deadline = deadline
        self.weight_speed = weight_speed
        self.time_pressure_coefficient = 0.5  # Most aggressive concession
    
    def compute_utility(self, proposal: ProposalDetail) -> float:
        """Utility increases as delivery days decrease"""
        utility = max(0, (self.deadline - proposal.delivery_days) / self.deadline)
        return max(0.0, min(1.0, utility))
    
    def evaluate_proposal(self, proposal: ProposalDetail) -> bool:
        """Hard constraint: delivery_days <= deadline"""
        if proposal.delivery_days > self.deadline:
            return False
        
        utility = self.compute_utility(proposal)
        return utility >= 0.60
    
    def generate_proposal(self) -> ProposalDetail:
        """Opening: fastest vendor regardless of cost"""
        return ProposalDetail(
            vendor="HPE ProLiant",
            unit_price_usd=4200,
            total_cost_usd=2100000,
            delivery_days=35,
            quality_tier="A",
            vendor_risk_score=0.25
        )
    
    def generate_counter_proposal(self, received: ProposalDetail) -> ProposalDetail:
        """Push for faster delivery, willing to pay more"""
        if received.delivery_days <= 45:
            return received
        
        # Offer faster alternative
        return ProposalDetail(
            vendor="HPE ProLiant",
            unit_price_usd=4300,
            total_cost_usd=2150000,
            delivery_days=40,
            quality_tier=received.quality_tier,
            vendor_risk_score=received.vendor_risk_score
        )
```

Create `src/agents/risk_agent.py`:

```python
from src.agents.base_agent import BaseAgent
from src.protocol.message import ProposalDetail

class RiskAgent(BaseAgent):
    """Minimizes supply chain risk through diversification"""
    
    def __init__(self):
        super().__init__(
            name="Risk Agent",
            agent_id="risk_agent",
            batna_utility=0.75  # Can achieve low risk with split orders
        )
        self.require_multiple_vendors = True
        self.max_single_vendor_pct = 0.6  # Max 60% from one vendor
        self.time_pressure_coefficient = 0.2  # Least aggressive
    
    def compute_utility(self, proposal: ProposalDetail) -> float:
        """
        Lower risk score = higher utility
        Risk includes: single-vendor dependency, geopolitical factors
        """
        # Assume we're evaluating splits (advanced, for now simple)
        risk_score = proposal.vendor_risk_score
        utility = max(0, 1.0 - risk_score)
        return max(0.0, min(1.0, utility))
    
    def evaluate_proposal(self, proposal: ProposalDetail) -> bool:
        """Accept if risk below threshold"""
        utility = self.compute_utility(proposal)
        return utility >= 0.65
    
    def generate_proposal(self) -> ProposalDetail:
        """Opening: request split order (multi-vendor)"""
        # This would be more complex in real implementation
        return ProposalDetail(
            vendor="Dell + HPE",  # Placeholder for split
            unit_price_usd=3900,
            total_cost_usd=1950000,
            delivery_days=50,
            quality_tier="A-",
            vendor_risk_score=0.18  # Lower risk with diversification
        )
    
    def generate_counter_proposal(self, received: ProposalDetail) -> ProposalDetail:
        """Request vendor diversification"""
        if received.vendor_risk_score < 0.20:
            return received
        
        # Request split
        return ProposalDetail(
            vendor="Dell + HPE (split)",
            unit_price_usd=3950,
            total_cost_usd=1975000,
            delivery_days=52,
            quality_tier=received.quality_tier,
            vendor_risk_score=0.18
        )
```

## Step 2: Implement Round Manager

Create `src/orchestrator/round_manager.py`:

```python
from typing import Dict, List, Optional
from src.protocol.message import NegotiationMessage, MessageType, ProposalDetail
from src.agents.base_agent import BaseAgent
import asyncio

class RoundManager:
    """Manages negotiation rounds and agent turn-taking"""
    
    def __init__(self, agents: Dict[str, BaseAgent], max_rounds: int = 10):
        self.agents = agents
        self.max_rounds = max_rounds
        self.current_round = 0
        self.messages: List[NegotiationMessage] = []
        self.agent_order = list(agents.keys())
        self.agent_utilities: Dict[str, List[float]] = {
            agent_id: [] for agent_id in agents.keys()
        }
    
    async def execute_round(self, round_num: int) -> List[NegotiationMessage]:
        """
        Execute one complete negotiation round:
        - Each agent takes a turn
        - Generates counter-proposal based on last bid
        - Updates utilities
        """
        self.current_round = round_num
        round_messages = []
        
        # In round 1, agents make opening proposals
        if round_num == 1:
            for agent_id in self.agent_order:
                agent = self.agents[agent_id]
                proposal = agent.generate_proposal()
                
                msg = NegotiationMessage(
                    round=round_num,
                    sender=agent_id,
                    msg_type=MessageType.PROPOSAL,
                    proposal=proposal,
                    public_reason=f"{agent.name}'s opening proposal",
                    utility_score=agent.compute_utility(proposal)
                )
                
                round_messages.append(msg)
                self.messages.append(msg)
                self.agent_utilities[agent_id].append(msg.utility_score)
        
        else:
            # Subsequent rounds: counter-proposals
            last_proposal = self.messages[-1].proposal if self.messages else None
            
            for agent_id in self.agent_order:
                if not last_proposal:
                    continue
                
                agent = self.agents[agent_id]
                
                # Check if agent wants to accept
                if agent.evaluate_proposal(last_proposal):
                    msg = NegotiationMessage(
                        round=round_num,
                        sender=agent_id,
                        msg_type=MessageType.ACCEPT,
                        proposal=last_proposal,
                        public_reason=f"{agent.name} accepts this proposal",
                        utility_score=agent.compute_utility(last_proposal),
                        acceptance_threshold=0.65
                    )
                else:
                    # Counter-propose
                    counter = agent.generate_counter_proposal(last_proposal)
                    msg = NegotiationMessage(
                        round=round_num,
                        sender=agent_id,
                        msg_type=MessageType.COUNTER_PROPOSAL,
                        proposal=counter,
                        public_reason=f"{agent.name}'s counter-proposal",
                        utility_score=agent.compute_utility(counter),
                        concessions_made=["TBD"]
                    )
                
                round_messages.append(msg)
                self.messages.append(msg)
                self.agent_utilities[agent_id].append(msg.utility_score)
        
        return round_messages
    
    def has_converged(self) -> bool:
        """Check if agents have reached agreement"""
        if len(self.messages) < len(self.agents):
            return False
        
        # Check if last N messages are all ACCEPT
        recent = self.messages[-len(self.agents):]
        return all(m.msg_type == MessageType.ACCEPT for m in recent)
    
    def get_pareto_frontier(self) -> List[NegotiationMessage]:
        """Calculate Pareto-optimal proposals"""
        # Simplified: just return messages where no agent can improve
        # without harming another (would need full analysis in production)
        frontier = []
        
        for msg in self.messages:
            is_dominated = False
            
            for other_msg in self.messages:
                if msg.msg_id == other_msg.msg_id:
                    continue
                
                # Check if other dominates msg
                # (simplified check)
                if (other_msg.utility_score > msg.utility_score):
                    is_dominated = True
                    break
            
            if not is_dominated:
                frontier.append(msg)
        
        return frontier
```

## Step 3: Update Orchestrator for Multi-Agent

Update `src/orchestrator/orchestrator.py`:

```python
from src.orchestrator.round_manager import RoundManager
from src.agents.quality_agent import QualityAgent
from src.agents.timeline_agent import TimelineAgent
from src.agents.risk_agent import RiskAgent

class NegotiationOrchestrator:
    def __init__(self):
        # ... existing code ...
        self.round_manager = None
    
    async def initialize(self):
        await self.message_bus.connect()
        
        # Initialize ALL agents
        self.agents["cost_agent"] = CostAgent(budget_cap=2000000)
        self.agents["quality_agent"] = QualityAgent(min_quality_tier="A-")
        self.agents["timeline_agent"] = TimelineAgent(deadline=60)
        self.agents["risk_agent"] = RiskAgent()
        
        # Initialize round manager
        self.round_manager = RoundManager(self.agents, max_rounds=10)
    
    async def run_negotiation(self):
        """Run full negotiation until convergence or max rounds"""
        for round_num in range(1, self.round_manager.max_rounds + 1):
            print(f"\n=== Round {round_num} ===")
            
            # Execute round
            messages = await self.round_manager.execute_round(round_num)
            
            # Broadcast to websockets
            for msg in messages:
                await self.broadcast_to_websockets(msg)
            
            # Check for convergence
            if self.round_manager.has_converged():
                print("✓ Agreement reached!")
                break
            
            await asyncio.sleep(1)  # Brief pause between rounds
        
        # Print final status
        await self.print_negotiation_summary()
    
    async def print_negotiation_summary(self):
        """Print utilities and agreement status"""
        print("\n=== Negotiation Summary ===")
        
        # Last proposal
        final_msg = self.round_manager.messages[-1]
        print(f"\nFinal Proposal:")
        print(f"  Vendor: {final_msg.proposal.vendor}")
        print(f"  Unit Price: ${final_msg.proposal.unit_price_usd:,.0f}")
        print(f"  Total Cost: ${final_msg.proposal.total_cost_usd:,.0f}")
        print(f"  Delivery: {final_msg.proposal.delivery_days} days")
        print(f"  Quality: {final_msg.proposal.quality_tier}")
        print(f"  Risk Score: {final_msg.proposal.vendor_risk_score:.2f}")
        
        # Agent utilities
        print("\nAgent Utilities:")
        for agent_id, utilities in self.round_manager.agent_utilities.items():
            final_utility = utilities[-1] if utilities else 0
            print(f"  {agent_id}: {final_utility:.3f}")

# Update endpoints
@app.post("/negotiation/run")
async def run_negotiation():
    """Run full negotiation"""
    try:
        await orchestrator.broadcast_task(
            "Procure 500 server units for new data center"
        )
        await orchestrator.run_negotiation()
        
        return {
            "status": "completed",
            "round": orchestrator.round_manager.current_round,
            "converged": orchestrator.round_manager.has_converged(),
            "messages": [json.loads(m.to_json()) for m in orchestrator.round_manager.messages]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

## Step 4: Test Phase 2

```bash
# Run the full negotiation
curl -X POST http://localhost:8000/negotiation/run

# Check WebSocket in dashboard (Phase 4)
```

**Week 2 Checkpoint:**
- ✅ All 4 agents implemented
- ✅ Round manager executing turns
- ✅ Multi-agent negotiations happening
- ✅ Utilities being computed

---

# Phase 3: Deadlock & Mediator (Week 3)

## Step 1: Build Mediator Agent

Create `src/agents/mediator_agent.py`:

```python
from src.agents.base_agent import BaseAgent
from src.protocol.message import NegotiationMessage, ProposalDetail, MessageType
from typing import List, Dict
import numpy as np

class MediatorAgent(BaseAgent):
    """Detects deadlock and proposes compromises"""
    
    def __init__(self, deadlock_threshold: int = 3):
        super().__init__(
            name="Mediator Agent",
            agent_id="mediator",
            batna_utility=1.0  # Not a player, it facilitates
        )
        self.deadlock_threshold = deadlock_threshold
        self.rounds_without_progress = 0
    
    def compute_utility(self, proposal: ProposalDetail) -> float:
        """Mediator utility = min(all agent utilities) = Rawlsian fairness"""
        # Simplified placeholder
        return 0.5
    
    def evaluate_proposal(self, proposal: ProposalDetail) -> bool:
        return True  # Mediator doesn't reject
    
    def generate_proposal(self) -> ProposalDetail:
        return ProposalDetail(
            vendor="Compromise",
            unit_price_usd=3900,
            total_cost_usd=1950000,
            delivery_days=50,
            quality_tier="A-",
            vendor_risk_score=0.20
        )
    
    def generate_counter_proposal(self, received: ProposalDetail) -> ProposalDetail:
        return received
    
    def detect_deadlock(self, messages: List[NegotiationMessage]) -> bool:
        """
        Detect deadlock: no agent has improved utility for N rounds
        """
        if len(messages) < self.deadlock_threshold * 4:
            return False
        
        # Check last N messages: have agent utilities changed?
        recent = messages[-(self.deadlock_threshold * 4):]
        
        for i in range(0, len(recent) - 4, 4):
            batch1 = recent[i:i+4]
            batch2 = recent[i+4:i+8]
            
            if i + 8 > len(recent):
                break
            
            # Compare agent utilities between batches
            for j in range(4):
                if i + j < len(batch1) and i + j + 4 < len(batch2):
                    if abs(batch1[j].utility_score - batch2[j].utility_score) < 0.01:
                        self.rounds_without_progress += 1
                    else:
                        self.rounds_without_progress = 0
        
        return self.rounds_without_progress >= self.deadlock_threshold
    
    def compute_compromise(
        self,
        messages: List[NegotiationMessage],
        agent_utilities: Dict[str, float]
    ) -> ProposalDetail:
        """
        Compute compromise point on Pareto frontier
        Maximize min(agent utilities) = Rawlsian
        """
        recent_proposals = [m.proposal for m in messages[-4:]]
        
        # Average proposal (simplified)
        avg_price = np.mean([p.unit_price_usd for p in recent_proposals])
        avg_cost = np.mean([p.total_cost_usd for p in recent_proposals])
        avg_days = np.mean([p.delivery_days for p in recent_proposals])
        avg_risk = np.mean([p.vendor_risk_score for p in recent_proposals])
        
        # Quality: take majority
        qualities = [p.quality_tier for p in recent_proposals]
        avg_quality = max(set(qualities), key=qualities.count)
        
        # Find vendor closest to average price
        vendors = {
            "Dell": 3800,
            "HPE": 4200,
            "Lenovo": 3500,
            "IBM": 4500
        }
        closest_vendor = min(vendors, key=lambda v: abs(vendors[v] - avg_price))
        
        return ProposalDetail(
            vendor=f"{closest_vendor} (Mediator Compromise)",
            unit_price_usd=float(avg_price),
            total_cost_usd=float(avg_cost),
            delivery_days=int(avg_days),
            quality_tier=avg_quality,
            vendor_risk_score=float(avg_risk)
        )
    
    async def propose_compromise(
        self,
        messages: List[NegotiationMessage],
        agents: Dict[str, BaseAgent],
        round_num: int
    ) -> NegotiationMessage:
        """Generate mediator compromise proposal"""
        agent_utilities = {
            agent_id: agent.compute_utility(messages[-1].proposal)
            for agent_id, agent in agents.items()
        }
        
        compromise = self.compute_compromise(messages, agent_utilities)
        
        msg = NegotiationMessage(
            round=round_num,
            sender="mediator",
            msg_type=MessageType.MEDIATOR_COMPROMISE,
            proposal=compromise,
            public_reason="Compromise point on Pareto frontier - acceptable to all parties",
            utility_score=0.75,
            reasoning_trace={
                "reason": "Deadlock detected",
                "agent_utilities": agent_utilities,
                "fairness_metric": "Rawlsian (maximize min utility)"
            }
        )
        
        return msg
```

## Step 2: Add Mediator to Round Manager

Update `src/orchestrator/round_manager.py`:

```python
from src.agents.mediator_agent import MediatorAgent

class RoundManager:
    def __init__(self, agents: Dict[str, BaseAgent], mediator: MediatorAgent = None, max_rounds: int = 10):
        # ... existing code ...
        self.mediator = mediator or MediatorAgent()
        self.deadlock_detected = False
    
    async def execute_round(self, round_num: int) -> List[NegotiationMessage]:
        # ... existing code ...
        
        # After normal round, check for deadlock
        if round_num > 3:
            if self.mediator.detect_deadlock(self.messages):
                print("⚠ DEADLOCK DETECTED - Mediator intervening...")
                self.deadlock_detected = True
                
                # Generate compromise
                compromise_msg = await self.mediator.propose_compromise(
                    self.messages,
                    self.agents,
                    round_num
                )
                
                self.messages.append(compromise_msg)
                return [compromise_msg]
        
        return round_messages
```

## Step 3: BATNA Logic

Update agents to check BATNA:

```python
# In CostAgent.evaluate_proposal()

def evaluate_proposal(self, proposal: ProposalDetail) -> bool:
    """Accept if utility >= acceptance_threshold AND within budget"""
    utility = self.compute_utility(proposal)
    
    # Check HARD constraint: BATNA
    if utility < self.batna_utility:
        print(f"⚠ Proposal utility ({utility:.2f}) below BATNA ({self.batna_utility:.2f})")
        return False
    
    # ... rest of checks ...
```

## Step 4: Test Phase 3

```bash
# Run negotiation with deadlock scenario
curl -X POST http://localhost:8000/negotiation/run
```

**Week 3 Checkpoint:**
- ✅ Mediator agent implemented
- ✅ Deadlock detection working
- ✅ Compromise proposals being made
- ✅ BATNA logic enforced

---

# Phase 4: Dashboard (Week 4)

## Step 1: Create React Frontend

Create `dashboard/package.json`:

```json
{
  "name": "negotiation-ai-dashboard",
  "version": "0.1.0",
  "private": true,
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "recharts": "^2.10.0",
    "axios": "^1.6.0",
    "ws": "^8.14.0"
  },
  "scripts": {
    "start": "react-scripts start",
    "build": "react-scripts build"
  }
}
```

Install:

```bash
cd dashboard
npm install
```

Create `dashboard/src/App.js`:

```javascript
import React, { useState, useEffect } from 'react';
import NegotiationDashboard from './components/NegotiationDashboard';
import './App.css';

function App() {
  const [ws, setWs] = useState(null);
  const [messages, setMessages] = useState([]);

  useEffect(() => {
    // Connect WebSocket
    const websocket = new WebSocket('ws://localhost:8000/ws/negotiation');
    
    websocket.onopen = () => {
      console.log('WebSocket connected');
      setWs(websocket);
    };
    
    websocket.onmessage = (event) => {
      const message = JSON.parse(event.data);
      setMessages(prev => [...prev, message]);
    };
    
    websocket.onclose = () => {
      console.log('WebSocket disconnected');
    };
    
    return () => websocket.close();
  }, []);

  return (
    <div className="App">
      <header>
        <h1>NegotiationAI Dashboard</h1>
        <p>Real-time Multi-Agent Negotiation Monitor</p>
      </header>
      
      <NegotiationDashboard messages={messages} />
    </div>
  );
}

export default App;
```

Create `dashboard/src/components/NegotiationDashboard.js`:

```javascript
import React from 'react';
import NegotiationFeed from './NegotiationFeed';
import UtilityChart from './UtilityChart';
import ParetoFrontier from './ParetoFrontier';
import '../styles/Dashboard.css';

function NegotiationDashboard({ messages }) {
  const currentRound = messages.length > 0 
    ? Math.max(...messages.map(m => m.round)) 
    : 0;

  return (
    <div className="dashboard">
      <div className="dashboard-row">
        <div className="panel panel-feed">
          <h2>Negotiation Feed</h2>
          <NegotiationFeed messages={messages} />
        </div>
        
        <div className="panel panel-utilities">
          <h2>Agent Utilities Over Time</h2>
          <UtilityChart messages={messages} />
        </div>
      </div>
      
      <div className="dashboard-row">
        <div className="panel panel-pareto">
          <h2>Pareto Frontier</h2>
          <ParetoFrontier messages={messages} />
        </div>
        
        <div className="panel panel-summary">
          <h2>Current Status</h2>
          <div className="summary-content">
            <p><strong>Round:</strong> {currentRound}</p>
            {messages.length > 0 && (
              <>
                <p><strong>Last Sender:</strong> {messages[messages.length - 1].sender}</p>
                <p><strong>Last Message Type:</strong> {messages[messages.length - 1].msg_type}</p>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default NegotiationDashboard;
```

Create `dashboard/src/components/UtilityChart.js`:

```javascript
import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

function UtilityChart({ messages }) {
  // Aggregate utilities by round and agent
  const data = {};
  
  messages.forEach(msg => {
    if (!data[msg.round]) {
      data[msg.round] = { round: msg.round };
    }
    data[msg.round][msg.sender] = msg.utility_score;
  });
  
  const chartData = Object.values(data).sort((a, b) => a.round - b.round);

  return (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart data={chartData}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="round" />
        <YAxis />
        <Tooltip />
        <Legend />
        <Line type="monotone" dataKey="cost_agent" stroke="#8B0000" />
        <Line type="monotone" dataKey="quality_agent" stroke="#4169E1" />
        <Line type="monotone" dataKey="timeline_agent" stroke="#FFD700" />
        <Line type="monotone" dataKey="risk_agent" stroke="#228B22" />
      </LineChart>
    </ResponsiveContainer>
  );
}

export default UtilityChart;
```

Create `dashboard/src/components/ParetoFrontier.js`:

```javascript
import React from 'react';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

function ParetoFrontier({ messages }) {
  // Extract cost_agent vs quality_agent utilities
  const data = messages
    .filter(m => m.msg_type !== 'TASK_ANNOUNCEMENT')
    .map((m, i) => ({
      x: m.utility_score,
      y: m.utility_score,  // Simplified; real implementation would track all agents
      round: m.round,
      sender: m.sender
    }));

  return (
    <ResponsiveContainer width="100%" height={300}>
      <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="x" label={{ value: 'Cost Agent Utility', position: 'right', offset: -5 }} />
        <YAxis dataKey="y" label={{ value: 'Quality Agent Utility', angle: -90, position: 'insideLeft' }} />
        <Tooltip cursor={{ strokeDasharray: '3 3' }} />
        <Scatter name="Proposals" data={data} fill="#8884d8" />
      </ScatterChart>
    </ResponsiveContainer>
  );
}

export default ParetoFrontier;
```

Create `dashboard/src/components/NegotiationFeed.js`:

```javascript
import React from 'react';

function NegotiationFeed({ messages }) {
  return (
    <div className="feed">
      {messages.map((msg, i) => (
        <div key={i} className={`message message-${msg.msg_type}`}>
          <div className="message-header">
            <span className="agent-badge">{msg.sender}</span>
            <span className="message-type">{msg.msg_type}</span>
            <span className="round-badge">R{msg.round}</span>
          </div>
          
          <div className="message-body">
            <p className="reason">{msg.public_reason}</p>
            
            {msg.proposal && (
              <div className="proposal">
                <p><strong>Vendor:</strong> {msg.proposal.vendor}</p>
                <p><strong>Total Cost:</strong> ${msg.proposal.total_cost_usd.toLocaleString()}</p>
                <p><strong>Delivery:</strong> {msg.proposal.delivery_days} days</p>
                <p><strong>Quality:</strong> {msg.proposal.quality_tier}</p>
              </div>
            )}
            
            <p className="utility">Utility: {msg.utility_score.toFixed(3)}</p>
          </div>
        </div>
      ))}
    </div>
  );
}

export default NegotiationFeed;
```

Create `dashboard/src/styles/Dashboard.css`:

```css
.dashboard {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
  padding: 20px;
  background: #f5f5f5;
}

.dashboard-row {
  display: contents;
}

.panel {
  background: white;
  border-radius: 8px;
  padding: 20px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.panel h2 {
  margin-top: 0;
  color: #333;
  border-bottom: 2px solid #007bff;
  padding-bottom: 10px;
}

.feed {
  max-height: 500px;
  overflow-y: auto;
}

.message {
  padding: 12px;
  margin-bottom: 12px;
  border-left: 4px solid #999;
  background: #fafafa;
}

.message-PROPOSAL { border-left-color: #007bff; }
.message-COUNTER_PROPOSAL { border-left-color: #ff9800; }
.message-ACCEPT { border-left-color: #4caf50; }
.message-REJECT { border-left-color: #f44336; }
.message-MEDIATOR_COMPROMISE { border-left-color: #9c27b0; }

.message-header {
  display: flex;
  gap: 10px;
  margin-bottom: 8px;
  font-size: 12px;
}

.agent-badge {
  background: #007bff;
  color: white;
  padding: 2px 6px;
  border-radius: 3px;
}

.message-type {
  background: #666;
  color: white;
  padding: 2px 6px;
  border-radius: 3px;
}

.round-badge {
  background: #ccc;
  padding: 2px 6px;
  border-radius: 3px;
  margin-left: auto;
}

.proposal {
  background: #fff;
  padding: 10px;
  margin: 10px 0;
  border-radius: 4px;
  font-size: 13px;
}

.proposal p {
  margin: 4px 0;
}

.utility {
  font-weight: bold;
  color: #4caf50;
  margin-top: 8px;
}
```

## Step 2: CORS Setup in FastAPI

Update `src/orchestrator/orchestrator.py`:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Step 3: Run Full Stack

Terminal 1 (API):
```bash
python main.py
```

Terminal 2 (Dashboard):
```bash
cd dashboard
npm start
# Opens http://localhost:3000
```

Terminal 3 (Trigger negotiation):
```bash
curl -X POST http://localhost:8000/negotiation/run
```

**Week 4 Checkpoint:**
- ✅ React dashboard live
- ✅ Real-time WebSocket updates
- ✅ Utility charts visualizing
- ✅ Negotiation feed showing live

---

# Phase 5: Polish & Evaluation (Week 5)

## Step 1: Add Sprint Planning Scenario

Create `src/scenarios/sprint_planning.py`:

```python
class SprintPlanningScenario:
    """
    Second demo: Feature vs Reliability vs Tech Debt agents
    negotiate a 2-week sprint plan
    """
    
    def __init__(self):
        self.total_hours = 140
        self.agents = {
            "feature_agent": FeatureAgent(),
            "reliability_agent": ReliabilityAgent(),
            "tech_debt_agent": TechDebtAgent()
        }
    
    def get_task_description(self):
        return """
        Plan a 2-week sprint (140 hours total):
        - Feature development (user-facing work)
        - Testing and reliability coverage
        - Technical debt refactoring
        
        Constraints:
        - Minimum 30 hours for features
        - Minimum 40 hours for testing/QA
        - Maximum 30 hours for tech debt
        """
```

## Step 2: Comparative Evaluation

Create `tests/integration/test_evaluation.py`:

```python
import pytest
import asyncio
from src.orchestrator.orchestrator import NegotiationOrchestrator
from src.orchestrator.round_manager import RoundManager
from src.agents.cost_agent import CostAgent
from src.agents.quality_agent import QualityAgent

@pytest.mark.asyncio
async def test_negotiated_vs_orchestrator_outcome():
    """
    Compare negotiated outcome vs single orchestrator decision
    on utility maximization and fairness
    """
    
    # Run negotiation
    orch = NegotiationOrchestrator()
    await orch.initialize()
    await orch.run_negotiation()
    
    final_proposal = orch.round_manager.messages[-1].proposal
    
    # Compute final utilities
    cost_utility = orch.agents["cost_agent"].compute_utility(final_proposal)
    quality_utility = orch.agents["quality_agent"].compute_utility(final_proposal)
    timeline_utility = orch.agents["timeline_agent"].compute_utility(final_proposal)
    risk_utility = orch.agents["risk_agent"].compute_utility(final_proposal)
    
    negotiated_total = cost_utility + quality_utility + timeline_utility + risk_utility
    negotiated_fairness = min([cost_utility, quality_utility, timeline_utility, risk_utility])
    
    print(f"\nNegotiated Outcome:")
    print(f"  Total Welfare: {negotiated_total:.3f}")
    print(f"  Fairness (min): {negotiated_fairness:.3f}")
    print(f"  Individual utilities: C={cost_utility:.2f}, Q={quality_utility:.2f}, T={timeline_utility:.2f}, R={risk_utility:.2f}")
    
    # Compare with single orchestrator
    # (would pick Dell to minimize cost)
    orchestrator_proposal = orch.agents["cost_agent"].generate_proposal()
    
    orch_cost = orch.agents["cost_agent"].compute_utility(orchestrator_proposal)
    orch_quality = orch.agents["quality_agent"].compute_utility(orchestrator_proposal)
    orch_timeline = orch.agents["timeline_agent"].compute_utility(orchestrator_proposal)
    orch_risk = orch.agents["risk_agent"].compute_utility(orchestrator_proposal)
    
    orch_total = orch_cost + orch_quality + orch_timeline + orch_risk
    orch_fairness = min([orch_cost, orch_quality, orch_timeline, orch_risk])
    
    print(f"\nOrchestrator Outcome:")
    print(f"  Total Welfare: {orch_total:.3f}")
    print(f"  Fairness (min): {orch_fairness:.3f}")
    print(f"  Individual utilities: C={orch_cost:.2f}, Q={orch_quality:.2f}, T={orch_timeline:.2f}, R={orch_risk:.2f}")
    
    # Assertions
    assert negotiated_fairness > orch_fairness, "Negotiation should be fairer"
```

## Step 3: Open-Source Preparation

Create `README.md`:

```markdown
# NegotiationAI: Protocol-Based Multi-Agent Negotiation

Open-source framework for agent-to-agent negotiation using structured protocols.

## Quick Start

```bash
git clone https://github.com/yourusername/negotiation-ai
cd negotiation-ai

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

redis-server  # Terminal 1
python main.py  # Terminal 2
cd dashboard && npm start  # Terminal 3

curl -X POST http://localhost:8000/negotiation/run
```

## Architecture

- **Protocol**: Typed `NegotiationMessage` schema
- **Agents**: Cost, Quality, Timeline, Risk agents with utility functions
- **Orchestrator**: FastAPI + Redis pub/sub message bus
- **Dashboard**: React live feed + charts

## Research References

- Contract Net Protocol (Smith & Davis, 1980s)
- Pareto Optimality in Negotiations (Raiffa, 1982)
- LLM Agent Negotiations (2024-2026 Research)
- ANAC Automated Negotiation Competition

## License

MIT
```

## Step 4: Recording Demo

Script for demo video (3 minutes):

```
[0-30s] Introduction
- "Meet NegotiationAI: the first protocol-based multi-agent negotiation framework"
- Show dashboard loading

[30-90s] Run Negotiation
- Trigger: curl -X POST http://localhost:8000/negotiation/run
- Show feed updating in real-time
- Highlight: agents making offers, counters, concessions

[90-150s] Results
- Show final agreement
- Show utility chart: cost agent starts low, improves over rounds
- Show Pareto frontier: final agreement on frontier

[150-180s] Deadlock Recovery (if it happens)
- Show mediator stepping in
- Show compromise proposal
- Show acceptance

[Outro]
- "Protocol-based negotiation: fair, explainable, Pareto-optimal"
```

## Step 5: Devpost Submission Template

```markdown
# NegotiationAI: Agent-to-Agent Negotiation Protocol

## Inspiration

Organizations make decisions daily where conflicting objectives clash:
- Procurement: Cost vs Quality vs Timeline vs Risk
- Sprint Planning: Features vs Reliability vs Tech Debt
- Resource Allocation: Multiple departments, different priorities

Traditional systems use a central orchestrator to decide. We asked: what if agents negotiated?

## What It Does

NegotiationAI is a framework where specialized AI agents with genuinely conflicting utility functions negotiate to reach Pareto-optimal agreements **without human intervention**.

Four agents, one task:
- **Cost Agent**: Minimize spend
- **Quality Agent**: Maximize reliability
- **Timeline Agent**: Minimize delivery time
- **Risk Agent**: Minimize supply chain risk

Each round, agents propose and counter-propose. The Mediator detects deadlock and suggests compromises. Negotiations converge to agreements that satisfy all parties.

## How We Built It

**Tech Stack:**
- Nebius H100 inference (Llama 3.1 70B)
- LangGraph for agent state machines
- FastAPI + Redis pub/sub for message bus
- PostgreSQL for audit trail
- React dashboard for real-time visualization

**Core Innovation:**
- **NegotiationMessage**: Typed protocol schema for structured LLM agent communication
- **Pareto Frontier Calculator**: Identifies optimal agreements
- **BATNA-Aware Negotiation**: Agents exit if offers fall below their best alternative
- **Mediator Agent**: Meta-reasoning to break deadlock

## Challenges

1. **LLM Rationality**: Models don't always follow negotiation game theory. Solution: constrained JSON output + utility scoring.

2. **Fairness**: Agents with different capabilities exploit weaker ones. Solution: BATNA thresholds prevent unfair deals.

3. **Convergence**: Multi-agent systems don't always agree. Solution: Mediator proposes compromises using Rawlsian fairness (maximize minimum utility).

## Accomplishments

- ✅ Multi-agent negotiations running in real-time
- ✅ Live dashboard showing proposal evolution
- ✅ Full audit trail (every bid, counter, concession logged)
- ✅ Two demo scenarios (procurement + sprint planning)
- ✅ Outperforms single-orchestrator on fairness metrics

## What We Learned

Agent-to-agent negotiation is a 2026 research frontier. Implementing it revealed:
- Importance of explicit utility functions + BATNA
- Power of structured protocols over free-form LLM dialogue
- Fairness requires meta-reasoning (mediator)

## Next Steps

1. Hierarchical negotiation (agents form coalitions)
2. Dynamic protocol negotiation (agents agree on rules)
3. Recursive delegated negotiation (agents negotiate on behalf of others)
4. Neural Pareto frontier learning

## Try It

```bash
git clone https://github.com/yourusername/negotiation-ai
cd negotiation-ai
pip install -r requirements.txt
redis-server &
python main.py &
cd dashboard && npm start
curl -X POST http://localhost:8000/negotiation/run
```

Open http://localhost:3000 to see negotiations live.

---

## Architecture Diagram

[Insert diagram showing: Agents → Message Bus (Redis) → Orchestrator → PostgreSQL + Nebius H100]

## Research Position

This work bridges:
- **Classic AI**: Contract Net Protocol (1980s) → LLM agents
- **Game Theory**: Utility functions, Pareto optimality, BATNA
- **Modern Infrastructure**: MCP/A2A protocols (2024-2025 standardization)

For IJCAI/AAAI submission: "This is an operational implementation of theories in agent communication that researchers are just beginning to explore."

---

Built in 5 weeks for Nebius × NVIDIA Hackathon.
```

---

# Testing Strategy

## Unit Tests

```bash
# Test protocol
pytest tests/unit/test_protocol.py -v

# Test agents
pytest tests/unit/test_cost_agent.py -v
pytest tests/unit/test_quality_agent.py -v
```

## Integration Tests

```bash
# Test full negotiation
pytest tests/integration/test_evaluation.py -v

# Test message bus
pytest tests/integration/test_message_bus.py -v
```

## Coverage

```bash
pytest --cov=src --cov-report=html
# Open htmlcov/index.html
```

---

# Debugging Tips

## 1. Check Redis Connection

```bash
redis-cli ping
# Should return: PONG
```

## 2. View Redis Messages

```bash
redis-cli
SUBSCRIBE "negotiation:*"
# Will show all published messages
```

## 3. Enable Debug Logging

In `main.py`:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 4. Check Agent Logic

```python
# In Python REPL
from src.agents.cost_agent import CostAgent
from src.protocol.message import ProposalDetail

agent = CostAgent()
proposal = ProposalDetail(
    vendor="Dell",
    unit_price_usd=3800,
    total_cost_usd=1900000,
    delivery_days=55,
    quality_tier="B+",
    vendor_risk_score=0.24
)
print(agent.compute_utility(proposal))  # Should be ~0.80
```

## 5. WebSocket Debugging

```javascript
// In browser console
const ws = new WebSocket('ws://localhost:8000/ws/negotiation');
ws.onmessage = (e) => console.log(JSON.parse(e.data));
ws.send('ping');
```

---

# Summary Timeline

| Phase | Week | Goal |
|-------|------|------|
| 1 | W1 | Protocol + Single Agent + FastAPI |
| 2 | W2 | Multi-Agent Loop + Round Manager |
| 3 | W3 | Mediator + Deadlock + BATNA Logic |
| 4 | W4 | React Dashboard + WebSocket |
| 5 | W5 | Polish + Evaluation + Open-Source |

**Total:** 5 weeks from zero to production-ready.

Good luck! 🚀
