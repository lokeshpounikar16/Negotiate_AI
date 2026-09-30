import { useEffect, useMemo, useState } from 'react';
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

const sampleMessages = [
  { round: 1, sender: 'cost_agent', msg_type: 'PROPOSAL', utility_score: 0.59 },
  { round: 1, sender: 'quality_agent', msg_type: 'PROPOSAL', utility_score: 0.82 },
  { round: 1, sender: 'timeline_agent', msg_type: 'PROPOSAL', utility_score: 0.74 },
  { round: 1, sender: 'risk_agent', msg_type: 'PROPOSAL', utility_score: 0.68 },
  { round: 2, sender: 'cost_agent', msg_type: 'COUNTER_PROPOSAL', utility_score: 0.66 },
  { round: 2, sender: 'quality_agent', msg_type: 'COUNTER_PROPOSAL', utility_score: 0.8 },
  { round: 2, sender: 'timeline_agent', msg_type: 'COUNTER_PROPOSAL', utility_score: 0.77 },
  { round: 2, sender: 'risk_agent', msg_type: 'COUNTER_PROPOSAL', utility_score: 0.73 },
  { round: 3, sender: 'mediator_agent', msg_type: 'MEDIATOR_PROPOSAL', utility_score: 0.71 },
];

const agentColors = {
  cost_agent: '#ef4444',
  quality_agent: '#3b82f6',
  timeline_agent: '#f59e0b',
  risk_agent: '#10b981',
  mediator_agent: '#8b5cf6',
};

const defaultForm = {
  title: 'Hardware procurement',
  description: 'Buy 100 servers for a lab workload.',
  max_budget_usd: 500000,
  quantity: 100,
  max_delivery_days: 60,
  min_quality_tier: 'B',
  max_vendor_risk: 0.4,
  min_warranty_months: 12,
  max_rounds: 6,
};

export default function App() {
  const [messages, setMessages] = useState(sampleMessages);
  const [status, setStatus] = useState('connecting');
  const [form, setForm] = useState(defaultForm);
  const [taskId, setTaskId] = useState('');
  const [isRunning, setIsRunning] = useState(false);

  useEffect(() => {
    let cancelled = false;

    async function loadStatus() {
      try {
        const res = await fetch('http://localhost:8000/health');
        if (!res.ok) throw new Error('health check failed');
        const data = await res.json();
        if (!cancelled) {
          setStatus(data.status || 'ok');
        }
      } catch (error) {
        if (!cancelled) {
          setStatus('demo-mode');
        }
      }
    }

    loadStatus();

    const socket = new WebSocket('ws://localhost:8000/ws/negotiation');

    socket.onopen = () => {
      if (!cancelled) {
        setStatus('live');
      }
    };

    socket.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        const inner = payload.message ?? payload;
        const message = typeof inner === 'string' ? JSON.parse(inner) : inner;
        if (!message || !message.sender) return;
        if (!cancelled) {
          setMessages((prev) => {
            const next = [...prev];
            const existingIndex = next.findIndex(
              (item) => item.msg_id === message.msg_id || (item.sender === message.sender && item.round === message.round && item.msg_type === message.msg_type)
            );
            if (existingIndex >= 0) {
              next[existingIndex] = {
                ...next[existingIndex],
                ...message,
                utility_score: message.utility_score ?? next[existingIndex].utility_score,
              };
              return next;
            }
            return [...next, {
              round: message.round ?? 0,
              sender: message.sender,
              msg_type: message.msg_type,
              utility_score: message.utility_score ?? 0,
              public_reason: message.public_reason,
              msg_id: message.msg_id,
            }];
          });
        }
      } catch (error) {
        console.error('Failed to parse websocket payload', error);
      }
    };

    socket.onclose = () => {
      if (!cancelled) {
        setStatus('demo-mode');
      }
    };

    return () => {
      cancelled = true;
      socket.close();
    };
  }, []);

  const chartData = useMemo(() => {
    const rounds = {};

    messages.forEach((msg) => {
      if (!rounds[msg.round]) {
        rounds[msg.round] = { round: msg.round };
      }
      rounds[msg.round][msg.sender] = msg.utility_score;
    });

    return Object.values(rounds).sort((a, b) => a.round - b.round);
  }, [messages]);

  const handleChange = (event) => {
    const { name, value } = event.target;
    setForm((prev) => ({
      ...prev,
      [name]: name === 'max_budget_usd' || name === 'quantity' || name === 'max_delivery_days' || name === 'min_warranty_months' || name === 'max_rounds'
        ? Number(value)
        : value,
    }));
  };

  const handleRunNegotiation = async (event) => {
    event.preventDefault();
    setIsRunning(true);
    try {
      const res = await fetch('http://localhost:8000/negotiation/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(form),
      });

      if (!res.ok) {
        throw new Error(`Negotiation request failed: ${res.status}`);
      }

      const data = await res.json();
      setTaskId(data.task_id || '');
      if (Array.isArray(data.messages)) {
        setMessages((prev) => {
          const mapped = data.messages.map((msg) => ({
            round: msg.round ?? 0,
            sender: msg.sender,
            msg_type: msg.msg_type,
            utility_score: msg.utility_score ?? 0,
            public_reason: msg.public_reason,
            msg_id: msg.msg_id,
          }));
          return [...prev, ...mapped];
        });
      }
    } catch (error) {
      console.error(error);
      setStatus('demo-mode');
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">NegotiateAI</p>
          <h1>Week 4 dashboard</h1>
        </div>
        <span className={`status-pill ${status}`}>{status}</span>
      </header>

      <section className="hero-grid">
        <div className="panel">
          <h2>Live negotiation overview</h2>
          <p>
            This dashboard runs a real negotiation round against the FastAPI backend and streams the
            subsequent utility updates over the WebSocket channel.
          </p>
          <div className="stats-row">
            <div>
              <span>Total rounds</span>
              <strong>{Math.max(...messages.map((m) => m.round), 0)}</strong>
            </div>
            <div>
              <span>Agents</span>
              <strong>4</strong>
            </div>
            <div>
              <span>Task</span>
              <strong>{taskId ? taskId.slice(0, 8) : 'idle'}</strong>
            </div>
          </div>

          <form className="task-form" onSubmit={handleRunNegotiation}>
            <label>
              Title
              <input name="title" value={form.title} onChange={handleChange} />
            </label>
            <label>
              Description
              <textarea name="description" value={form.description} onChange={handleChange} rows={3} />
            </label>
            <div className="inline-grid">
              <label>
                Budget
                <input name="max_budget_usd" type="number" value={form.max_budget_usd} onChange={handleChange} />
              </label>
              <label>
                Quantity
                <input name="quantity" type="number" value={form.quantity} onChange={handleChange} />
              </label>
              <label>
                Max days
                <input name="max_delivery_days" type="number" value={form.max_delivery_days} onChange={handleChange} />
              </label>
              <label>
                Quality
                <select name="min_quality_tier" value={form.min_quality_tier} onChange={handleChange}>
                  <option value="B+">B+</option>
                  <option value="B">B</option>
                  <option value="A-">A-</option>
                  <option value="A">A</option>
                </select>
              </label>
            </div>
            <button type="submit" disabled={isRunning}>{isRunning ? 'Running...' : 'Run negotiation'}</button>
          </form>
        </div>

        <div className="panel chart-panel">
          <h2>Utility trend</h2>
          <div className="chart-wrap">
            <ResponsiveContainer width="100%" height={260}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2a2d35" />
                <XAxis dataKey="round" stroke="#d7dfeb" />
                <YAxis domain={[0, 1]} stroke="#d7dfeb" />
                <Tooltip />
                <Line type="monotone" dataKey="cost_agent" stroke={agentColors.cost_agent} strokeWidth={3} />
                <Line type="monotone" dataKey="quality_agent" stroke={agentColors.quality_agent} strokeWidth={3} />
                <Line type="monotone" dataKey="timeline_agent" stroke={agentColors.timeline_agent} strokeWidth={3} />
                <Line type="monotone" dataKey="risk_agent" stroke={agentColors.risk_agent} strokeWidth={3} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </section>

      <section className="panel">
        <h2>Negotiation feed</h2>
        <div className="feed-list">
          {messages.map((msg, index) => (
            <div key={`${msg.sender}-${msg.round}-${msg.msg_id ?? index}`} className="feed-item">
              <div className="feed-header">
                <span className="agent-badge" style={{ background: agentColors[msg.sender] || '#64748b' }}>
                  {msg.sender}
                </span>
                <span className="msg-type">{msg.msg_type}</span>
                <span className="round-pill">Round {msg.round}</span>
              </div>
              <div className="feed-body">
                <strong>Utility:</strong> {msg.utility_score?.toFixed(2) ?? '0.00'}
              </div>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
