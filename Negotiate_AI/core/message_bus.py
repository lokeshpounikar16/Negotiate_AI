"""
Redis Pub/Sub Message Bus
Agents publish NegotiationMessages and subscribe to each other's messages.
Every message is also persisted to a task-specific log list in Redis.

Channel naming:
  negotiation:{task_id}          — broadcast channel (all agents subscribe)
  negotiation:{task_id}:{agent}  — unicast channel (directed messages)
  negotiation:{task_id}:log      — Redis list of all messages (append-only audit)
"""

from __future__ import annotations
import asyncio
import json
import logging
import os
from typing import AsyncIterator, Callable, Optional

try:
    import redis.asyncio as aioredis
except ModuleNotFoundError:  # pragma: no cover - dependency is installed in project env
    aioredis = None

from .schema import AgentID, NegotiationMessage, TaskAnnouncement

logger = logging.getLogger(__name__)


class MessageBus:
    """
    Async Redis pub/sub message bus.

    Usage:
        bus = MessageBus()
        await bus.connect()

        # Publisher (agent sends a message)
        await bus.publish(task_id="abc123", message=my_msg)

        # Subscriber (agent listens for incoming messages)
        async for msg in bus.subscribe(task_id="abc123"):
            handle(msg)

        await bus.disconnect()
    """

    def __init__(self, redis_url: Optional[str] = None):
        self.redis_url = redis_url or os.environ.get("REDIS_URL", "redis://localhost:6379")
        self._redis: Optional[aioredis.Redis] = None
        self._pubsub: Optional[aioredis.client.PubSub] = None

    async def connect(self) -> None:
        if aioredis is None:
            raise ModuleNotFoundError(
                "The 'redis' package is required to use MessageBus. "
                "Install the project dependencies first."
            )
        self._redis = await aioredis.from_url(
            self.redis_url,
            encoding="utf-8",
            decode_responses=True,
        )
        logger.info(f"[MessageBus] Connected to Redis at {self.redis_url}")

    async def disconnect(self) -> None:
        if self._pubsub:
            await self._pubsub.close()
        if self._redis:
            await self._redis.aclose()
        logger.info("[MessageBus] Disconnected from Redis")

    def _broadcast_channel(self, task_id: str) -> str:
        return f"negotiation:{task_id}"

    def _log_key(self, task_id: str) -> str:
        return f"negotiation:{task_id}:log"

    def _task_key(self, task_id: str) -> str:
        return f"negotiation:{task_id}:task"

    async def store_task(self, task: TaskAnnouncement) -> None:
        """Persist task details so agents can be run after task creation."""
        if not self._redis:
            raise RuntimeError("MessageBus not connected. Call connect() first.")
        await self._redis.set(self._task_key(task.task_id), task.model_dump_json())

    async def get_task(self, task_id: str) -> Optional[TaskAnnouncement]:
        """Retrieve a task previously stored in Redis."""
        if not self._redis:
            raise RuntimeError("MessageBus not connected. Call connect() first.")
        payload = await self._redis.get(self._task_key(task_id))
        return TaskAnnouncement.model_validate_json(payload) if payload else None

    async def publish(self, task_id: str, message: NegotiationMessage) -> None:
        """
        Publish a message to the negotiation broadcast channel.
        Also appends it to the persistent log list.
        """
        if not self._redis:
            raise RuntimeError("MessageBus not connected. Call connect() first.")

        payload = message.model_dump_json()
        broadcast_payload = json.dumps(message.to_agent_view())
        channel = self._broadcast_channel(task_id)

        # Broadcast to all subscribers
        await self._redis.publish(channel, broadcast_payload)

        # Persist to log (append-only audit trail)
        await self._redis.rpush(self._log_key(task_id), payload)

        logger.debug(
            f"[MessageBus] Published {message.msg_type.value} "
            f"from {message.sender.value} on {channel}"
        )

    async def subscribe(
        self,
        task_id: str,
        agent_id: Optional[AgentID] = None,
    ) -> AsyncIterator[NegotiationMessage]:
        """
        Subscribe to the broadcast channel for a task.
        Yields NegotiationMessage objects as they arrive.

        If agent_id is provided, the agent's own messages are filtered out
        so agents don't react to themselves.
        """
        if not self._redis:
            raise RuntimeError("MessageBus not connected. Call connect() first.")

        pubsub = self._redis.pubsub()
        channel = self._broadcast_channel(task_id)
        await pubsub.subscribe(channel)
        logger.info(f"[MessageBus] {agent_id or 'unknown'} subscribed to {channel}")

        try:
            async for raw_msg in pubsub.listen():
                if raw_msg["type"] != "message":
                    continue
                try:
                    msg = NegotiationMessage.model_validate_json(raw_msg["data"])
                    # Skip own messages
                    if agent_id and msg.sender == agent_id:
                        continue
                    yield msg
                except Exception as e:
                    logger.warning(f"[MessageBus] Failed to parse message: {e}")
        finally:
            await pubsub.unsubscribe(channel)
            await pubsub.aclose()

    async def get_log(self, task_id: str) -> list[NegotiationMessage]:
        """
        Retrieve the full audit log for a task session.
        Returns messages in chronological order.
        """
        if not self._redis:
            raise RuntimeError("MessageBus not connected. Call connect() first.")

        raw_list = await self._redis.lrange(self._log_key(task_id), 0, -1)
        messages = []
        for raw in raw_list:
            try:
                messages.append(NegotiationMessage.model_validate_json(raw))
            except Exception as e:
                logger.warning(f"[MessageBus] Could not parse log entry: {e}")
        return messages

    async def get_log_count(self, task_id: str) -> int:
        """Returns the number of messages logged for a task."""
        if not self._redis:
            raise RuntimeError("MessageBus not connected. Call connect() first.")
        return await self._redis.llen(self._log_key(task_id))

    async def clear_task(self, task_id: str) -> None:
        """Delete all Redis data for a task (useful in tests)."""
        if not self._redis:
            raise RuntimeError("MessageBus not connected. Call connect() first.")
        keys = [
            self._broadcast_channel(task_id),
            self._log_key(task_id),
            self._task_key(task_id),
        ]
        await self._redis.delete(*keys)
        logger.info(f"[MessageBus] Cleared all data for task {task_id}")

    async def health_check(self) -> dict:
        """Ping Redis to verify connection."""
        try:
            if not self._redis:
                await self.connect()
            pong = await self._redis.ping()
            return {"status": "ok", "redis": str(pong)}
        except Exception as e:
            return {"status": "error", "error": str(e)}


# ── Module-level singleton ────────────────────────────────────────────────────

_bus: Optional[MessageBus] = None


def get_message_bus() -> MessageBus:
    global _bus
    if _bus is None:
        _bus = MessageBus()
    return _bus
