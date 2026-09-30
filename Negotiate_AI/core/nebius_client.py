"""
Nebius LLM Inference Client
Wraps the Nebius Token Factory OpenAI-compatible API.
Uses structured output (JSON schema enforcement) for protocol compliance.
"""

from __future__ import annotations
import json
import logging
import os
from typing import Any, Optional, Type, TypeVar

try:
    from openai import AsyncOpenAI
except ModuleNotFoundError:  # pragma: no cover - dependency is installed in project env
    AsyncOpenAI = None

from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class NebiusClient:
    """
    Async wrapper around the Nebius Token Factory API.
    Nebius exposes an OpenAI-compatible endpoint, so we use the openai SDK
    pointed at Nebius's base URL.

    Usage:
        client = NebiusClient()
        result = await client.complete_structured(
            system_prompt="You are the Cost Agent...",
            user_prompt="Evaluate this task and propose a vendor.",
            output_schema=ProposalPayload,
        )
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        if AsyncOpenAI is None:
            raise ModuleNotFoundError(
                "The 'openai' package is required to use NebiusClient. "
                "Install the project dependencies first."
            )

        self.api_key  = api_key  or os.environ["NEBIUS_API_KEY"]
        self.base_url = base_url or os.environ.get(
            "NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1/"
        )
        self.model    = model    or os.environ.get(
            "NEBIUS_MODEL", "meta-llama/Meta-Llama-3.1-70B-Instruct"
        )

        self._client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        """
        Raw completion — returns the model's text response.
        Use complete_structured when you need a typed Pydantic object back.
        """
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content
        logger.debug(f"[Nebius] raw response: {content[:200]}...")
        return content

    async def complete_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        output_schema: Type[T],
        temperature: float = 0.2,   # lower temp = more reliable JSON
        max_tokens: int = 2048,
        max_retries: int = 3,
    ) -> T:
        """
        Structured completion — enforces the Pydantic schema via JSON mode.
        Returns a validated Pydantic model instance.

        We instruct the model to respond only with valid JSON matching the schema,
        then parse and validate with Pydantic. On parse failure we retry.
        """
        schema_json = json.dumps(output_schema.model_json_schema(), indent=2)

        system_with_schema = (
            f"{system_prompt}\n\n"
            f"CRITICAL: Respond ONLY with a valid JSON object matching this schema. "
            f"No preamble, no explanation, no markdown fences — just the JSON.\n\n"
            f"Schema:\n{schema_json}"
        )

        last_error: Optional[Exception] = None

        for attempt in range(1, max_retries + 1):
            try:
                raw = await self.complete(
                    system_prompt=system_with_schema,
                    user_prompt=user_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                # Strip any accidental markdown fences the model adds
                cleaned = raw.strip()
                if cleaned.startswith("```"):
                    lines = cleaned.split("\n")
                    cleaned = "\n".join(lines[1:-1]) if lines[-1] == "```" else "\n".join(lines[1:])

                parsed = output_schema.model_validate_json(cleaned)
                logger.info(f"[Nebius] structured output parsed successfully (attempt {attempt})")
                return parsed

            except Exception as e:
                last_error = e
                logger.warning(f"[Nebius] parse attempt {attempt}/{max_retries} failed: {e}")
                if attempt < max_retries:
                    # Give the model a hint about what went wrong
                    user_prompt = (
                        f"{user_prompt}\n\n"
                        f"[RETRY {attempt}] Your previous response failed validation with: {e}. "
                        f"Please respond with ONLY valid JSON."
                    )

        raise ValueError(
            f"Failed to get valid structured output after {max_retries} attempts. "
            f"Last error: {last_error}"
        )

    async def health_check(self) -> dict[str, Any]:
        """Quick sanity check to confirm Nebius API is reachable and responding."""
        try:
            response = await self.complete(
                system_prompt="You are a health check endpoint.",
                user_prompt='Respond with exactly: {"status": "ok"}',
                temperature=0.0,
                max_tokens=20,
            )
            return {"status": "ok", "model": self.model, "raw": response.strip()}
        except Exception as e:
            return {"status": "error", "error": str(e)}


# Module-level singleton — agents import this directly
_client: Optional[NebiusClient] = None


def get_nebius_client() -> NebiusClient:
    """Returns the shared NebiusClient instance (lazy-init)."""
    global _client
    if _client is None:
        _client = NebiusClient()
    return _client
