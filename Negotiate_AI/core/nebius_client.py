"""Async client for Google's Gemini generateContent API."""

from __future__ import annotations
import json
import logging
import os
from typing import Any, Optional, Type, TypeVar
from urllib.parse import quote

import httpx
from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class GeminiClient:
    """
    Async wrapper around Google's native Gemini generateContent API.

    Usage:
        client = GeminiClient()
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
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.base_url = (base_url or os.environ.get(
            "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta"
        )).rstrip("/")
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-flash-latest")

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        response_mime_type: Optional[str] = None,
    ) -> str:
        """
        Raw completion — returns the model's text response.
        Use complete_structured when you need a typed Pydantic object back.
        """
        if not self.api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not configured. Set it in the environment or .env "
                "before invoking a model call."
            )

        generation_config: dict[str, Any] = {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        }
        if response_mime_type:
            generation_config["responseMimeType"] = response_mime_type

        endpoint = (
            f"{self.base_url}/models/{quote(self.model, safe='')}:generateContent"
        )
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                endpoint,
                headers={"X-goog-api-key": self.api_key},
                json={
                    "systemInstruction": {"parts": [{"text": system_prompt}]},
                    "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
                    "generationConfig": generation_config,
                },
            )
        response.raise_for_status()
        payload = response.json()
        candidates = payload.get("candidates", [])
        if not candidates:
            raise ValueError("Gemini returned no response candidates.")
        parts = candidates[0].get("content", {}).get("parts", [])
        content = "".join(part["text"] for part in parts if "text" in part)
        if not content:
            raise ValueError("Gemini returned no text content.")
        logger.debug(f"[Gemini] raw response: {content[:200]}...")
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
                    response_mime_type="application/json",
                )

                # Strip any accidental markdown fences the model adds
                cleaned = raw.strip()
                if cleaned.startswith("```"):
                    lines = cleaned.split("\n")
                    cleaned = "\n".join(lines[1:-1]) if lines[-1] == "```" else "\n".join(lines[1:])

                parsed = output_schema.model_validate_json(cleaned)
                logger.info(f"[Gemini] structured output parsed successfully (attempt {attempt})")
                return parsed

            except Exception as e:
                last_error = e
                logger.warning(f"[Gemini] parse attempt {attempt}/{max_retries} failed: {e}")
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
        """Quick sanity check to confirm the Gemini API is reachable and responding."""
        if not self.api_key:
            return {
                "status": "degraded",
                "error": "GEMINI_API_KEY is not configured; model calls are disabled in local mode.",
                "model": self.model,
            }

        try:
            response = await self.complete(
                system_prompt="You are a health check endpoint.",
                user_prompt='Respond with exactly: {"status": "ok"}',
                temperature=0.0,
                max_tokens=20,
            )
            return {"status": "ok", "model": self.model, "raw": response.strip()}
        except Exception as e:
            return {"status": "degraded", "error": str(e), "model": self.model}


# Keep the previous names available for existing agents and integrations.
NebiusClient = GeminiClient


_client: Optional[GeminiClient] = None


def get_gemini_client() -> GeminiClient:
    """Returns the shared GeminiClient instance (lazy-init)."""
    global _client
    if _client is None:
        _client = GeminiClient()
    return _client


def get_nebius_client() -> GeminiClient:
    """Backward-compatible alias for get_gemini_client()."""
    return get_gemini_client()
