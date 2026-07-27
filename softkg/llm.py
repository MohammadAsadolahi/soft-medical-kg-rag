"""LLM access for the extraction and query-rewriting stages.

Two call shapes are needed and they have different failure modes, so they get different methods:

  ``chat``       free-text in, free-text out. Used by the decontextualization stages, where the
                 output is a rewritten passage.
  ``tool_call``  forced structured output. Used by relation extraction, where we need a JSON array
                 of typed triplets and cannot afford to parse it out of prose.

``tool_call`` uses the forced-tool-call pattern: a function schema is declared, ``tool_choice`` pins
the model to it, and the *arguments* the model is compelled to produce are the payload. The function
itself is never implemented or invoked. The point is that the provider constrains decoding to the
declared JSON grammar, so malformed output is impossible by construction rather than something to
retry around. Extracting 27k triplets by asking for JSON in the prompt and parsing the reply is a
different and much worse experience.

Any OpenAI-compatible endpoint works (``base_url``), including Azure OpenAI, a local vLLM or Ollama
server, or a self-hosted fine-tuned Qwen3 from ``finetuning/``.
"""
from __future__ import annotations

import json
import logging
import random
import time
from typing import Any

logger = logging.getLogger(__name__)


class RetryExhausted(RuntimeError):
    """Raised when every attempt for a single call failed."""


class LLMClient:
    """OpenAI-compatible client with backoff tuned for long unattended extraction runs.

    A full corpus extraction is tens of thousands of calls over many hours, so transient failures
    are certain rather than unlikely. The retry policy honours a server-supplied ``Retry-After``
    when present, and otherwise backs off exponentially with jitter -- the jitter matters because
    parallel workers that back off in lockstep re-collide on every wave.
    """

    BASE_BACKOFF = 2.0     # seconds; doubles per attempt
    MAX_BACKOFF = 120.0    # never sleep longer than this on one attempt
    JITTER = 0.5           # +/-50%, decorrelates concurrent workers

    def __init__(
        self,
        api_key: str,
        endpoint: str,
        *,
        default_model: str = "gpt-5-mini",
        use_azure: bool = False,
        api_version: str = "2024-12-01-preview",
        reasoning_effort: str | None = "high",
        max_completion_tokens: int = 16000,
    ) -> None:
        if use_azure:
            from openai import AzureOpenAI
            self.client: Any = AzureOpenAI(
                api_key=api_key, api_version=api_version, azure_endpoint=endpoint)
        else:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key, base_url=endpoint)

        self.default_model = default_model
        self.reasoning_effort = reasoning_effort
        self.max_completion_tokens = max_completion_tokens

    # -- internals ---------------------------------------------------------
    def _extra(self) -> dict:
        """Params only reasoning models accept, omitted otherwise.

        Sending ``reasoning_effort`` to a model that does not support it is a hard 400, which would
        make the client unusable against a local fine-tune. Pass ``reasoning_effort=None`` for those.
        """
        return {"reasoning_effort": self.reasoning_effort} if self.reasoning_effort else {}

    def _sleep(self, attempt: int, error: Exception | None = None) -> None:
        delay: float | None = None
        response = getattr(error, "response", None)
        if response is not None:
            headers = getattr(response, "headers", None) or {}
            raw = headers.get("Retry-After") or headers.get("retry-after")
            if raw is not None:
                try:
                    delay = float(raw)
                except (TypeError, ValueError):
                    delay = None
        if delay is None:
            delay = self.BASE_BACKOFF * (2 ** attempt)

        delay = min(delay, self.MAX_BACKOFF)
        delay = max(1.0, delay + delay * self.JITTER * (2 * random.random() - 1))
        logger.warning("transient LLM error, backing off %.1fs (attempt %d): %s",
                       delay, attempt + 1, error)
        time.sleep(delay)

    # -- public API --------------------------------------------------------
    def chat(self, system: str, user: str, *, model: str | None = None,
             max_retries: int = 8) -> str:
        """Plain completion. Returns the assistant message text."""
        from openai import BadRequestError

        last: Exception | None = None
        for attempt in range(max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model=model or self.default_model,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}],
                    **self._extra(),
                )
                return completion.choices[0].message.content or ""
            except BadRequestError:
                # Content filter or malformed request. Retrying sends the identical payload and
                # fails identically, so surface it and let the caller record the document as failed.
                raise
            except Exception as exc:
                last = exc
                if attempt == max_retries - 1:
                    break
                self._sleep(attempt, exc)
        raise RetryExhausted(f"chat failed after {max_retries} attempts") from last

    def tool_call(self, system: str, user: str, tool: dict, tool_name: str, *,
                  model: str | None = None, max_retries: int = 8) -> dict:
        """Forced structured output. Returns the parsed tool arguments.

        The tool is never executed -- ``tool_choice`` is used purely to obtain grammar-constrained
        JSON. An empty arguments object is treated as a retryable failure, not as "no triplets":
        genuinely triplet-free text still returns ``{"triplets": []}``, whereas an empty object
        means the call degenerated.
        """
        from openai import BadRequestError

        last: Exception | None = None
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=model or self.default_model,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}],
                    tools=[tool],
                    tool_choice={"type": "function", "function": {"name": tool_name}},
                    max_completion_tokens=self.max_completion_tokens,
                    **self._extra(),
                )
                calls = response.choices[0].message.tool_calls
                if calls:
                    parsed = json.loads(calls[0].function.arguments)
                    if parsed:
                        return parsed
                logger.warning("forced tool call produced no arguments (attempt %d)", attempt + 1)
                time.sleep(3)
            except BadRequestError:
                raise
            except Exception as exc:
                last = exc
                if attempt == max_retries - 1:
                    break
                self._sleep(attempt, exc)
        raise RetryExhausted(f"tool call failed after {max_retries} attempts") from last
