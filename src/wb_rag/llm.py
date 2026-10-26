"""Minimal Gemini REST client shared by generation, summaries, query rewriting and the judge."""

import json
import logging
import os
import time
from dataclasses import dataclass

import httpx

from wb_rag.config import get_settings

log = logging.getLogger(__name__)

API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
RETRYABLE = {429, 500, 502, 503, 504}


class LLMError(RuntimeError):
    pass


@dataclass
class LLMResult:
    text: str
    model: str
    finish: str
    usage: dict

    def json(self) -> dict | list:
        t = self.text.strip()
        if t.startswith("```"):
            t = t.strip("`").removeprefix("json").strip()
        return json.loads(t)


def api_key() -> str | None:
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def _post(url: str, body: dict, key: str, retries: int) -> httpx.Response | None:
    s = get_settings()
    resp = None
    for attempt in range(retries + 1):
        try:
            resp = httpx.post(url, json=body, headers={"x-goog-api-key": key}, timeout=s.gemini_timeout)
        except httpx.HTTPError:
            resp = None
        if resp is not None and resp.status_code not in RETRYABLE:
            return resp
        if attempt < retries:
            wait = min(60, 4 * 2 ** attempt)          # 4, 8, 16, 32 s: rate limits are per minute
            if resp is not None and resp.status_code == 429:
                wait = max(wait, float(resp.headers.get("retry-after", 0) or 0))
            time.sleep(wait)
    return resp


def complete(system: str, user: str, model: str | None = None, *, json_schema: dict | None = None,
             temperature: float | None = None, max_tokens: int | None = None,
             fallbacks: list[str] | None = None, retries: int = 3) -> LLMResult:
    s = get_settings()
    key = api_key()
    if not key:
        raise LLMError("No Gemini credentials found — set GEMINI_API_KEY in .env.")
    config: dict = {"maxOutputTokens": max_tokens or s.max_tokens,
                    "temperature": s.temperature if temperature is None else temperature}
    if json_schema is not None:
        config["responseMimeType"] = "application/json"
        config["responseSchema"] = json_schema
    body = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": config,
    }
    models = [model or s.gemini_model, *(s.gemini_fallback_models if fallbacks is None else fallbacks)]
    resp = None
    for m in models:
        resp = _post(API_URL.format(model=m), body, key, retries=retries)
        if resp is not None and resp.status_code not in RETRYABLE:
            break
        log.warning("Gemini model %s unavailable (%s); trying next", m,
                    resp.status_code if resp is not None else "timeout")
    if resp is None:
        raise LLMError("Could not reach the Gemini API — check the network connection.")
    if resp.status_code in (401, 403):
        raise LLMError("Gemini authentication failed — check GEMINI_API_KEY.")
    if resp.status_code != 200:
        msg = resp.json().get("error", {}).get("message", resp.text[:300]) if resp.content else ""
        raise LLMError(f"Gemini API error {resp.status_code}: {msg}")

    data = resp.json()
    usage = data.get("usageMetadata", {})
    usage_out = {"input_tokens": usage.get("promptTokenCount", 0),
                 "output_tokens": usage.get("candidatesTokenCount", 0) + usage.get("thoughtsTokenCount", 0)}
    used = data.get("modelVersion", models[0])
    candidates = data.get("candidates") or []
    if not candidates:
        return LLMResult("", used, "blocked", usage_out)
    cand = candidates[0]
    text = "".join(p.get("text", "") for p in cand.get("content", {}).get("parts", []) if not p.get("thought"))
    return LLMResult(text, used, cand.get("finishReason", "STOP"), usage_out)
