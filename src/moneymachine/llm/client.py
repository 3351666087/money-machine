"""LLM client for the relay (中转站).

Design rules (the LLM is an experimental arm, never a dependency):
- every failure — timeout, HTTP error, refusal, bad JSON, schema mismatch,
  budget exceeded, *relay returned a different model* — yields ``ok=False``
  and the calling strategy must then do nothing;
- every call is written to the ledger (prompt, response, tokens, latency, and
  cost at the official list price);
- Claude models use the official Anthropic SDK (Messages API + structured
  outputs); other models use an OpenAI-compatible /chat/completions.
"""
from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from moneymachine.config import LLMConfig, LLMModel
from moneymachine.timeutil import day_start


@dataclass
class LLMResult:
    ok: bool
    data: dict | None = None
    error: str | None = None
    model_requested: str = ""
    model_returned: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    cost_usd: float = 0.0
    text: str = ""
    extra: dict = field(default_factory=dict)


def extract_json(text: str) -> dict:
    """Parse a JSON object from model text (tolerates ``` fences / prose)."""
    t = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", t, re.S)
    if fence:
        t = fence.group(1)
    try:
        obj = json.loads(t)
    except json.JSONDecodeError:
        start, end = t.find("{"), t.rfind("}")
        if start < 0 or end <= start:
            raise
        obj = json.loads(t[start : end + 1])
    if not isinstance(obj, dict):
        raise ValueError("top-level JSON is not an object")
    return obj


_PY_TYPES = {"string": str, "number": (int, float), "integer": int, "boolean": bool, "object": dict, "array": list}


def validate(data: dict, schema: dict) -> str | None:
    """Tiny JSON-schema subset check: required, types, enum."""
    props = schema.get("properties", {})
    for key in schema.get("required", []):
        if key not in data:
            return f"missing field {key!r}"
    for key, spec in props.items():
        if key not in data:
            continue
        v = data[key]
        types = spec.get("type")
        types = types if isinstance(types, list) else [types] if types else []
        if types:
            ok = False
            for ty in types:
                if ty == "null" and v is None:
                    ok = True
                elif ty in _PY_TYPES and isinstance(v, _PY_TYPES[ty]) and not (ty in ("number", "integer") and isinstance(v, bool)):
                    ok = True
            if not ok:
                return f"field {key!r} has wrong type ({type(v).__name__}, want {types})"
        if "enum" in spec and v not in spec["enum"]:
            return f"field {key!r}={v!r} not in {spec['enum']}"
    return None


def same_model(requested: str, returned: str) -> bool:
    """Relays often echo dated / provider-prefixed ids; accept those, reject swaps."""
    if not returned:
        return False
    r = returned.lower().split("/")[-1]
    q = requested.lower().split("/")[-1]
    return r == q or r.startswith(q + "-") or r.startswith(q + "@") or r.startswith(q + ":")


class LLMClient:
    def __init__(self, cfg: LLMConfig, ledger=None, run_id: str = "", *,
                 http: httpx.Client | None = None, anthropic_client: Any = None):
        self.cfg = cfg
        self.ledger = ledger
        self._http = http
        self._anthropic = anthropic_client
        self._lock = threading.Lock()
        self._usage: dict[tuple[str, int], list[int]] = {}  # (strategy, day) -> [calls, tokens]

    # ── transports (lazy) ────────────────────────────────────────────────
    def _http_client(self) -> httpx.Client:
        if self._http is None:
            self._http = httpx.Client(timeout=self.cfg.timeout_s)
        return self._http

    def _anthropic_client(self):
        if self._anthropic is None:
            import anthropic

            self._anthropic = anthropic.Anthropic(
                base_url=self.cfg.anthropic_base_url or None,
                api_key=self.cfg.api_key or None,
                timeout=self.cfg.timeout_s,
                max_retries=1,
            )
        return self._anthropic

    def configured(self) -> bool:
        return bool(self.cfg.base_url and self.cfg.api_key)

    # ── budget ───────────────────────────────────────────────────────────
    def _budget_ok(self, strategy: str, now: int) -> str | None:
        with self._lock:
            calls, tokens = self._usage.get((strategy, day_start(now)), [0, 0])
        if calls >= self.cfg.max_calls_per_day:
            return f"daily call budget exhausted ({calls}/{self.cfg.max_calls_per_day})"
        if tokens >= self.cfg.max_tokens_per_day:
            return f"daily token budget exhausted ({tokens}/{self.cfg.max_tokens_per_day})"
        return None

    def preload_usage(self, db, run_id: str, now: int) -> None:
        """Restore today's per-strategy usage after a restart so budgets still hold."""
        day = day_start(now)
        rows = db.execute(
            "SELECT strategy, COUNT(*), COALESCE(SUM(prompt_tokens + completion_tokens), 0) FROM llm_calls "
            "WHERE run_id=? AND ts>=? AND (error IS NULL OR error NOT LIKE 'daily % budget exhausted%') "
            "GROUP BY strategy", (run_id, day)).fetchall()
        with self._lock:
            for strategy, calls, tokens in rows:
                self._usage[(strategy, day)] = [int(calls), int(tokens)]

    def _account(self, strategy: str, now: int, tokens: int) -> None:
        with self._lock:
            u = self._usage.setdefault((strategy, day_start(now)), [0, 0])
            u[0] += 1
            u[1] += tokens

    # ── main entry ───────────────────────────────────────────────────────
    def call_json(self, *, strategy: str, purpose: str, symbol: str, model: str,
                  system: str, user: str, schema: dict, now: int) -> LLMResult:
        res = LLMResult(ok=False, model_requested=model)
        try:
            m = self.cfg.model(model)
        except KeyError as e:
            res.error = str(e)
            return self._finish(res, strategy, purpose, symbol, now, system, user)
        if not self.configured():
            res.error = f"relay not configured (set {self.cfg.base_url_env} / {self.cfg.api_key_env} in .env)"
            return self._finish(res, strategy, purpose, symbol, now, system, user)
        why = self._budget_ok(strategy, now)
        if why:
            res.error = why
            return self._finish(res, strategy, purpose, symbol, now, system, user, count=False)

        t0 = time.monotonic()
        try:
            if m.protocol == "anthropic":
                self._call_anthropic(m, system, user, schema, res)
            else:
                self._call_openai(m, system, user, res)
        except Exception as e:  # noqa: BLE001 — any transport failure = no trade
            res.error = f"{type(e).__name__}: {str(e)[:300]}"
        res.latency_ms = int((time.monotonic() - t0) * 1000)
        res.cost_usd = (res.prompt_tokens * m.price_in + res.completion_tokens * m.price_out) / 1e6

        if res.error is None:
            if not same_model(model, res.model_returned):
                res.error = f"model mismatch: requested {model!r}, relay returned {res.model_returned!r}"
                if not self.cfg.strict_model:
                    res.extra["model_mismatch"] = res.error
                    res.error = None
        if res.error is None:
            try:
                res.data = extract_json(res.text)
                err = validate(res.data, schema)
                if err:
                    res.error = f"schema: {err}"
            except Exception as e:  # noqa: BLE001
                res.error = f"bad JSON: {e}"
        res.ok = res.error is None
        return self._finish(res, strategy, purpose, symbol, now, system, user)

    def _call_openai(self, m: LLMModel, system: str, user: str, res: LLMResult) -> None:
        body: dict[str, Any] = {
            "model": m.name,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": self.cfg.temperature,
            "max_tokens": m.max_output_tokens or 8000,
        }
        if self.cfg.json_mode:
            body["response_format"] = {"type": "json_object"}
        r = self._http_client().post(
            f"{self.cfg.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.cfg.api_key}"},
            json=body,
        )
        if r.status_code >= 400:
            res.error = f"HTTP {r.status_code}: {r.text[:300]}"
            return
        j = r.json()
        res.model_returned = str(j.get("model") or "")
        usage = j.get("usage") or {}
        res.prompt_tokens = int(usage.get("prompt_tokens") or 0)
        res.completion_tokens = int(usage.get("completion_tokens") or 0)
        choice = (j.get("choices") or [{}])[0]
        res.text = str((choice.get("message") or {}).get("content") or "")
        if choice.get("finish_reason") == "length":
            res.error = "output truncated (finish_reason=length)"

    def _call_anthropic(self, m: LLMModel, system: str, user: str, schema: dict, res: LLMResult) -> None:
        output_config: dict[str, Any] = {"format": {"type": "json_schema", "schema": schema}}
        if m.effort:
            output_config["effort"] = m.effort
        resp = self._anthropic_client().messages.create(
            model=m.name,
            max_tokens=m.max_output_tokens or 16000,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config=output_config,
        )
        res.model_returned = str(getattr(resp, "model", "") or "")
        u = resp.usage
        res.prompt_tokens = int((u.input_tokens or 0) + (getattr(u, "cache_read_input_tokens", 0) or 0)
                                + (getattr(u, "cache_creation_input_tokens", 0) or 0))
        res.completion_tokens = int(u.output_tokens or 0)
        if resp.stop_reason == "refusal":
            details = getattr(resp, "stop_details", None)
            res.error = f"refusal ({getattr(details, 'category', None)})"
            return
        if resp.stop_reason == "max_tokens":
            res.error = "output truncated (max_tokens)"
            return
        res.text = next((b.text for b in resp.content if b.type == "text"), "")

    def _finish(self, res: LLMResult, strategy: str, purpose: str, symbol: str, now: int,
                system: str, user: str, count: bool = True) -> LLMResult:
        if count:
            self._account(strategy, now, res.prompt_tokens + res.completion_tokens)
        if self.ledger is not None:
            self.ledger.llm_call({
                "strategy": strategy, "ts": now, "purpose": purpose, "symbol": symbol,
                "model_requested": res.model_requested, "model_returned": res.model_returned,
                "ok": res.ok, "error": res.error, "prompt_tokens": res.prompt_tokens,
                "completion_tokens": res.completion_tokens, "latency_ms": res.latency_ms,
                "cost_usd": res.cost_usd, "request": json.dumps({"system": system, "user": user}, ensure_ascii=False),
                "response": res.text, "parsed": res.data,
            })
        return res
