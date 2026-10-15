"""TOML config → typed dataclasses. See configs/*.toml for documented examples."""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CostConfig:
    maker_fee: float = 0.0002
    taker_fee: float = 0.0005
    slippage_bps: float = 2.0
    # Used only when no real funding data covers a period (old backtests).
    default_funding_rate: float = 0.0001
    default_funding_interval_h: int = 8


@dataclass
class AccountConfig:
    initial_usdt: float = 10_000.0
    max_leverage: float = 3.0
    risk_per_trade: float = 0.01
    daily_loss_limit: float = 0.05
    max_open_positions: int = 3
    maintenance_margin_rate: float = 0.005


@dataclass
class StrategyConfig:
    id: str
    kind: str
    symbols: list[str]
    params: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True


@dataclass
class LLMModel:
    """One model reachable through the relay.

    protocol:
      - "anthropic": Claude models via the official Anthropic SDK (Messages API),
        relay must expose /v1/messages;
      - "openai": everything else via an OpenAI-compatible /chat/completions.
    price_in / price_out: USD per 1M tokens at the vendor's *official* list
    price, so LLM cost is judged as if paying full price.
    """

    name: str
    protocol: str = "openai"
    price_in: float = 0.0
    price_out: float = 0.0
    effort: str | None = None          # anthropic only: low/medium/high/xhigh/max
    max_output_tokens: int | None = None


@dataclass
class LLMConfig:
    base_url_env: str = "MM_LLM_BASE_URL"
    api_key_env: str = "MM_LLM_API_KEY"
    anthropic_base_url_env: str = "MM_LLM_ANTHROPIC_BASE_URL"
    timeout_s: float = 180.0
    temperature: float = 0.2           # openai protocol only
    json_mode: bool = True             # openai protocol: send response_format=json_object
    # Relay answered with a different model than requested → treat call as failed.
    strict_model: bool = True
    max_calls_per_day: int = 200
    max_tokens_per_day: int = 3_000_000
    models: list[LLMModel] = field(default_factory=list)

    @property
    def base_url(self) -> str:
        return os.environ.get(self.base_url_env, "").rstrip("/")

    @property
    def anthropic_base_url(self) -> str:
        """Anthropic SDK appends /v1/messages itself, so strip a trailing /v1."""
        url = os.environ.get(self.anthropic_base_url_env, "").rstrip("/") or self.base_url
        return url[:-3] if url.endswith("/v1") else url

    @property
    def api_key(self) -> str:
        return os.environ.get(self.api_key_env, "")

    def model(self, name: str) -> LLMModel:
        for m in self.models:
            if m.name == name:
                return m
        raise KeyError(f"model {name!r} is not listed under [[llm.models]]")


@dataclass
class GoNoGo:
    min_forward_days: int = 56
    min_trades: int = 100          # portfolio strategies: executed fills (they rarely go flat)
    max_drawdown: float = 0.15
    # 1.0 directional strategies must beat the benchmark (buy & hold BTC). A market-
    # neutral strategy is instead judged against its placebo and must be net positive;
    # the benchmark is then shown for reference only.
    beat_benchmark: bool = True
    require_positive_net: bool = False


@dataclass
class Config:
    run_name: str
    db_path: Path
    symbols: list[str]
    strategy_tf: str = "1H"
    matching_tf: str = "1m"
    backtest_matching_tf: str = "5m"
    store_dir: Path = Path("store/okx")
    archive_dir: Path = Path("data/OKX_K线数据")
    start: str | None = None
    end: str | None = None
    warmup_bars: int = 300
    seed: int = 42
    benchmark: str = "S0_hold_btc"
    costs: CostConfig = field(default_factory=CostConfig)
    account: AccountConfig = field(default_factory=AccountConfig)
    strategies: list[StrategyConfig] = field(default_factory=list)
    llm: LLMConfig = field(default_factory=LLMConfig)
    gonogo: GoNoGo = field(default_factory=GoNoGo)
    source_path: Path | None = None

    @property
    def enabled_strategies(self) -> list[StrategyConfig]:
        return [s for s in self.strategies if s.enabled]


def _pick(cls, data: dict[str, Any]):
    known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
    unknown = set(data) - set(known)
    if unknown:
        raise ValueError(f"unknown keys for [{cls.__name__}]: {sorted(unknown)}")
    return cls(**known)


def load_env_file(path: Path = Path(".env")) -> None:
    """Minimal .env loader (KEY=VALUE lines); never overrides real env vars."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _load_llm(data: dict[str, Any]) -> LLMConfig:
    data = dict(data)
    models = [_pick(LLMModel, m) for m in data.pop("models", [])]
    cfg = _pick(LLMConfig, data)
    cfg.models = models
    return cfg


def load_config(path: str | Path) -> Config:
    path = Path(path)
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    run = raw.get("run", {})
    market = raw.get("market", {})
    symbols = list(market.get("symbols", []))
    if market.get("symbols_file"):  # e.g. the 2.0 universe candidates, one instId per line
        sf = Path(market["symbols_file"])
        symbols += [ln.strip() for ln in sf.read_text().splitlines() if ln.strip() and ln.strip() not in symbols]
    if not symbols:
        raise ValueError("[market].symbols is empty")

    strategies = []
    seen = set()
    for s in raw.get("strategy", []):
        sid = s["id"]
        if sid in seen:
            raise ValueError(f"duplicate strategy id {sid!r}")
        seen.add(sid)
        strategies.append(
            StrategyConfig(
                id=sid,
                kind=s["kind"],
                symbols=list(s.get("symbols", symbols)),
                params=dict(s.get("params", {})),
                enabled=bool(s.get("enabled", True)),
            )
        )

    return Config(
        run_name=run.get("name", path.stem),
        db_path=Path(run.get("db", f"runs/{path.stem}.sqlite")),
        symbols=symbols,
        strategy_tf=market.get("strategy_timeframe", "1H"),
        matching_tf=market.get("matching_timeframe", "1m"),
        backtest_matching_tf=market.get("backtest_matching_timeframe", "5m"),
        store_dir=Path(market.get("store_dir", "store/okx")),
        archive_dir=Path(market.get("archive_dir", "data/OKX_K线数据")),
        start=run.get("start"),
        end=run.get("end"),
        warmup_bars=int(run.get("warmup_bars", 300)),
        seed=int(run.get("seed", 42)),
        benchmark=run.get("benchmark", "S0_hold_btc"),
        costs=_pick(CostConfig, raw.get("costs", {})),
        account=_pick(AccountConfig, raw.get("account", {})),
        strategies=strategies,
        llm=_load_llm(raw.get("llm", {})),
        gonogo=_pick(GoNoGo, raw.get("gonogo", {})),
        source_path=path,
    )
