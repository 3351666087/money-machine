"""mm — command line for 刷钱机 1.0."""
from __future__ import annotations

import argparse
import asyncio
import logging
import sqlite3
import sys
from pathlib import Path

from moneymachine.config import load_config, load_env_file
from moneymachine.timeutil import fmt_ms, now_ms, tf_ms, to_ms


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-5s %(message)s",
        datefmt="%m-%d %H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)


def _latest_run(db: Path, mode: str | None = None) -> str:
    if not db.exists():
        raise SystemExit(f"{db} does not exist yet")
    q = "SELECT run_id FROM runs" + (" WHERE mode=?" if mode else "") + " ORDER BY created_at DESC LIMIT 1"
    with sqlite3.connect(str(db)) as c:
        row = c.execute(q, (mode,) if mode else ()).fetchone()
    if not row:
        raise SystemExit(f"no {mode or ''} runs in {db}")
    return row[0]


def cmd_data_sync(args) -> None:
    from moneymachine.market.okx_rest import OkxRest
    from moneymachine.market.store import Store

    cfg = load_config(args.config)
    rest = OkxRest()
    store = Store(cfg.store_dir, cfg.archive_dir)
    all_inst = rest.instruments("SWAP")
    symbols = sorted(set(cfg.symbols) | {s for st in cfg.strategies for s in st.symbols})
    missing = [s for s in symbols if s not in all_inst]
    if missing:
        raise SystemExit(f"unknown OKX instruments: {missing}")
    store.save_instruments({**store.load_instruments(), **{s: all_inst[s] for s in symbols}})
    since = to_ms(args.since or cfg.start or "2024-01-01")
    tfs = args.tf.split(",") if args.tf else [cfg.backtest_matching_tf]
    for sym in symbols:
        for tf in tfs:
            warm = (cfg.warmup_bars + 5) * tf_ms(cfg.strategy_tf)
            store.sync_bars(rest, sym, tf, since_ms=since - warm, tick=all_inst[sym].tick_sz)
        store.sync_funding(rest, sym, since)
    print("done. store:", store.dir)


def cmd_backtest(args) -> None:
    from moneymachine.backtest import run_backtest
    from moneymachine.report import render

    cfg = load_config(args.config)
    llm = None
    if args.allow_llm:
        from moneymachine.llm.client import LLMClient

        load_env_file()
        llm = LLMClient(cfg.llm)
    run_id = run_backtest(cfg, start=args.start, end=args.end, allow_llm=args.allow_llm, llm_client=llm)
    print(render(cfg.db_path, run_id, cfg.benchmark, cfg.gonogo, out_dir=cfg.db_path.parent))
    print(f"run id: {run_id}  (report + daily equity CSV written to {cfg.db_path.parent}/)")


def cmd_paper(args) -> None:
    from moneymachine.live import run_paper

    cfg = load_config(args.config)
    resume = args.resume
    if resume == "latest":
        try:
            resume = _latest_run(cfg.db_path, "paper")
        except SystemExit:
            print("no previous paper run found — starting a new one")
            resume = None
    run_id = asyncio.run(run_paper(cfg, resume_run=resume, max_minutes=args.max_minutes))
    print(f"paper run {run_id} stopped. Resume with: mm paper --config {args.config} --resume {run_id}")


def cmd_report(args) -> None:
    from moneymachine.report import render

    cfg = load_config(args.config)
    db = Path(args.db) if args.db else cfg.db_path
    run_id = args.run or _latest_run(db)
    print(render(db, run_id, cfg.benchmark, cfg.gonogo, out_dir=db.parent))


def cmd_runs(args) -> None:
    from moneymachine.report import list_runs

    cfg = load_config(args.config)
    df = list_runs(Path(args.db) if args.db else cfg.db_path)
    if df.empty:
        print("no runs")
        return
    df["created_at"] = df["created_at"].map(lambda v: fmt_ms(int(v)))
    print(df.to_markdown(index=False))


def cmd_llm_check(args) -> None:
    """One tiny call per configured model through the relay: shows returned model id, latency, tokens."""
    from moneymachine.llm.client import LLMClient

    cfg = load_config(args.config)
    load_env_file()
    client = LLMClient(cfg.llm)
    if not client.configured():
        raise SystemExit(f"set {cfg.llm.base_url_env} and {cfg.llm.api_key_env} in .env first")
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}, "note": {"type": "string"}},
              "required": ["ok", "note"], "additionalProperties": False}
    names = [args.model] if args.model else [m.name for m in cfg.llm.models]
    for name in names:
        r = client.call_json(strategy="llm-check", purpose="check", symbol="-", model=name,
                             system="Reply with a JSON object only.",
                             user='Return {"ok": true, "note": "<one short sentence about yourself>"}',
                             schema=schema, now=now_ms())
        status = "OK " if r.ok else "FAIL"
        print(f"{status} {name:<24} returned={r.model_returned or '-':<28} {r.latency_ms:>6}ms "
              f"tokens={r.prompt_tokens}+{r.completion_tokens}  {r.error or (r.data or {}).get('note', '')}")


def cmd_ml_data(args) -> None:
    from moneymachine.market.okx_rest import OkxRest
    from moneymachine.ml import data as D, universe as U

    ins = D.crypto_instruments(OkxRest())
    D.run_parallel(lambda r, i: D.sync_candles(r, i, "1Dutc", "bars_1D"), list(ins["inst"]), workers=3, label="1D bars")
    P = U.daily_panel(list(ins["inst"]))
    uni = U.pit_universe(P["vol_quote"], ins.set_index("inst")["list_ts"], min_adv_usd=args.min_adv * 1e6,
                         max_names=args.max_names)
    cands = U.candidates(uni)
    Path("store/ml/candidates.txt").write_text("\n".join(cands) + "\n")
    spot = {d["instId"] for d in OkxRest().get("/api/v5/public/instruments", {"instType": "SPOT"})}
    D.run_parallel(lambda r, i: D.sync_candles(r, i, "4H", "bars_4H", start=1590969600000), cands, workers=3, label="4H")
    D.run_parallel(lambda r, s: D.sync_candles(r, s, "1Dutc", "spot_1D"),
                   [c.replace("-SWAP", "") for c in cands if c.replace("-SWAP", "") in spot], workers=3, label="spot")
    D.run_parallel(lambda r, i: D.sync_funding(r, i), cands, workers=2, label="funding")
    for kind in ("oi", "taker", "ls"):
        D.run_parallel(lambda r, i, k=kind: D.sync_rubik(r, i, k), cands, workers=1, label=f"rubik {kind}")
    print(f"done: {len(cands)} universe candidates")


def cmd_ml_research(args) -> None:
    from moneymachine.ml.research import gate_a, run_grid

    run_grid(rebuild_panel=args.rebuild)
    print(gate_a())


def cmd_ml_kronos(args) -> None:
    from moneymachine.ml.research import run_kronos

    print(run_kronos(model=args.model, sample_count=args.samples))


def cmd_ml_train(args) -> None:
    from moneymachine.ml.portfolio import CostModel, PortfolioConfig
    from moneymachine.ml.production import train_artifact

    kronos = None
    if args.kronos:
        kronos = {"model": args.kronos, "context": 256, "pred_len": 1, "sample_count": 5, "bar": "1D"}
    d = train_artifact(args.name, args.model, {}, args.horizon,
                       PortfolioConfig(kappa=args.kappa, alpha_horizon=args.horizon), CostModel(), kronos=kronos)
    print(f"artifact written to {d}")


def cmd_ml_collect(args) -> None:
    from moneymachine.ml.collector import run

    run(interval_s=args.interval, max_iters=args.iters)


def cmd_okx_check(args) -> None:
    """Read-only OKX account check (stage 1 of real trading): permissions, equity, positions."""
    from moneymachine.okx_private import ReadOnlyOkx

    load_env_file()
    snap = ReadOnlyOkx().snapshot()
    print(f"key label {snap['label']!r} | permission {snap['perm']} | IP whitelist {'set' if snap['ip_bound'] else 'NOT set'}")
    print(f"account level {snap['account_level']} | position mode {snap['position_mode']}")
    print(f"total equity {snap['total_equity_usd']:,.2f} USD | non-zero balances {len(snap['balances'])} | "
          f"positions {len(snap['positions'])} | open orders {snap['open_orders']}")


def cmd_execlab(args) -> None:
    """v4 E2: shadow post-only orders for the paper arms' rebalances (observation only)."""
    import asyncio

    from moneymachine.execlab import ExecLab

    cfg = load_config(args.config)
    asyncio.run(ExecLab(cfg.db_path).run())


def cmd_monitor(args) -> None:
    from moneymachine.monitor import daily_summary, run_once, send

    load_env_file()
    cfg = load_config(args.config)
    if args.summary:
        send("【刷钱机日报】", daily_summary(cfg.db_path))
        return
    if args.test:
        print("sent via:", send("【刷钱机测试】", "监控通道工作正常。") or ["journal-only"])
        return
    print(run_once(cfg.db_path, cfg.db_path.parent / "monitor_state.json", collect_dir=Path("store/ml/collect")))


def cmd_ml_gate(args) -> None:
    from moneymachine.ml.research import gate_a

    print(gate_a())


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="mm", description="刷钱机 1.0 — 虚拟钱包 + OKX 实时行情")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name, fn, help_):
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("--config", default="configs/v1.toml")
        sp.set_defaults(fn=fn)
        return sp

    data = sub.add_parser("data", help="market data")
    dsub = data.add_subparsers(dest="data_cmd", required=True)
    ds = dsub.add_parser("sync", help="download / update candles + funding into store/")
    ds.add_argument("--config", default="configs/v1.toml")
    ds.add_argument("--since", help="earliest date to cover (default [run].start)")
    ds.add_argument("--tf", help="comma list of timeframes (default backtest matching timeframe)")
    ds.set_defaults(fn=cmd_data_sync)

    bt = add("backtest", cmd_backtest, "replay history through the engine")
    bt.add_argument("--start")
    bt.add_argument("--end")
    bt.add_argument("--allow-llm", action="store_true", help="also run LLM strategies (costly, unreliable)")

    pp = add("paper", cmd_paper, "live paper trading (Ctrl+C to stop, --resume to continue)")
    pp.add_argument("--resume", help="run id to continue, or 'latest' (starts a new run if none exists)")
    pp.add_argument("--max-minutes", type=float, help="stop automatically after N minutes")

    rp = add("report", cmd_report, "performance report + go/no-go checklist")
    rp.add_argument("--run", help="run id (default: latest)")
    rp.add_argument("--db", help="ledger DB to read (default: [run].db from config)")

    rs = add("runs", cmd_runs, "list runs in the DB")
    rs.add_argument("--db", help="ledger DB to read (default: [run].db from config)")

    mo = add("monitor", cmd_monitor, "health checks + alerts (run every few minutes by a systemd timer)")
    mo.add_argument("--summary", action="store_true", help="send the daily summary now")
    mo.add_argument("--test", action="store_true", help="send a test message through the configured channels")

    add("execlab", cmd_execlab, "v4: shadow post-only orders next to the paper rebalances (never trades)")
    add("okx-check", cmd_okx_check, "read-only OKX account check (needs MM_OKX_* in .env; never trades)")

    lc = add("llm-check", cmd_llm_check, "test the relay with each configured model")
    lc.add_argument("--model")

    ml = sub.add_parser("ml", help="2.0 machine-learning research pipeline")
    msub = ml.add_subparsers(dest="ml_cmd", required=True)
    md = msub.add_parser("data", help="download / update the 2.0 data foundation (store/ml)")
    md.add_argument("--min-adv", type=float, default=20.0, help="universe: min 30d median volume, million USD")
    md.add_argument("--max-names", type=int, default=50)
    md.set_defaults(fn=cmd_ml_data)
    mr = msub.add_parser("research", help="run the pre-registered walk-forward grid, then print Gate A")
    mr.add_argument("--rebuild", action="store_true", help="rebuild the feature panel cache")
    mr.set_defaults(fn=cmd_ml_research)
    mk = msub.add_parser("kronos", help="Kronos signal on the clean period (2024-07 → holdout) + its trials")
    mk.add_argument("--model", default="NeoQuasar/Kronos-small")
    mk.add_argument("--samples", type=int, default=5)
    mk.set_defaults(fn=cmd_ml_kronos)
    mt = msub.add_parser("train", help="train a production artifact (pre-holdout data until the exam)")
    mt.add_argument("--name", required=True)
    mt.add_argument("--model", default="lgbm")
    mt.add_argument("--horizon", type=int, default=1)
    mt.add_argument("--kappa", type=float, default=1.0)
    mt.add_argument("--kronos", help="add a Kronos component, e.g. NeoQuasar/Kronos-small")
    mt.set_defaults(fn=cmd_ml_train)
    mc = msub.add_parser("collect", help="forward-collect spreads + open interest for all perps (runs forever)")
    mc.add_argument("--interval", type=int, default=300, help="seconds between snapshots")
    mc.add_argument("--iters", type=int, help="stop after N snapshots")
    mc.set_defaults(fn=cmd_ml_collect)
    mg = msub.add_parser("gate", help="print the Gate A report from recorded trials")
    mg.set_defaults(fn=cmd_ml_gate)

    args = p.parse_args(argv)
    _setup_logging(args.verbose)
    try:
        args.fn(args)
    except (ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
