"""2.0 research framework: leakage, validation, statistics, portfolio constraints."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from moneymachine.ml import stats, validation as V
from moneymachine.ml.features import BTC, build_panel, to_long
from moneymachine.ml.portfolio import CostModel, PortfolioConfig, optimize

DAY = 86_400_000
H4 = 4 * 3_600_000
T0 = 1_640_995_200_000  # 2022-01-01
INSTS = [BTC, "ETH-USDT-SWAP", "SOL-USDT-SWAP", "DOGE-USDT-SWAP"]


def _write_store(root, n_days=400, cut: int | None = None, seed=0):
    rng = np.random.default_rng(seed)
    for k, inst in enumerate(INSTS):
        n4 = n_days * 6
        r = rng.normal(0.0002 * (k - 1), 0.01, n4)
        c = 100 * (k + 1) * np.exp(np.cumsum(r))
        o = np.concatenate(([c[0]], c[:-1]))
        h = np.maximum(o, c) * 1.002
        l = np.minimum(o, c) * 0.998
        ts = T0 + np.arange(n4) * H4
        df4 = pd.DataFrame({"time": ts, "open": o, "high": h, "low": l, "close": c,
                            "vol_ccy": 1.0, "vol_quote": 5e7 / 6})
        if cut is not None:
            df4 = df4[df4["time"] + H4 <= cut]
        (root / "bars_4H").mkdir(parents=True, exist_ok=True)
        df4.to_parquet(root / "bars_4H" / f"{inst}.parquet", index=False)
        g = (df4["time"] - T0) // DAY
        d1 = df4.groupby(g).agg(time=("time", "first"), open=("open", "first"), high=("high", "max"),
                                low=("low", "min"), close=("close", "last"), vol_ccy=("vol_ccy", "sum"),
                                vol_quote=("vol_quote", "sum"))
        d1 = d1[d1["time"] + DAY <= (cut if cut is not None else 10**15)]
        (root / "bars_1D").mkdir(parents=True, exist_ok=True)
        d1.to_parquet(root / "bars_1D" / f"{inst}.parquet", index=False)
        fts = T0 + np.arange(n_days * 3) * 8 * 3_600_000
        fr = pd.DataFrame({"time": fts, "rate": rng.normal(1e-4, 5e-5, len(fts))})
        if cut is not None:
            fr = fr[fr["time"] <= cut]
        (root / "funding").mkdir(parents=True, exist_ok=True)
        fr.to_parquet(root / "funding" / f"{inst}.parquet", index=False)


def _universe(n_days=400):
    days = T0 + np.arange(n_days) * DAY
    return pd.DataFrame(True, index=pd.Index(days, dtype="int64"), columns=INSTS)


def test_features_do_not_peek_into_the_future(tmp_path):
    full, cut = tmp_path / "full", tmp_path / "cut"
    cutoff = T0 + 300 * DAY
    _write_store(full)
    _write_store(cut, cut=cutoff)
    pf = build_panel(_universe(), root=full, start="2022-01-01")
    pc = build_panel(_universe(), root=cut, start="2022-01-01")
    days = pc.days[(pc.days <= cutoff) & (pc.days >= T0 + 120 * DAY)]
    for name in pf.feats:
        a = pf.feats[name].reindex(days)
        b = pc.feats[name].reindex(days)
        pd.testing.assert_frame_equal(a, b, check_names=False, obj=f"feature {name}")
    pd.testing.assert_frame_equal(pf.market.reindex(days), pc.market.reindex(days), check_names=False)


def test_labels_are_forward_and_residualised(tmp_path):
    _write_store(tmp_path)
    p = build_panel(_universe(), root=tmp_path, start="2022-01-01")
    lp = np.log(p.px)
    assert np.allclose(p.fwd_lr[1].iloc[100:200], (lp.shift(-1) - lp).iloc[100:200], equal_nan=True)
    df = to_long(p, 1, min_names=3)
    ranks = df["y"].dropna()
    assert ranks.between(-0.5, 0.5).all()


def test_holdout_guard_and_exam_once(tmp_path, monkeypatch):
    with pytest.raises(V.HoldoutViolation):
        V.guard_dev(np.array([V.HOLDOUT_START - DAY]), label_horizon_days=3)
    V.guard_dev(np.array([V.HOLDOUT_START - 4 * DAY]), label_horizon_days=3)
    monkeypatch.setattr(V, "EXAM_LOG", tmp_path / "exam.jsonl")
    V.open_exam({"a": 1}, "first")
    with pytest.raises(V.HoldoutViolation):
        V.open_exam({"a": 1}, "second")
    assert pd.Timestamp(V.HOLDOUT_START, unit="ms") == pd.Timestamp("2026-04-01")


def test_walk_forward_purges_label_overlap():
    days = T0 + np.arange(900) * DAY
    splits = V.walk_forward(days, first_test=T0 + 400 * DAY, test_days=91, horizon_days=3, embargo_days=1)
    assert len(splits) >= 5
    for sp in splits:
        assert sp.train_end + 3 * DAY <= sp.test_start - DAY + 1  # label window + embargo before test
        assert sp.test_end - sp.test_start <= 91 * DAY


def test_dsr_penalises_many_trials():
    rng = np.random.default_rng(0)
    r = rng.normal(0.001, 0.02, 1000)  # ~1 annual Sharpe
    one = stats.deflated_sharpe(r, np.array([0.05]))
    many = stats.deflated_sharpe(r, rng.normal(0, 0.03, 500))
    assert many["dsr"] < one["dsr"]
    assert many["sr0_per_period"] > 0


def test_pbo_detects_pure_noise_selection():
    # on pure noise PBO is ~0.5 on average (a single draw is very noisy)
    vals = [stats.pbo_cscv(np.random.default_rng(s).normal(0, 0.01, (600, 30)), 8)["pbo"] for s in range(20)]
    assert 0.35 < np.mean(vals) < 0.65
    noise = np.random.default_rng(1).normal(0, 0.01, (800, 40))
    skill = noise.copy()
    skill[:, 0] += 0.004  # one genuinely good config
    assert stats.pbo_cscv(skill, 10)["pbo"] < 0.1


def test_optimizer_respects_constraints():
    rng = np.random.default_rng(2)
    n = 20
    sig = rng.normal(size=n)
    beta = rng.uniform(0.6, 1.4, n)
    active = np.ones(n, bool)
    active[-3:] = False
    cfg = PortfolioConfig(cap=0.08, gross_max=1.5, vol_target=0.10, nav=1e5)
    w = optimize(sig, np.zeros(n), active, beta, np.full(n, 0.03 ** 2), 0.03 ** 2, np.full(n, 0.04),
                 np.full(n, 5e7), cfg, CostModel())
    assert abs(beta @ w) < 1e-5
    assert np.abs(w).sum() <= 1.5 + 1e-5
    assert np.abs(w).max() <= 0.08 + 1e-6
    assert np.allclose(w[-3:], 0)
    assert np.corrcoef(w[:-3], sig[:-3])[0, 1] > 0.5


def test_costs_shrink_trading():
    rng = np.random.default_rng(3)
    n = 15
    sig = rng.normal(size=n)
    w0 = rng.normal(0, 0.03, n)
    args = (np.ones(n, bool), np.ones(n), np.full(n, 0.03 ** 2), 0.03 ** 2, np.full(n, 0.04), np.full(n, 5e7))
    cheap = optimize(sig, w0, *args, PortfolioConfig(kappa=0.1), CostModel())
    dear = optimize(sig, w0, *args, PortfolioConfig(kappa=20.0), CostModel())
    assert np.abs(dear - w0).sum() < np.abs(cheap - w0).sum()


def test_universe_is_known_at_decision_time():
    from moneymachine.ml.universe import pit_universe

    days = pd.Index(T0 + np.arange(100) * DAY, dtype="int64")
    vq = pd.DataFrame({"A": 5e7, "B": 5e7}, index=days)
    lt = pd.Series({"A": T0 - 100 * DAY, "B": T0 - 100 * DAY})
    u = pit_universe(vq, lt, min_adv_usd=1e7, min_age_days=0)
    today = int(days.max()) + DAY  # the day whose bar has not started yet
    assert today in u.index and u.loc[today].all()
    # changing data on day t must not change membership on day t
    vq2 = vq.copy()
    vq2.loc[days[60]] = 1.0
    u2 = pit_universe(vq2, lt, min_adv_usd=1e7, min_age_days=0)
    assert u2.loc[days[60]].equals(u.loc[days[60]])


def test_placebo_signal_persistence_and_parity():
    from moneymachine.ml.placebo import placebo_signal

    coins = [f"C{i}" for i in range(30)]
    days = T0 + np.arange(600) * DAY
    z = placebo_signal(coins, days, seed=3)
    ac1 = np.mean([z[c].autocorr(1) for c in coins])
    ac5 = np.mean([z[c].autocorr(5) for c in coins])
    assert 0.47 < ac1 < 0.60 and 0.24 < ac5 < 0.40  # model signal: 0.536 / 0.324
    assert abs(z.values.std() - 1) < 0.05
    one = placebo_signal(coins[:4], [days[400]], seed=3)  # live computes a single day
    assert np.allclose(one.iloc[0].values, z.iloc[400][coins[:4]].values)


def test_placebo_strategy_registered():
    from moneymachine.strategies.registry import KINDS

    assert KINDS["ml_placebo"].portfolio and KINDS["ml_placebo"].slow


def test_v3_smoothing_matches_manual_ema_and_resets_after_gap():
    from moneymachine.ml.models import zscore_by_date
    from moneymachine.ml.research_v3 import smooth

    rng = np.random.default_rng(3)
    coins = ["A", "B", "C", "D"]
    days = T0 + np.arange(30) * DAY
    raw = pd.DataFrame(rng.normal(size=(30, 4)), index=days, columns=coins)
    raw.loc[days[10:17], "D"] = np.nan          # D absent 7 days (> 5) → restarts at day 17
    raw.loc[days[20:23], "C"] = np.nan          # C absent 3 days → keeps its state
    long = raw.stack().rename("pred")
    long.index.names = ["t", "inst"]
    got = smooth(long, half_life=2)

    a = 1 - 0.5 ** 0.5
    ema = raw.copy()
    for c in coins:
        st, last = np.nan, None
        for k, t in enumerate(days):
            x = raw.at[t, c]
            if np.isnan(x):
                continue
            st = x if (np.isnan(st) or (t - last) // DAY - 1 > 5) else (1 - a) * st + a * x
            ema.at[t, c], last = st, t
    exp = ema.where(raw.notna()).stack().dropna()
    exp = pd.Series(zscore_by_date(exp.to_numpy(), exp.index.get_level_values(0).to_numpy()), index=exp.index)
    assert np.allclose(got.to_numpy(), exp.to_numpy())


def test_v3_concentrated_book(tmp_path):
    from moneymachine.ml.portfolio import Lots, run_concentrated

    _write_store(tmp_path)
    panel = build_panel(_universe(), root=tmp_path, start="2022-01-01")
    rng = np.random.default_rng(1)
    sig = pd.DataFrame(np.cumsum(rng.normal(size=(len(panel.days), len(INSTS))), axis=0) * 0.1,
                       index=panel.days, columns=INSTS)
    lot = np.full(len(INSTS), 0.01)
    res = run_concentrated(panel, sig, PortfolioConfig(), CostModel(), start=int(panel.days[60]), n=1, buffer=2,
                           nav=1000.0, lots=Lots(lot, lot))
    d, W = res.daily, res.weights
    legs = [c for c in INSTS if c != BTC]
    assert (d["n_long"] <= 1).all() and (d["n_short"] <= 1).all() and (d["n_long"] == 1).mean() > 0.9
    assert ((W[legs] > 0).sum(axis=1) <= 1).all()
    assert d["beta_exp"].abs().max() < 0.05                  # BTC hedge, up to one lot
    eq = 1000 * np.prod(1 + d["net"].to_numpy())
    assert abs(eq - res.config["final_equity"]) < 1e-6
    held = (W[legs] != 0)
    changes = (held != held.shift()).any(axis=1).iloc[1:].mean()
    assert changes < 0.5                                      # buffer: membership is sticky


def test_event_ticker_extraction():
    from moneymachine.ml.events import tickers

    bases = {"AI", "AIA", "BNB", "ZKC", "PEPE", "EPIC", "QUICK", "WAL", "PLUME", "ETH", "KMNO", "SYRUP"}
    assert tickers("DeAgent AI (AIA) Will Be Available on Binance Alpha", bases) == {"AIA"}
    assert tickers("Introducing Boundless (ZKC) on Binance HODLer Airdrops! Earn ZKC With Retroactive BNB", bases) == {"ZKC"}
    assert tickers("Binance Will Delist EPIC, QUICK on 2026-10-28", bases) == {"EPIC", "QUICK"}
    assert tickers("Binance Futures Will Launch USDⓈ-Margined 1000PEPEUSDT Perpetual Contract", bases) == {"PEPE"}
    assert tickers("OKX to list perpetual futures for WAL, NIL, PLUME crypto", bases) == {"WAL", "PLUME"}
    assert tickers("Binance Will List Maple Finance (SYRUP) and Kamino Finance (KMNO)", bases) == {"SYRUP", "KMNO"}
    assert tickers("Binance Will Add New Trading Pairs ZKC/ETH", bases) == {"ZKC"}


def test_live_ema_matches_research_smoothing():
    from moneymachine.ml.production import ema_update
    from moneymachine.ml.research_v3 import smooth

    rng = np.random.default_rng(5)
    coins = [f"C{i}" for i in range(8)]
    days = T0 + np.arange(40) * DAY
    raw = pd.DataFrame(rng.normal(size=(40, 8)), index=days, columns=coins)
    raw.iloc[10:18, 2] = np.nan          # long gap → restart
    raw.iloc[25:28, 5] = np.nan          # short gap → continue
    long = raw.stack().rename("pred")
    long.index.names = ["t", "inst"]
    want = smooth(long, half_life=2)
    state: dict = {}
    for t in days:
        row = raw.loc[t].dropna()
        got = ema_update(state, row, int(t), 2)
        assert np.allclose(got.sort_index().to_numpy(), want.xs(t, level="t").sort_index().to_numpy())
    again = ema_update(state, raw.loc[days[-1]].dropna(), int(days[-1]), 2)   # same day twice: no double update
    assert np.allclose(again.sort_index().to_numpy(), want.xs(days[-1], level="t").sort_index().to_numpy())


def _synthetic_panel(n_coins=12, n_days=120, seed=2):
    from types import SimpleNamespace

    rng = np.random.default_rng(seed)
    insts = [BTC] + [f"C{i}-USDT-SWAP" for i in range(n_coins - 1)]
    days = pd.Index(T0 + np.arange(n_days) * DAY, dtype="int64")
    lr = rng.normal(0, 0.04, (n_days, n_coins))
    px = pd.DataFrame(np.exp(np.cumsum(lr, axis=0)) * rng.uniform(0.05, 500, n_coins), index=days, columns=insts)
    uni = pd.DataFrame(rng.uniform(size=(n_days, n_coins)) > 0.1, index=days, columns=insts)
    beta = pd.DataFrame(rng.uniform(0.6, 1.6, (n_days, n_coins)), index=days, columns=insts)
    beta[BTC] = 1.0
    fwd = np.log(px).shift(-1) - np.log(px)
    const = lambda v: pd.DataFrame(v, index=days, columns=insts)  # noqa: E731
    return SimpleNamespace(days=days, insts=insts, px=px, universe=uni, beta=beta, fwd_lr={1: fwd},
                           fund_daily=const(1e-4), vol_daily=const(0.04), adv=const(5e7))


def test_live_concentrated_book_matches_research_backtest():
    """Feed the live rule research's holdings and equity each day; it must choose the same quantities."""
    from moneymachine.ml.portfolio import Lots, run_concentrated
    from moneymachine.ml.production import concentrated_targets
    from moneymachine.types import Instrument

    panel = _synthetic_panel()
    rng = np.random.default_rng(9)
    sig = pd.DataFrame(np.cumsum(rng.normal(size=(len(panel.days), len(panel.insts))), axis=0) * 0.3,
                       index=panel.days, columns=panel.insts)
    specs = {s: Instrument(s, ct_val=float(rng.choice([0.001, 0.01, 1, 10])), lot_sz=1.0, min_sz=1.0)
             for s in panel.insts}
    specs[BTC] = Instrument(BTC, ct_val=0.0001, lot_sz=1.0, min_sz=1.0)
    lots = Lots(np.array([specs[s].qty_step for s in panel.insts]), np.array([specs[s].min_qty for s in panel.insts]))
    nav = 1000.0
    res = run_concentrated(panel, sig, PortfolioConfig(), CostModel(), n=3, nav=nav, lots=lots)
    W, net = res.weights, res.daily["net"].to_numpy()
    eq = nav * np.concatenate(([1.0], np.cumprod(1 + net)[:-1]))
    q_res = W.to_numpy() * eq[:, None] / panel.px.ffill(limit=3).to_numpy()
    prev = {}
    checked = 0
    for k, t in enumerate(panel.days):
        marks = panel.px.iloc[k].to_dict()
        s = sig.iloc[k][panel.universe.iloc[k]]
        targets, info = concentrated_targets(s, panel, int(t), positions=prev, marks=marks, equity=eq[k],
                                             instruments=specs, n=3)
        live = {x: w * eq[k] / marks[x] for x, w in targets.items() if w}
        want = {x: q for x, q in zip(panel.insts, q_res[k]) if abs(q) > 1e-12}
        assert set(live) == set(want), (k, sorted(live), sorted(want))
        assert all(abs(live[x] - want[x]) <= 1e-6 * max(1.0, abs(want[x])) for x in want)
        assert info["n_long"] <= 3 and info["n_short"] <= 3
        prev = {x: q for x, q in want.items()}
        checked += bool(want)
    assert checked > 100
    assert res.daily["turnover"].mean() < 0.5     # the buffer keeps it from churning every day


def test_minute_executor_fill_rules():
    from moneymachine.ml.execution import MAKER_FEE, SLIP, TAKER_FEE, MinuteExecutor

    t, inst, p = 1_720_000_000_000 - 1_720_000_000_000 % DAY, "X-USDT-SWAP", 100.0
    px = pd.DataFrame({inst: [p]}, index=[t])
    costs = CostModel(hs_a=1e-4, hs_b=0.0, hs_min=1e-4, hs_max=1e-4)          # half-spread exactly 1bp
    flat = np.tile([100.0, 100.05, 99.995], (240, 1))                           # never trades through 99.99
    dip = flat.copy()
    dip[5] = [100.0, 100.0, 99.98]                                              # minute 5 trades 1 tick through
    dip[30:, 0] = 100.2
    flat[30:, 0] = 100.2                                                        # drifted away by minute 30
    run = lambda pol, bars: MinuteExecutor(pol, px, {(inst, t): bars}, {inst: 0.01}, costs)(  # noqa: E731
        t, [inst], np.array([0.0]), np.array([0.1]), np.array([1e9]), np.array([0.0]), 1e5)
    w, c = run("taker", flat)
    assert c[0] == pytest.approx(0.1 * ((100.0 * (1 + 1e-4 + SLIP) - p) / p + TAKER_FEE))
    w, c = run("M30", dip)                                                      # filled as maker at 99.99
    assert w[0] == 0.1 and c[0] == pytest.approx(0.1 * ((99.99 - p) / p + MAKER_FEE))
    w, c = run("M30", flat)                                                     # not filled → taker at minute 30
    assert c[0] == pytest.approx(0.1 * ((100.2 * (1 + 1e-4 + SLIP) - p) / p + TAKER_FEE))
    w, c = run("MO", flat)                                                      # not filled → cancelled
    assert w[0] == 0.0 and c[0] == 0.0
    ex = MinuteExecutor("M30", px, {(inst, t): dip}, {inst: 0.01}, costs, through=3)   # stricter: 3 ticks
    w, c = ex(t, [inst], np.array([0.0]), np.array([0.1]), np.array([1e9]), np.array([0.0]), 1e5)
    assert ex.stats["fallback"] == 1
