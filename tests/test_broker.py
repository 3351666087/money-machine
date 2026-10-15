from __future__ import annotations

import pytest

from conftest import M, SYM, bar, make_broker, rows
from moneymachine.sim.funding import FundingBook


def test_market_order_fills_next_open_with_slippage_and_taker_fee(inst, ledger):
    b = make_broker(inst, ledger, slip_bps=10)
    b.on_bar(SYM, bar(0, 100, 101, 99, 100))
    b.submit("s", symbol=SYM, side="buy", type="market", qty=1.0, price=None, role="entry",
             active_from=M, now=M)
    b.on_bar(SYM, bar(M, 100, 102, 99.5, 101))
    acct = b.accounts["s"]
    p = acct.positions[SYM]
    assert p.qty == pytest.approx(1.0)
    assert p.avg_px == pytest.approx(100 * 1.001)
    assert acct.balance == pytest.approx(10_000 - 100.1 * 0.0005)


def test_order_never_fills_on_bar_before_it_was_active(inst, ledger):
    b = make_broker(inst, ledger)
    b.on_bar(SYM, bar(0, 100, 101, 99, 100))
    b.submit("s", symbol=SYM, side="buy", type="market", qty=1.0, price=None, role="entry",
             active_from=2 * M, now=M + 30_000)
    b.on_bar(SYM, bar(M, 100, 101, 99, 100))
    assert SYM not in b.accounts["s"].positions or b.accounts["s"].positions[SYM].qty == 0
    b.on_bar(SYM, bar(2 * M, 105, 106, 104, 105))
    assert b.accounts["s"].positions[SYM].avg_px == pytest.approx(105)


def test_limit_requires_strict_trade_through(inst, ledger):
    b = make_broker(inst, ledger)
    b.on_bar(SYM, bar(0, 100, 101, 99, 100))
    b.submit("s", symbol=SYM, side="buy", type="limit", qty=1.0, price=98.0, role="entry",
             active_from=M, now=M)
    b.on_bar(SYM, bar(M, 100, 100, 98.0, 99))  # touches 98 exactly → no fill
    assert not b.accounts["s"].positions.get(SYM) or b.accounts["s"].positions[SYM].qty == 0
    b.on_bar(SYM, bar(2 * M, 99, 99, 97.9, 98.5))  # trades through → fill at limit, maker
    p = b.accounts["s"].positions[SYM]
    assert p.qty == pytest.approx(1.0) and p.avg_px == pytest.approx(98.0)
    fee = rows(ledger, "SELECT fee, liquidity FROM fills")[0]
    assert fee[1] == "maker" and fee[0] == pytest.approx(98 * 0.0002)


def test_marketable_limit_becomes_taker(inst, ledger):
    b = make_broker(inst, ledger)
    b.on_bar(SYM, bar(0, 100, 101, 99, 100))
    o = b.submit("s", symbol=SYM, side="buy", type="limit", qty=1.0, price=105.0, role="entry",
                 active_from=M, now=M)
    assert o.type == "market"


def test_stop_loss_gap_fills_at_open_and_beats_take_profit_in_same_bar(inst, ledger):
    b = make_broker(inst, ledger)
    b.on_bar(SYM, bar(0, 100, 100, 100, 100))
    b.submit("s", symbol=SYM, side="buy", type="market", qty=1.0, price=None, role="entry",
             active_from=M, now=M, stop_loss=95.0, take_profit=110.0)
    b.on_bar(SYM, bar(M, 100, 100.5, 99.5, 100))      # entry
    b.on_bar(SYM, bar(2 * M, 94, 111, 93, 105))       # gap below stop AND tp touched
    t = rows(ledger, "SELECT exit_reason, exit_px, gross_pnl FROM trades")
    assert len(t) == 1
    assert t[0][0] == "sl"
    assert t[0][1] == pytest.approx(94.0)  # gap → filled at open, not at 95
    assert b.accounts["s"].open_orders() == []  # OCO cancelled the TP


def test_entry_bar_stop_is_live_but_tp_is_not(inst, ledger):
    b = make_broker(inst, ledger)
    b.on_bar(SYM, bar(0, 100, 100, 100, 100))
    b.submit("s", symbol=SYM, side="buy", type="market", qty=1.0, price=None, role="entry",
             active_from=M, now=M, stop_loss=99.0, take_profit=101.0)
    b.on_bar(SYM, bar(M, 100, 102, 100, 101))  # TP level crossed on entry bar → not taken
    assert b.accounts["s"].positions[SYM].qty == pytest.approx(1.0)
    b.on_bar(SYM, bar(2 * M, 101, 101.5, 100.5, 101))  # now TP fills
    t = rows(ledger, "SELECT exit_reason, exit_px FROM trades")
    assert t == [("tp", 101.0)]


def test_short_trade_pnl_and_record(inst, ledger):
    b = make_broker(inst, ledger)
    b.on_bar(SYM, bar(0, 100, 100, 100, 100))
    b.submit("s", symbol=SYM, side="sell", type="market", qty=2.0, price=None, role="entry",
             active_from=M, now=M, stop_loss=105.0)
    b.on_bar(SYM, bar(M, 100, 100, 99, 99))
    b.close_position("s", SYM, now=2 * M, active_from=2 * M)
    b.on_bar(SYM, bar(2 * M, 90, 91, 89, 90))
    side, gross, fees, net, r = rows(ledger, "SELECT side, gross_pnl, fees, net_pnl, r_multiple FROM trades")[0]
    assert side == "short"
    assert gross == pytest.approx(2 * (100 - 90))
    assert fees == pytest.approx(2 * 100 * 0.0005 + 2 * 90 * 0.0005)
    assert net == pytest.approx(gross - fees)
    assert r == pytest.approx(net / (5 * 2))
    acct = b.accounts["s"]
    assert acct.balance == pytest.approx(10_000 + net)


def test_funding_long_pays_positive_rate(inst, ledger):
    fb = FundingBook(0.0, 8)
    fb.add(SYM, [(10 * M, 0.001)])
    b = make_broker(inst, ledger, funding=fb)
    b.on_bar(SYM, bar(0, 100, 100, 100, 100))
    b.submit("s", symbol=SYM, side="buy", type="market", qty=1.0, price=None, role="entry",
             active_from=M, now=M)
    for k in range(1, 12):
        b.on_bar(SYM, bar(k * M, 100, 100, 100, 100))
    f = rows(ledger, "SELECT rate, amount FROM funding")
    assert f == [(0.001, pytest.approx(-0.1))]


def test_funding_not_charged_when_flat(inst, ledger):
    fb = FundingBook(0.0, 8)
    fb.add(SYM, [(5 * M, 0.01)])
    b = make_broker(inst, ledger, funding=fb)
    for k in range(10):
        b.on_bar(SYM, bar(k * M, 100, 100, 100, 100))
    assert rows(ledger, "SELECT COUNT(*) FROM funding")[0][0] == 0


def test_liquidation_on_worst_price(inst, ledger):
    b = make_broker(inst, ledger)
    b.on_bar(SYM, bar(0, 100, 100, 100, 100))
    # 50x notional on 10k equity: a 2% drop wipes it out
    b.submit("s", symbol=SYM, side="buy", type="market", qty=5000.0, price=None, role="entry",
             active_from=M, now=M)
    b.on_bar(SYM, bar(M, 100, 100, 99.9, 100))
    b.on_bar(SYM, bar(2 * M, 100, 100, 97.5, 99))
    acct = b.accounts["s"]
    assert acct.liquidations == 1
    assert acct.positions[SYM].qty == 0
    assert acct.balance >= 0
    assert rows(ledger, "SELECT exit_reason FROM trades")[0][0] == "liquidation"


def test_limit_entry_expires(inst, ledger):
    b = make_broker(inst, ledger)
    b.on_bar(SYM, bar(0, 100, 100, 100, 100))
    b.submit("s", symbol=SYM, side="buy", type="limit", qty=1.0, price=90.0, role="entry",
             active_from=M, now=M, expire_at=3 * M)
    for k in range(1, 5):
        b.on_bar(SYM, bar(k * M, 100, 100, 99, 100))
    assert b.accounts["s"].open_orders() == []
    assert rows(ledger, "SELECT status FROM orders")[0][0] == "expired"


def test_state_roundtrip(inst, ledger):
    b = make_broker(inst, ledger)
    b.on_bar(SYM, bar(0, 100, 100, 100, 100))
    b.submit("s", symbol=SYM, side="buy", type="market", qty=1.0, price=None, role="entry",
             active_from=M, now=M, stop_loss=90.0)
    b.on_bar(SYM, bar(M, 100, 100, 100, 100))
    st = b.to_state()
    b2 = make_broker(inst, ledger)
    b2.load_state(st)
    assert b2.to_state() == st
