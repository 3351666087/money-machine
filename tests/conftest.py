from __future__ import annotations

import pytest

from moneymachine.ledger import Ledger
from moneymachine.sim.broker import Broker
from moneymachine.sim.funding import FundingBook
from moneymachine.types import Bar, Instrument

M = 60_000  # 1 minute
H = 3_600_000
SYM = "BTC-USDT-SWAP"


@pytest.fixture
def inst():
    return {SYM: Instrument(SYM, ct_val=0.01, lot_sz=0.01, min_sz=0.01, tick_sz=0.1)}


@pytest.fixture
def ledger():
    lg = Ledger(":memory:", "t")
    yield lg
    lg.close()


def make_broker(inst, ledger, *, funding=None, slip_bps=0.0, mmr=0.005):
    b = Broker(maker_fee=0.0002, taker_fee=0.0005, slippage_bps=slip_bps, mmr=mmr, instruments=inst,
               funding=funding or FundingBook(0.0, 8), bar_ms=M, ledger=ledger)
    b.add_account("s", 10_000.0)
    return b


def bar(ts, o, h, l, c, v=1.0):
    return Bar(ts, o, h, l, c, v)


def rows(ledger, sql, *params):
    return ledger.db.execute(sql, params).fetchall()
