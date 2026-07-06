"""Cost model v2: two instrument profiles, hand-computed values for every
component (SPEC_ADDENDUM_v2 R1)."""

import pytest

from backtest.costs import CASH, CFD, CostModel, InstrumentCosts


@pytest.fixture()
def cm():
    return CostModel(profiles={
        CASH: InstrumentCosts(
            commission_pct=0.0008, commission_min=1.0,
            half_spread_pct=0.0002, slippage_pct=0.0008,
            financing_enabled=False,
        ),
        CFD: InstrumentCosts(
            commission_pct=0.0, commission_min=0.0,
            half_spread_pct=0.0003, slippage_pct=0.0007,
            financing_enabled=True, benchmark_rate=0.05,
            long_markup=0.035, short_rate=0.0, day_count=360,
        ),
    })


def test_fill_price_by_instrument(cm):
    # cash impact = 0.0002 + 0.0008 = 0.001
    assert cm.fill_price(100.0, +1, CASH) == pytest.approx(100.1)
    assert cm.fill_price(100.0, -1, CASH) == pytest.approx(99.9)
    # cfd impact = 0.0003 + 0.0007 = 0.001
    assert cm.fill_price(200.0, -1, CFD) == pytest.approx(199.8)


def test_commission_cash_min_and_pct(cm):
    assert cm.commission(1000.0, CASH) == 1.0             # 0.08% = 0.8 < min 1.0
    assert cm.commission(10000.0, CASH) == pytest.approx(8.0)
    assert cm.commission(50000.0, CFD) == 0.0             # index CFD: spread only


def test_cash_equity_never_accrues_financing(cm):
    """R1 core: cash stock longs have NO overnight financing."""
    assert cm.financing_cost(50000, +1, 1, CASH) == 0.0
    assert cm.financing_cost(50000, +1, 30, CASH) == 0.0


def test_cfd_long_financing_hand_computed(cm):
    # 50,000 notional, 1 day: 50000 * (0.05+0.035) / 360
    assert cm.financing_cost(50000, +1, 1, CFD) == pytest.approx(50000 * 0.085 / 360)
    # weekend (3 days)
    assert cm.financing_cost(50000, +1, 3, CFD) == pytest.approx(50000 * 0.085 * 3 / 360)


def test_cfd_short_hedge_zero_financing(cm):
    """R1: hedge shorts booked at ZERO financing (no credit, conservative)."""
    assert cm.financing_cost(50000, -1, 1, CFD) == 0.0
    assert cm.financing_cost(50000, -1, 3, CFD) == 0.0


def test_unknown_instrument_rejected(cm):
    with pytest.raises(ValueError, match="unknown instrument"):
        cm.fill_price(100.0, 1, "swap")


def test_from_config_loads():
    m = CostModel.from_config()
    cash = m.profile(CASH)
    cfd = m.profile(CFD)
    assert not cash.financing_enabled                     # R1
    assert cfd.financing_enabled and cfd.short_rate == 0.0
    assert cash.slippage_pct >= 0.0005                    # SPEC §7 slippage floor
    assert cash.commission_min == 1.0                     # R1: min $1
