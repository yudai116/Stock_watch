"""Cost model: hand-computed values for every component."""

import pytest

from backtest.costs import CostModel


@pytest.fixture()
def cm():
    return CostModel(
        commission_pct=0.001, commission_min=3.0,
        half_spread_pct=0.0003, slippage_pct=0.0007,
        benchmark_rate=0.05, long_markup=0.025,
        short_markdown=0.025, borrow_default=0.005, day_count=360,
    )


def test_fill_price_buy_and_sell(cm):
    # impact = 0.0003 + 0.0007 = 0.001
    assert cm.fill_price(100.0, +1) == pytest.approx(100.1)
    assert cm.fill_price(100.0, -1) == pytest.approx(99.9)


def test_commission_min_and_pct(cm):
    assert cm.commission(1000.0) == 3.0            # 0.1% = 1.0 < min 3.0
    assert cm.commission(10000.0) == pytest.approx(10.0)


def test_financing_long_hand_computed(cm):
    # 50,000 notional, 1 day: 50000 * (0.05+0.025) / 360 = 10.4166..
    assert cm.financing_cost(50000, +1, 1) == pytest.approx(50000 * 0.075 / 360)
    # weekend (3 days)
    assert cm.financing_cost(50000, +1, 3) == pytest.approx(50000 * 0.075 * 3 / 360)


def test_financing_short_net(cm):
    # short: borrow + markdown - benchmark = 0.005 + 0.025 - 0.05 = -0.02 -> CREDIT
    assert cm.financing_cost(50000, -1, 1) == pytest.approx(50000 * -0.02 / 360)
    # hard-to-borrow overrides default
    assert cm.financing_cost(50000, -1, 1, borrow_annual=0.08) == pytest.approx(
        50000 * (0.08 + 0.025 - 0.05) / 360)


def test_financing_zero_days(cm):
    assert cm.financing_cost(50000, +1, 0) == 0.0


def test_from_config_loads():
    m = CostModel.from_config()
    assert 0 < m.commission_pct < 0.01
    assert m.slippage_pct >= 0.0005  # SPEC §7: slippage floor 0.05%
