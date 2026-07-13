import pandas as pd
import pytest

import factor_builder as fb


def _panel(period="P1"):
    net_income = pd.DataFrame({"cik": [1, 2, 3, 4, 5], "val": [100, -50, 20, 30, 5],
                                "source_tag": ["NetIncomeLoss"] * 5})
    equity = pd.DataFrame({"cik": [1, 2, 3, 4, 5], "val": [500, -10, 200, 150, 400],
                            "source_tag": ["StockholdersEquity"] * 5})
    assets = pd.DataFrame({"cik": [1, 2, 3, 4, 5], "val": [1000, 800, 600, 900, 700],
                            "source_tag": ["Assets"] * 5})
    liabs = pd.DataFrame({"cik": [1, 2, 3, 4, 5], "val": [400, 700, 100, 300, 600],
                           "source_tag": ["Liabilities"] * 5})
    return fb.assemble_accounting_panel(net_income, equity, assets, liabs, period=period)


def test_assemble_accounting_panel_shapes_and_provenance():
    panel = _panel()
    assert len(panel) == 5
    assert {"period", "cik", "net_income", "stockholders_equity",
            "total_assets", "liabilities", "net_income_source_tag"}.issubset(panel.columns)


def test_roe_masks_negative_equity():
    out = fb.compute_roe(_panel())
    row2 = out[out["cik"] == 2].iloc[0]
    assert row2["equity_negative_flag"]
    assert pd.isna(row2["roe"])
    row1 = out[out["cik"] == 1].iloc[0]
    assert row1["roe"] == pytest.approx(100 / 500)


def test_leverage_basic():
    out = fb.compute_leverage(_panel())
    row1 = out[out["cik"] == 1].iloc[0]
    assert row1["leverage"] == pytest.approx(400 / 1000)


def test_winsorize_percentile_clips_outliers():
    df = pd.DataFrame({"period": ["P1"] * 10, "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 1000]})
    out = fb.winsorize_cross_sectional(df, "x", group_cols=["period"],
                                        method="percentile", lower=0.01, upper=0.90)
    assert out.max() < 1000


def test_winsorize_mad_method():
    df = pd.DataFrame({"period": ["P1"] * 8, "x": [10, 11, 9, 10, 12, 11, 9, 500]})
    out = fb.winsorize_cross_sectional(df, "x", group_cols=["period"], method="mad", n_mad=3)
    assert out.max() < 500


def test_zscore_respects_min_group_size():
    df = pd.DataFrame({"period": ["P1"] * 3, "x": [1.0, 2.0, 3.0]})
    out = fb.zscore_cross_sectional(df, "x", group_cols=["period"], min_group_size=5)
    assert out.isna().all()


def test_zscore_basic_stats():
    df = pd.DataFrame({"period": ["P1"] * 6, "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    out = fb.zscore_cross_sectional(df, "x", group_cols=["period"], min_group_size=5)
    assert out.mean() == pytest.approx(0, abs=1e-9)
    assert out.std(ddof=0) == pytest.approx(1, abs=1e-9)


def test_build_qmj_panel_end_to_end_and_leverage_sign_flip():
    result = fb.build_qmj_panel(_panel(), config=fb.QMJConfig(min_group_size=3))
    assert result.index.names == ["period", "cik"]
    assert "qmj_score" in result.columns
    row_high_lev = result.loc[("P1", 2)]   # cik 2: 700/800 -> leverage alto
    row_low_lev = result.loc[("P1", 3)]    # cik 3: 100/600 -> leverage bajo
    assert row_high_lev["safety_zscore"] < row_low_lev["safety_zscore"]


def test_build_qmj_panel_sector_relative():
    universe = pd.DataFrame({
        "cik": [1, 2, 3, 4, 5],
        "ticker": ["AAA", "BBB", "CCC", "DDD", "EEE"],
        "gics_sector": ["Tech", "Tech", "Financials", "Financials", "Financials"],
    })
    result = fb.build_qmj_panel(_panel(), universe=universe,
                                 config=fb.QMJConfig(sector_relative=True, min_group_size=3))
    assert "gics_sector" in result.columns
    assert "ticker" in result.columns


def test_stack_periods():
    p1 = _panel(period="P1")
    p2 = _panel(period="P2")
    stacked = fb.stack_periods([p1, p2])
    assert set(stacked["period"].unique()) == {"P1", "P2"}
    assert len(stacked) == 10
