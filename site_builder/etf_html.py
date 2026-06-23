"""Bare HTML generation for the ETF static site."""

from __future__ import annotations

import html
from pathlib import Path

from site_builder.etf_data import (
    AllocationRow,
    AthSnapshot,
    DrawdownSnapshot,
    PeriodReturnRow,
    SummaryStats,
)
from site_builder.investengine_portfolio import InvestEngineSnapshot
from site_builder.metrics import BenchmarkRegressionStats
from strategy.data import Universe


def _pct(value: float, *, signed: bool = True) -> str:
    if value != value:
        return "n/a"
    return f"{value:+.2%}" if signed else f"{value:.2%}"


def _sign_colour(value: float) -> str | None:
    if value > 0:
        return "green"
    if value < 0:
        return "red"
    return None


def _coloured_span(text: str, value: float) -> str:
    colour = _sign_colour(value)
    if colour is None:
        return html.escape(text)
    return f'<span style="color: {colour}">{html.escape(text)}</span>'


def _coloured_pct(value: float) -> str:
    if value != value:
        return "n/a"
    return _coloured_span(_pct(value), value)


def _plain_number(value: float | None, *, decimals: int = 2) -> str:
    if value is None or value != value:
        return "n/a"
    return f"{value:.{decimals}f}"


def _backtest_percentile(value: float | None) -> str:
    if value is None or value != value:
        return "No backtest comparison"
    better_than = max(0.0, min(100.0, value))
    return f"Better than {better_than:.0f}% of backtest"


def _dd(value: float | None) -> str:
    if value is None or value != value:
        return "n/a"
    return f"{value:.2%}"


def _backtest_dd_fraction(value: float | None) -> str:
    if value is None or value != value:
        return "n/a"
    same_or_worse = max(0.0, min(100.0, value))
    return f"Better than {same_or_worse:.0f}% of backtest"


def _panel_tone(value: float | None) -> str:
    if value is None or value != value:
        return "neutral"
    if value > 0:
        return "green"
    if value < 0:
        return "red"
    return "neutral"


def _regime_panel_tone(
    regime_rows: list[tuple[int, float | None, str]] | None,
) -> str:
    if not regime_rows:
        return "neutral"
    return "red" if all(label == "bearish" for _, _, label in regime_rows) else "green"


def _regime_summary_html(regime_rows: list[tuple[int, float | None, str]] | None) -> str:
    if regime_rows is None:
        return "Not enough history"
    bits = []
    for months, trailing, label in regime_rows:
        tone = _panel_tone(trailing)
        bits.append(
            "<span class=\"mini-stat\">"
            f"{months}m <strong class=\"stat-pct {tone}\">"
            f"{html.escape(_pct(trailing if trailing is not None else float('nan')))}</strong> "
            f"{html.escape(label)}"
            "</span>"
        )
    return " ".join(bits)


def _summary_panel(label: str, value: str, *, tone: str = "neutral", detail: str = "") -> str:
    detail_html = f"<div class=\"panel-detail\">{detail}</div>" if detail else ""
    return (
        f"<section class=\"panel {html.escape(tone)}\">"
        f"<div class=\"panel-label\">{html.escape(label)}</div>"
        f"<div class=\"panel-value\">{value}</div>"
        f"{detail_html}"
        "</section>"
    )


def _icon_cell(icon_path: str, alt: str) -> str:
    if not icon_path:
        return ""
    return (
        f'<img class="holding-icon" src="{html.escape(icon_path)}" '
        f'alt="{html.escape(alt)}" width="24" height="24"> '
    )


def _source_badge(source: str) -> str:
    return f'<span class="source-badge {html.escape(source)}">{html.escape(source)}</span>'


def _weight_cell(value: float | None) -> str:
    if value is None:
        return "—"
    if value != value:
        return "n/a"
    return f"{value:.2%}"


def _backtest_weight_cell(value: float | None, change_1y: float | None) -> str:
    base = _weight_cell(value)
    if change_1y is None or change_1y != change_1y:
        return base
    pp_text = f"({change_1y * 100:+.2f}pp since 1y ago)"
    return f"{base} {_coloured_span(pp_text, change_1y)}"


def build_index_html(
    *,
    output: Path,
    universe: Universe,
    generated_at: str,
    tracking_start: str,
    as_of_date: str,
    strat_stats: SummaryStats,
    bench_regression: BenchmarkRegressionStats,
    bench_label: str,
    drawdown: DrawdownSnapshot,
    ath: AthSnapshot,
    period_returns: list[PeriodReturnRow],
    allocations: list[AllocationRow],
    invested_weight: float,
    cash_weight: float,
    sharpe_1y: float | None,
    portfolio_url: str | None = None,
    ie_snapshot: InvestEngineSnapshot | None = None,
    regime_rows: list[tuple[int, float | None, str]] | None = None,
) -> None:
    lines: list[str] = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "<meta charset=\"utf-8\">",
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">",
        "<title>ETF Engine</title>",
        "<style>",
        "* { box-sizing: border-box; }",
        "img { max-width: 100%; height: auto; }",
        ".chart { display: block; max-width: 100%; height: auto; }",
        ".sparkline { max-width: min(180px, 45vw); height: auto; vertical-align: middle; }",
        ".summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px; margin: 16px 0 24px; }",
        ".panel { border: 1px solid #d7d7d7; border-radius: 6px; padding: 12px; background: #f8f8f8; }",
        ".panel.green { border-color: #8bc49a; background: #eef8f0; }",
        ".panel.red { border-color: #dd9a9a; background: #fff0f0; }",
        ".panel-label { color: #555; font-size: 0.86rem; margin-bottom: 6px; }",
        ".panel-value { font-size: 1.35rem; font-weight: 700; line-height: 1.2; }",
        ".panel-detail { color: #444; font-size: 0.86rem; line-height: 1.45; margin-top: 8px; }",
        ".generated { color: #666; font-size: 0.86rem; margin-top: -6px; }",
        ".mini-stat { display: block; font-size: 0.9rem; font-weight: 400; }",
        ".stat-pct.green { color: green; font-weight: 700; }",
        ".stat-pct.red { color: red; font-weight: 700; }",
        ".stat-pct.neutral { font-weight: 700; }",
        ".portfolio-link { margin: 0 0 12px; font-size: 1rem; }",
        ".source-note { color: #444; font-size: 0.92rem; line-height: 1.5; max-width: 72rem; }",
        ".source-badge { display: inline-block; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.03em; text-transform: uppercase; border-radius: 4px; padding: 1px 6px; margin-right: 4px; vertical-align: middle; }",
        ".source-badge.backtest { color: #1f4f82; background: #e8f1fb; border: 1px solid #b8d4f0; }",
        ".source-badge.investengine { color: #5a3d12; background: #fff4df; border: 1px solid #f0d7a8; }",
        ".holding-icon { width: 24px; height: 24px; object-fit: contain; vertical-align: middle; margin-right: 6px; }",
        ".region-swatch { display: inline-block; width: 12px; height: 12px; border-radius: 2px; margin-right: 6px; vertical-align: middle; }",
        "h3.source-heading { margin: 1.5rem 0 0.5rem; font-size: 1.05rem; }",
        "</style>",
        "</head>",
        "<body>",
        "<h1>ETF Engine</h1>",
        "<a href=\"/\">Sharpening</a>",
    ]
    if portfolio_url:
        lines.append(
            f"<p class=\"portfolio-link\">"
            f"<a href=\"{html.escape(portfolio_url)}\">InvestEngine portfolio</a>"
            f"</p>"
        )
    lines.extend(
        [
        f"<p class=\"generated\">Generated {html.escape(generated_at)} UTC · "
        f"as of {html.escape(as_of_date)} · tracking from {html.escape(tracking_start)}</p>",
        "<div class=\"summary-grid\">",
        _summary_panel(
            "Regime votes",
            _regime_summary_html(regime_rows),
            tone=_regime_panel_tone(regime_rows),
        ),
        _summary_panel(
            "Strategy CAGR (full backtest)",
            _coloured_pct(strat_stats.cagr),
            tone=_panel_tone(strat_stats.cagr),
        ),
        _summary_panel(
            "Strategy Sharpe (full backtest)",
            html.escape(_plain_number(strat_stats.sharpe)),
            tone="neutral",
        ),
        _summary_panel(
            "Strategy vol (ann.)",
            html.escape(_pct(strat_stats.vol_ann, signed=False)),
            tone="neutral",
        ),
        _summary_panel(
            f"Beta vs {bench_label}",
            html.escape(_plain_number(bench_regression.beta, decimals=3)),
            tone="neutral",
        ),
        _summary_panel(
            f"Alpha vs {bench_label} (ann.)",
            _coloured_pct(bench_regression.alpha_ann),
            tone=_panel_tone(bench_regression.alpha_ann),
        ),
        _summary_panel(
            f"Residual vol vs {bench_label} (ann.)",
            html.escape(_pct(bench_regression.residual_vol_ann, signed=False)),
            tone="neutral",
        ),
        _summary_panel(
            "Sharpe (1y trailing)",
            html.escape(_plain_number(sharpe_1y)),
            tone="neutral",
        ),
        _summary_panel(
            "Current drawdown",
            _coloured_pct(drawdown.drawdown_pct if drawdown.drawdown_pct is not None else float("nan")),
            tone=_panel_tone(drawdown.drawdown_pct),
            detail=html.escape(_backtest_dd_fraction(drawdown.backtest_time_fraction_pct)),
        ),
        _summary_panel(
            "Days since ATH",
            html.escape(
                str(ath.days_since_ath) if ath.days_since_ath is not None else "n/a"
            ),
            tone="neutral",
            detail=html.escape(_backtest_percentile(ath.backtest_time_fraction_pct)),
        ),
        _summary_panel(
            "Invested weight",
            html.escape(_pct(invested_weight, signed=False)),
            detail=html.escape(f"Cash {_pct(cash_weight, signed=False)}"),
        ),
        ]
    )
    for row in period_returns:
        lines.append(
            _summary_panel(
                row.label,
                _coloured_pct(row.return_pct),
                tone=_panel_tone(row.return_pct),
                detail=html.escape(_backtest_percentile(row.percentile)),
            )
        )
    lines.extend(
        [
            "</div>",
            "<p><img class=\"chart\" src=\"equity.png\" alt=\"Equity curve\"></p>",
            "<p><img class=\"chart\" src=\"equity_tracking.png\" alt=\"Equity since tracking start\"></p>",
            "<p><img class=\"chart\" src=\"equity_vol_caps.png\" alt=\"Equity by vol cap\"></p>",
            "<p><img class=\"chart\" src=\"drawdown.png\" alt=\"Drawdown\"></p>",
            "<p><img class=\"chart\" src=\"weights.png\" alt=\"Portfolio weights\"></p>",
            "<p><img class=\"chart\" src=\"invested.png\" alt=\"Invested weight\"></p>",
            "<p><img class=\"chart\" src=\"regime_returns.png\" alt=\"Regime trailing returns\"></p>",
            "<p><img class=\"chart\" src=\"sharpe.png\" alt=\"Sharpe ratio chart\"></p>",
            "<p><img class=\"chart\" src=\"cagr.png\" alt=\"CAGR chart\"></p>",
            "<p><img class=\"chart\" src=\"vol.png\" alt=\"Volatility chart\"></p>",
            "<h2>Returns</h2>",
            "<table border=\"1\" cellpadding=\"4\" cellspacing=\"0\">",
            "<tr><th>Period</th><th>Return</th><th>Backtest percentile</th></tr>",
        ]
    )
    for row in period_returns:
        lines.append(
            "<tr>"
            f"<td>{html.escape(row.label)}</td>"
            f"<td>{_coloured_pct(row.return_pct)}</td>"
            f"<td>{html.escape(_backtest_percentile(row.percentile))}</td>"
            "</tr>"
        )
    lines.extend(
        [
            "</table>",
            "<h2>Weekly return distribution</h2>",
            "<p><img class=\"chart\" src=\"weekly_returns_hist.png\" alt=\"Weekly return histogram\"></p>",
            "<h2>Drawdown distribution</h2>",
            "<p><img class=\"chart\" src=\"drawdown_dist.png\" alt=\"Drawdown distribution\"></p>",
            "<h2>Days since ATH distribution</h2>",
            "<p><img class=\"chart\" src=\"ath_dist.png\" alt=\"Days since ATH distribution\"></p>",
            "<h2>Portfolio weights</h2>",
            "<p class=\"source-note\">"
            f"{_source_badge('backtest')} Simulated effective weights from the monthly walk-forward model "
            f"(as of {html.escape(as_of_date)}). "
            f"{_source_badge('investengine')} Live ETF weights from the InvestEngine shared portfolio API"
            + (
                f" (fetched {html.escape(ie_snapshot.fetched_date)})."
                if ie_snapshot
                else " (not available for this build)."
            )
            + " Return sparklines use Yahoo prices and are backtest context only."
            "</p>",
            "<table border=\"1\" cellpadding=\"4\" cellspacing=\"0\">",
            "<tr><th>ETF</th><th>Backtest weight</th><th>InvestEngine weight</th><th>Return (1y)</th></tr>",
        ]
    )
    if allocations:
        for row in allocations:
            return_label = (
                _coloured_pct(row.return_1y) if row.return_1y is not None else "n/a"
            )
            spark_html = (
                f' <img class="sparkline" src="{html.escape(row.spark_path)}" alt="">'
                if row.spark_path
                else ""
            )
            icon_html = _icon_cell(row.icon_path, row.label)
            lines.append(
                "<tr>"
                f"<td>{icon_html}{html.escape(row.label)}</td>"
                f"<td>{_backtest_weight_cell(row.weight_pct if row.weight_pct > 1e-6 else None, row.weight_change_1y)}</td>"
                f"<td>{_weight_cell(row.ie_weight_pct)}</td>"
                f"<td>{return_label}{spark_html}</td>"
                "</tr>"
            )
    else:
        lines.append("<tr><td colspan=\"4\">No holdings.</td></tr>")
    lines.append("</table>")

    if ie_snapshot and ie_snapshot.equity_holdings:
        lines.extend(
            [
                "<h3 class=\"source-heading\">Top 20 look-through equities "
                f"{_source_badge('investengine')}</h3>",
                "<p class=\"source-note\">Underlying stock exposures reported by InvestEngine "
                "(look-through from ETF holdings). Not from the backtest model.</p>",
                "<table border=\"1\" cellpadding=\"4\" cellspacing=\"0\">",
                "<tr><th>Equity</th><th>Look-through weight</th></tr>",
            ]
        )
        for row in ie_snapshot.equity_holdings:
            icon_html = _icon_cell(row.icon_path, row.name)
            lines.append(
                "<tr>"
                f"<td>{icon_html}{html.escape(row.name)}</td>"
                f"<td>{_weight_cell(row.weight_pct)}</td>"
                "</tr>"
            )
        lines.append("</table>")

    if ie_snapshot and ie_snapshot.region_breakdown:
        lines.extend(
            [
                "<h3 class=\"source-heading\">Regional breakdown "
                f"{_source_badge('investengine')}</h3>",
                "<p class=\"source-note\">InvestEngine regional look-through exposure "
                "(closest available geographic split in the shared portfolio API; not country-level).</p>",
                "<table border=\"1\" cellpadding=\"4\" cellspacing=\"0\">",
                "<tr><th>Region</th><th>Look-through weight</th></tr>",
            ]
        )
        for row in ie_snapshot.region_breakdown:
            swatch = (
                f'<span class="region-swatch" style="background: {html.escape(row.color)}"></span>'
            )
            lines.append(
                "<tr>"
                f"<td>{swatch}{html.escape(row.name)}</td>"
                f"<td>{_weight_cell(row.weight_pct)}</td>"
                "</tr>"
            )
        lines.append("</table>")

    cash_methodology = "Uninvested cash earns 0% (InvestEngine ISA cash balance)."

    lines.extend(
        [
            "<p><a href=\"/builds/\">Previous builds</a></p>",
            "<h2>Methodology</h2>",
            "<p>Monthly walk-forward backtest on the InvestEngine ISA ETF universe. "
            "Each rebalance maximises mean annualised return over a 12-month lookback, "
            "subject to a 25% annualised volatility cap. Weights are EWMA-smoothed (span 6), "
            "floored at 5%, vol-scaled, then simulated weekly with a 5% drift band and "
            f"bid–ask spread drag. The strategy sits in cash only when the always-invested "
            "shadow book lost money over all three regime windows: 3/6/12 months. "
            f"{cash_methodology}</p>",
            "<p>Benchmark: VWRP (FTSE All-World accumulating). Strategy and benchmark "
            "curves are rebased to equity = 1.0 at the tracking start date. "
            f"Alpha, beta, and residual vol are from an OLS regression of weekly strategy "
            f"returns on {html.escape(bench_label)} over the full backtest. "
            "Charts, summary stats, and backtest weight columns are simulated only. "
            "InvestEngine tables and weight columns come from the live shared portfolio API "
            "and are labelled accordingly.</p>",
            "</body>",
            "</html>",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
