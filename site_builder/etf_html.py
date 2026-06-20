"""Bare HTML generation for the ETF static site."""

from __future__ import annotations

import html
from pathlib import Path

from site_builder.etf_data import (
    AllocationRow,
    DrawdownSnapshot,
    PeriodReturnRow,
    SummaryStats,
)
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


def _summary_panel(label: str, value: str, *, tone: str = "neutral", detail: str = "") -> str:
    detail_html = f"<div class=\"panel-detail\">{detail}</div>" if detail else ""
    return (
        f"<section class=\"panel {html.escape(tone)}\">"
        f"<div class=\"panel-label\">{html.escape(label)}</div>"
        f"<div class=\"panel-value\">{value}</div>"
        f"{detail_html}"
        "</section>"
    )


def build_index_html(
    *,
    output: Path,
    universe: Universe,
    generated_at: str,
    tracking_start: str,
    as_of_date: str,
    strat_stats: SummaryStats,
    drawdown: DrawdownSnapshot,
    period_returns: list[PeriodReturnRow],
    allocations: list[AllocationRow],
    invested_weight: float,
    cash_weight: float,
    sharpe_1y: float | None,
    portfolio_url: str | None = None,
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
        ".portfolio-link { margin: 0 0 12px; font-size: 1rem; }",
        "</style>",
        "</head>",
        "<body>",
        "<h1>ETF Engine</h1>",
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
            "<p><img class=\"chart\" src=\"drawdown.png\" alt=\"Drawdown\"></p>",
            "<p><img class=\"chart\" src=\"weights.png\" alt=\"Portfolio weights\"></p>",
            "<p><img class=\"chart\" src=\"invested.png\" alt=\"Invested weight\"></p>",
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
            "<h2>Target weights</h2>",
            "<table border=\"1\" cellpadding=\"4\" cellspacing=\"0\">",
            "<tr><th>ETF</th><th>Weight</th><th>Return (1y)</th></tr>",
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
            lines.append(
                "<tr>"
                f"<td>{html.escape(row.label)}</td>"
                f"<td>{row.weight_pct:.2%}</td>"
                f"<td>{return_label}{spark_html}</td>"
                "</tr>"
            )
    else:
        lines.append("<tr><td colspan=\"3\">No holdings.</td></tr>")
    lines.extend(
        [
            "</table>",
            "<p><a href=\"/builds/\">Previous builds</a></p>",
            "<h2>Methodology</h2>",
            "<p>Monthly walk-forward backtest on the InvestEngine ISA ETF universe. "
            "Each rebalance maximises mean annualised return over a 12-month lookback, "
            "subject to a 25% annualised volatility cap. Weights are EWMA-smoothed (span 6), "
            "floored at 5%, vol-scaled, then simulated weekly with a 5% drift band and "
            "bid–ask spread drag. Uninvested cash earns the US fed funds rate.</p>",
            "<p>Benchmark: VWRP (FTSE All-World accumulating). Strategy and benchmark "
            "curves are rebased to equity = 1.0 at the tracking start date. "
            "There is no live account integration — this is an ongoing backtest.</p>",
            "</body>",
            "</html>",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
