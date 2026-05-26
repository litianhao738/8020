from __future__ import annotations

import pandas as pd


def summarize_performance(result: pd.DataFrame) -> dict[str, float]:
    equity = result["equity"]
    daily_equity = equity.resample("D").last().dropna()
    daily_returns = daily_equity.pct_change().dropna()

    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    annualized_return = 0.0
    if len(daily_returns) > 0:
        annualized_return = (1 + total_return) ** (252 / len(daily_returns)) - 1

    running_max = equity.cummax()
    drawdown = equity / running_max - 1

    annualized_volatility = daily_returns.std(ddof=0) * (252**0.5)
    sharpe = 0.0
    if annualized_volatility > 0:
        sharpe = annualized_return / annualized_volatility

    max_drawdown = float(drawdown.min())
    calmar = 0.0
    if max_drawdown < 0:
        calmar = annualized_return / abs(max_drawdown)

    trades = int((result["turnover"] > 0).sum())
    win_rate = float((result.loc[result["net_pnl"] != 0, "net_pnl"] > 0).mean())
    if pd.isna(win_rate):
        win_rate = 0.0

    return {
        "total_return": float(total_return),
        "annualized_return": float(annualized_return),
        "annualized_volatility": float(annualized_volatility),
        "sharpe": float(sharpe),
        "max_drawdown": max_drawdown,
        "calmar": float(calmar),
        "final_equity": float(equity.iloc[-1]),
        "trades": trades,
        "win_rate": win_rate,
        "total_cost": float(result["trading_cost"].sum()),
        "max_notional_exposure": float(result["notional_exposure"].max()),
        "max_margin_required": float(result["margin_required"].max()),
        "max_exposure_to_capital": float(result["exposure_to_capital"].max()),
    }


def format_summary(summary: dict[str, float]) -> str:
    lines = []
    for key, value in summary.items():
        if key in {"trades"}:
            lines.append(f"{key}: {int(value)}")
        elif key in {
            "total_return",
            "annualized_return",
            "annualized_volatility",
            "max_drawdown",
            "win_rate",
        }:
            lines.append(f"{key}: {value:.2%}")
        else:
            lines.append(f"{key}: {value:.4f}")
    return "\n".join(lines)
