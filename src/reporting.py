from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def save_price_volume_plot(data: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    data["close"].plot(ax=axes[0], color="#1f77b4", linewidth=0.8)
    axes[0].set_title("HI1 5-minute Close Price")
    axes[0].set_ylabel("Price")
    data["volume"].plot(ax=axes[1], color="#5f6f52", linewidth=0.6)
    axes[1].set_title("HI1 5-minute Volume")
    axes[1].set_ylabel("Volume")
    fig.tight_layout()
    fig.savefig(output_dir / "figure1_price_volume.png", dpi=160)
    plt.close(fig)


def save_return_distribution_plot(data: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    data["return"].clip(-0.02, 0.02).hist(ax=ax, bins=120, color="#4c78a8")
    ax.set_title("5-minute Return Distribution")
    ax.set_xlabel("Return")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig(output_dir / "figure2_return_distribution.png", dpi=160)
    plt.close(fig)


def save_equity_plot(results: dict[str, pd.DataFrame], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, result in results.items():
        result["equity"].plot(ax=ax, linewidth=0.9, label=name)
    ax.set_title("Out-of-sample Equity Curves")
    ax.set_ylabel("Equity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "figure3_oos_equity_curves.png", dpi=160)
    plt.close(fig)


def save_drawdown_plot(results: dict[str, pd.DataFrame], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    for name, result in results.items():
        equity = result["equity"]
        drawdown = equity / equity.cummax() - 1
        drawdown.plot(ax=ax, linewidth=0.9, label=name)
    ax.set_title("Out-of-sample Drawdown")
    ax.set_ylabel("Drawdown")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "figure4_oos_drawdown.png", dpi=160)
    plt.close(fig)


def save_sma_parameter_heatmap(sma_optimization: pd.DataFrame, output_dir: Path) -> None:
    heatmap = sma_optimization.pivot(
        index="short_window",
        columns="long_window",
        values="sharpe",
    ).sort_index(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    image = ax.imshow(heatmap.values, aspect="auto", cmap="RdYlGn")
    ax.set_title("SMA Parameter Search: In-sample Sharpe Ratio")
    ax.set_xlabel("Long Window")
    ax.set_ylabel("Short Window")
    ax.set_xticks(range(len(heatmap.columns)))
    ax.set_xticklabels(heatmap.columns)
    ax.set_yticks(range(len(heatmap.index)))
    ax.set_yticklabels(heatmap.index)

    for row_index, short_window in enumerate(heatmap.index):
        for col_index, long_window in enumerate(heatmap.columns):
            value = heatmap.loc[short_window, long_window]
            if pd.notna(value):
                ax.text(
                    col_index,
                    row_index,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black",
                )

    fig.colorbar(image, ax=ax, label="Sharpe Ratio")
    fig.tight_layout()
    fig.savefig(output_dir / "figure5_sma_parameter_heatmap.png", dpi=160)
    plt.close(fig)


def save_report_plots(
    bars: pd.DataFrame,
    oos_results: dict[str, pd.DataFrame],
    sma_optimization: pd.DataFrame,
    output_dir: Path,
) -> None:
    save_price_volume_plot(bars, output_dir)
    save_return_distribution_plot(bars, output_dir)
    save_equity_plot(oos_results, output_dir)
    save_drawdown_plot(oos_results, output_dir)
    save_sma_parameter_heatmap(sma_optimization, output_dir)
