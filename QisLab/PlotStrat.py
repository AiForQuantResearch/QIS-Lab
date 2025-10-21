import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import warnings
warnings.filterwarnings('ignore')

# =========================
# Helpers Benchmarks
# =========================
def _download_benchmarks_base100(tickers, start_date, end_date):
    """
    Télécharge les benchmarks (Close), normalise base 100 au premier point,
    aligne sur jours ouvrés, forward-fill. Retourne un DataFrame (index = dates, cols = tickers).
    """
    if not tickers:
        return pd.DataFrame()

    data_dict = {}
    for t in tickers:
        try:
            s = yf.download(t, start=start_date, end=end_date, progress=False)["Close"].dropna()
            if s.empty:
                continue
            s = s / s.iloc[0] * 100.0
            data_dict[t] = s
        except Exception as e:
            print(f"⚠️ Benchmark download failed for {t}: {e}")

    if not data_dict:
        return pd.DataFrame()

    idx = pd.date_range(start=min(s.index.min() for s in data_dict.values()),
                        end=end_date, freq="B")

    df = pd.DataFrame(index=idx)
    for t, s in data_dict.items():
        df[t] = s.reindex(idx).ffill()

    return df.dropna(how="all")


# =========================
# Plot multi (plusieurs courbes de stratégies) + multi benchmarks
# =========================
def plot_sim_live_multi(
    equity_sim_dict: dict,
    equity_live_dict: dict,
    title="Graph",
    signal_window: int = None,
    log_scale: bool = True,
    benchmark_tickers: tuple | list = ("GLD", "SPY"),
    benchmark_labels: tuple | list = None,
    benchmark_colors: tuple | list = ("gold", "#2E86C1")
):
    """Plot multi equity curves + jusqu’à 2 benchmarks."""
    fig, ax = plt.subplots(figsize=(18,8))
    min_sim_start, max_live_end = None, None

    # === Couleurs des index ===
    color_aiqr_dark   = "#0A2342"  # bleu nuit profond
    color_aiqr_light  = "#3A6EA5"  # bleu doux
    color_inv_dark    = "#4A6FA5"  # bleu/gris moyen
    color_inv_light   = "#89A7D0"  # bleu clair

    # Déterminer la fenêtre globale pour les benchmarks
    for w in equity_sim_dict.keys():
        sim = equity_sim_dict[w]
        live = equity_live_dict[w]

        if signal_window is not None:
            sim = sim.iloc[signal_window:]
        if len(sim) == 0:
            raise ValueError(f"After trimming, sim series for weight {w} is empty.")

        start_idx = sim.index.min()
        live = live.loc[live.index >= start_idx]

        min_sim_start = start_idx if (min_sim_start is None or start_idx < min_sim_start) else min_sim_start
        if not live.empty:
            end_live = live.index.max()
            max_live_end = end_live if (max_live_end is None or end_live > max_live_end) else max_live_end

    # Tracer chaque paire sim/live
    for w in equity_sim_dict.keys():
        sim = equity_sim_dict[w]
        live = equity_live_dict[w]
        if signal_window is not None:
            sim = sim.iloc[signal_window:]
        start_idx = sim.index.min()
        live = live.loc[live.index >= start_idx]

        split_date = live.index[0]
        equity_all = pd.concat([sim, live])
        equity_all = equity_all[~equity_all.index.duplicated(keep="last")].sort_index()
        pre, post = equity_all[equity_all.index < split_date], equity_all[equity_all.index >= split_date]

        if w == 1:
            col_oos, col_is = color_inv_light, color_inv_dark
            label_base = "Inverse Volatility Index"
        else:
            col_oos, col_is = color_aiqr_light, color_aiqr_dark
            label_base = "Ai for Quant Research Systematic Index"

        ax.plot(pre.index,  pre.values,  color=col_oos, linewidth=1.9, label=f"{label_base} – Out of Sample")
        ax.plot(post.index, post.values, color=col_is,  linewidth=1.9, label=f"{label_base} – In Sample")

        ax.axvline(split_date, color='gray', linewidth=1, linestyle="--")

    # === Benchmarks (ligne continue) ===
    if min_sim_start and max_live_end and benchmark_tickers:
        bench_df = _download_benchmarks_base100(list(benchmark_tickers), min_sim_start, max_live_end)
        if not bench_df.empty:
            if benchmark_labels is None:
                benchmark_labels = list(bench_df.columns)
            if benchmark_colors is None:
                benchmark_colors = ("gold", "#2E86C1")

            for i, col in enumerate(bench_df.columns):
                color = benchmark_colors[i % len(benchmark_colors)]
                label = f"{benchmark_labels[i]} ({col})" if (i < len(benchmark_labels)) else col
                ax.plot(bench_df.index, bench_df[col].values, linewidth=2.0, color=color, label=label)  # <— ligne continue

    # === Style général ===
    ax.set_title(title, loc='left', fontsize=16, weight='bold')
    ax.set_xlabel("")
    ax.set_ylabel("Index (base 100)")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    plt.tight_layout()

    if log_scale:
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.get_major_formatter().set_scientific(False)
        ax.yaxis.get_major_formatter().set_useOffset(False)

    plt.savefig("Gold_index_multi.png", dpi=200, bbox_inches="tight")
    plt.show()


# =========================
# Plot single (une stratégie) + multi benchmarks
# =========================
def plot_sim_live(
    equity_sim: pd.Series,
    equity_live: pd.Series,
    title: str = "Graph",
    signal_window: int = None,
    is_aiqr: bool = True,
    benchmark_tickers: tuple | list = ("GLD", "SPY"),
    benchmark_labels: tuple | list = None,
    benchmark_colors: tuple | list = ("gold", "#D35400")
):
    """Trace une courbe unique avec séparation OOS / IS + benchmarks (en ligne pleine)."""
    color_aiqr_dark   = "#0A2342"
    color_aiqr_light  = "#3A6EA5"
    color_inv_dark    = "#4A6FA5"
    color_inv_light   = "#89A7D0"

    col_oos, col_is = (color_aiqr_light, color_aiqr_dark) if is_aiqr else (color_inv_light, color_inv_dark)
    label_base = "Ai for Quant Research Systematic Index" if is_aiqr else "Inverse Volatility Index"

    if signal_window is not None:
        equity_sim = equity_sim.iloc[signal_window:]
    if len(equity_sim) == 0:
        raise ValueError("After trimming, equity_sim is empty. Check signal_window.")

    start_idx = equity_sim.index.min()
    end_idx   = equity_live.index.max()
    equity_live = equity_live.loc[equity_live.index >= start_idx]

    split_date = equity_live.index[0]
    equity_all = pd.concat([equity_sim, equity_live]).drop_duplicates().sort_index()
    pre  = equity_all[equity_all.index <  split_date]
    post = equity_all[equity_all.index >= split_date]

    fig, ax = plt.subplots(figsize=(18,8))
    ax.plot(pre.index,  pre.values,  color=col_oos, linewidth=2.0, label=f"{label_base} – Out of Sample")
    ax.plot(post.index, post.values, color=col_is,  linewidth=2.2, label=f"{label_base} – In Sample")
    ax.axvline(split_date, color='gray', linewidth=1, linestyle="--", label="Début live")

    # === Benchmarks (ligne pleine) ===
    if benchmark_tickers:
        bench_df = _download_benchmarks_base100(list(benchmark_tickers), start_idx, end_idx)
        if not bench_df.empty:
            if benchmark_labels is None:
                benchmark_labels = list(bench_df.columns)
            if benchmark_colors is None:
                benchmark_colors = ("gold", "#2E86C1")

            for i, col in enumerate(bench_df.columns):
                color = benchmark_colors[i % len(benchmark_colors)]
                label = f"{benchmark_labels[i]} ({col})" if (i < len(benchmark_labels)) else col
                s = bench_df[col].reindex(equity_all.index).ffill().dropna()
                ax.plot(s.index, s.values, linewidth=2.0, color=color, label=label)  # <— ligne continue

    ax.set_title(title, loc='left', fontsize=16, weight='bold')
    ax.set_xlabel("")
    ax.set_ylabel("Index (base 100)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()

    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.get_major_formatter().set_scientific(False)
    ax.yaxis.get_major_formatter().set_useOffset(False)

    plt.savefig("Gold_index_single.png", dpi=200, bbox_inches="tight")
    plt.show()
