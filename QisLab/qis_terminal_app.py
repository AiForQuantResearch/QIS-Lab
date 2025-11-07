import sys
import importlib

# --- imports sûrs après validation ---
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

import plotly.graph_objects as go
import plotly.express as px
import os, pickle, datetime
from pathlib import Path
import pickle
from pathlib import Path

# =========================
# Page config
# =========================
st.set_page_config(layout="wide", page_title="QIS Lab — AIQR Indices")

# =========================
# Sidebar -> Logo & Description
# =========================
def show_logo(name: str):
    try:
        p = _local_path(name)
        st.image(str(p), use_column_width=True)
        return True
    except Exception:
        return False

LOGO_PATH = Path(__file__).parent / "Logo.JPG"   # chemin absolu FIABLE

with st.sidebar:
    #st.caption(f"Logo path: {LOGO_PATH}")
    st.image(str(LOGO_PATH), use_column_width=True)
    st.markdown("## Ai for Quant Research")
    st.markdown(
        """
        An independent institute dedicated to advancing research in Quantitative Investment Strategies
        through artificial intelligence and systematic approaches.
        """
    )

# =========================
# Header
# =========================
st.markdown("""<h1 style='background-color:#00C29A; color:white; padding:10px; margin:0;text-align:center;border-radius:12px;'>Ai For Quant Research QIS Lab</h1>""",unsafe_allow_html=True)

# =========================
# Tabs
# =========================
tabs = st.tabs(["Systematic Gold Index", "Precious Metal Index", "Crypto Factor", "Trend Following", "Systematic Index Trading", "Optimized Portfolio"])

# =========================
# Helpers
# =========================

@st.cache_data(show_spinner=False)
def get_close(tickers, start, end):
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    data = data.ffill().dropna(how="all").sort_index()
    if isinstance(data, pd.Series):
        data = data.to_frame()
    return data

def compute_asset_metrics(asset_ret: pd.Series) -> pd.Series:
    r = asset_ret.dropna()
    if r.empty:
        return pd.Series({
            "Annual Return": np.nan,
            "Annual Volatility": np.nan,
            "Sharpe Ratio": np.nan,
            "Max Drawdown": np.nan,
            "1-Day 95% VaR": np.nan
        })
    ann_ret = r.mean() * 252
    ann_vol = r.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol != 0 else np.nan
    nav = (1 + r).cumprod()
    max_dd = (nav.cummax() - nav).max()
    var_95 = np.percentile(r, 5)
    return pd.Series({
        "Annual Return": ann_ret,
        "Annual Volatility": ann_vol,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_dd,
        "1-Day 95% VaR": var_95
    })

def ensure_datetime_index(x) -> pd.Series:
    s = pd.Series(x)
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index, errors="coerce")
    s = s.dropna()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s

def _get_from_any(d: dict, *keys):
    for k in keys:
        if k in d:
            return d[k]
    return None

def fig_rolling_corr_strategies(ret_df: pd.DataFrame, pairs: list[tuple[str,str]],
                                win: int, title: str) -> go.Figure:
    """Graphe corrélation roulante entre stratégies, avec légende à droite (hors tracé)."""
    fig = go.Figure()
    for a, b in pairs:
        cs = ret_df[a].rolling(win).corr(ret_df[b]).dropna()
        if not cs.empty:
            fig.add_trace(go.Scatter(x=cs.index, y=cs.values, mode="lines", name=f"{a} vs {b}"))
    # Ligne y=0
    fig.add_hline(y=0.0, line_dash="dot", line_color="gray", line_width=1)
    # Légende à droite, verticale, hors canvas -> rétrécir le domaine x et augmenter la marge droite
    fig.update_layout(
        title=title,
        height=420,
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis_showgrid=False, yaxis_showgrid=False,
        margin=dict(l=10, r=230, t=60, b=10),
        legend=dict(
            orientation="v",
            yanchor="top", y=1.0,
            xanchor="left", x=1.02,   # à l’extérieur, côté droit
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1,
            title="Pairs"
        ),
    )
    # Compacter le plot pour laisser la place à la légende
    fig.update_xaxes(domain=[0.0, 0.80])
    return fig

def _local_path(fname: str) -> Path:
    # dossier où se trouve *ce* script
    here = Path(__file__).parent
    # essais robustes : dossier du script d'abord, puis cwd (au cas où)
    candidates = [here / fname, Path.cwd() / fname]
    for p in candidates:
        if p.exists():
            return p
    # message utile en cas d'échec
    raise FileNotFoundError(f"Introuvable: {fname} dans {candidates}")

def load_pickle_local(fname: str):
    p = _local_path(fname)
    with p.open("rb") as f:
        return pickle.load(f)



# =============================
# TAB 0 — Systematic Gold Index
# =============================
with tabs[0]:

    # ---------- Description ----------
    st.markdown("""
        <div style='
            border:1px solid #00C29A;
            padding:15px;
            border-radius:12px;
            margin-top:10px;
            background-color:#F8FFFC;
            text-align:justify;
            font-size:15px;
            line-height:1.6;
        '>
        <b>The AIQR Systematic Gold Index</b> aims to deliver superior risk-adjusted exposure to gold through a systematic and risk-managed approach. 
        The index combines <b>Gold</b> and <b>Bitcoin (BTC)</b> as complementary assets: gold provides defensive and stable characteristics, while BTC captures market momentum and digital risk-premium dynamics.
        By dynamically allocating exposure between these two assets within a volatility-controlled framework, the index seeks to enhance long-term returns while limiting downside risk. 
        This cross-asset construction allows the strategy to benefit from both the store-of-value profile of gold and the asymmetric growth potential of BTC.
        The index is long-only and rebalanced weekly. It is designed to outperform traditional benchmarks such as <b>GLD</b>.
        </div>
    """, unsafe_allow_html=True)

    # ---------- Hidden params for BTC/GLD extras ----------
    selected_tickers = ["BTC-USD", "GLD"]
    start_date = "2016-01-01"
    end_date = pd.to_datetime("today").strftime("%Y-%m-%d")

    # ---------- Indices PKL : Head chart ----------
    st.markdown("""
        <div style='border:1px solid #00C29A; padding:10px; border-radius:15px; margin-top:20px;text-align:center;'>
            <h3 style='color:#00C29A; margin:0;'>Indices — AiQR vs Inverse Vol vs Gold (base 100)</h3>
        </div>
    """, unsafe_allow_html=True)
    top_log_scale = st.checkbox("Log scale", value=True, key="top_log_scale")

    equity_all_dict = gold_dict = None
    try:
        equity_all_dict = load_pickle_local("equity_all.pkl")
    except Exception as e:
        st.warning(f"equity_all.pkl : {e}")

    try:
        gold_dict = load_pickle_local("Gold.pkl")
    except Exception as e:
        st.warning(f"Gold.pkl : {e}")

    # Build PKL indices dataframe
    df_idx = None
    if isinstance(equity_all_dict, dict) and isinstance(gold_dict, dict):
        try:
            s_aiqr = _get_from_any(equity_all_dict, 0.9, "0.9")
            s_inv  = _get_from_any(equity_all_dict, 1, 1.0, "1", 0.1, "0.1")
            g_obj  = _get_from_any(gold_dict, "GLD", "gld")

            if s_aiqr is not None and s_inv is not None and g_obj is not None:
                s_aiqr = ensure_datetime_index(s_aiqr)
                s_inv  = ensure_datetime_index(s_inv)
                if isinstance(g_obj, pd.DataFrame):
                    s_gld = g_obj["GLD"] if "GLD" in g_obj.columns else g_obj.iloc[:, 0]
                else:
                    s_gld = pd.Series(g_obj)
                s_gld = ensure_datetime_index(s_gld)
                s_gld = 100 * s_gld / s_gld.iloc[0] if not s_gld.empty else s_gld

                global_start = min(s_aiqr.index.min(), s_inv.index.min(), s_gld.index.min())
                global_end   = max(s_aiqr.index.max(), s_inv.index.max(), s_gld.index.max())
                idx = pd.date_range(global_start, global_end, freq="B")
                s_aiqr = s_aiqr.reindex(idx).ffill()
                s_inv  = s_inv.reindex(idx).ffill()
                s_gld  = s_gld.reindex(idx).ffill()

                df_idx = pd.DataFrame({
                    "Ai for Quant Research Systematic Gold Index": s_aiqr,
                    "Inverse Volatility Gold Index": s_inv,
                    "Gold (GLD)": s_gld
                }, index=idx)

                fig_top = go.Figure()
                fig_top.add_trace(go.Scatter(x=df_idx.index, y=df_idx["Ai for Quant Research Systematic Gold Index"],
                                             mode="lines", name="Ai for Quant Research Systematic Gold Index"))
                fig_top.add_trace(go.Scatter(x=df_idx.index, y=df_idx["Inverse Volatility Gold Index"],
                                             mode="lines", name="Inverse Volatility Gold Index"))
                fig_top.add_trace(go.Scatter(x=df_idx.index, y=df_idx["Gold (GLD)"],
                                             mode="lines", name="Gold (GLD)", line=dict(color="red")))
                layout0 = dict(
                    title="AiQR vs Inverse Vol vs Gold (base 100)",
                    height=520,
                    plot_bgcolor="white", paper_bgcolor="white",
                    xaxis_showgrid=False, yaxis_showgrid=False,
                    margin=dict(l=10, r=10, t=60, b=10),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
                )
                fig_top.update_layout(yaxis_type="log" if top_log_scale else "linear", **layout0)
                st.plotly_chart(fig_top, use_container_width=True)
        except Exception as e:
            st.error(f"Erreur lors du tracé : {e}")
    else:
        st.info("Charge **equity_all.pkl** et **Gold.pkl** pour afficher le graphe principal.")

    # ---------- Metrics (live window selectable) ----------
    st.markdown("""
        <div style='border:1px solid #00C29A; padding:10px; border-radius:15px; margin-top:0px;text-align:center;'>
            <h3 style='color:#00C29A; margin:0;'>Strat Metrics — Live window</h3>
        </div>
    """, unsafe_allow_html=True)
    if df_idx is not None:
        min_d = df_idx.index.min().date()
        default_live = max(datetime.date(2024, 1, 1), min_d)
        live_start_date = st.date_input(
            "Metrics start date (live)",
            value=default_live,
            min_value=min_d,
            max_value=df_idx.index.max().date(),
            help="The metrics below will be calculated from this date."
        )

        df_idx_live = df_idx.loc[pd.to_datetime(live_start_date):]
        if df_idx_live.empty:
            st.warning("Pas de données après la date choisie. Choisis une date plus ancienne.")
        else:
            ret_idx_live = df_idx_live.pct_change().dropna()
            rows = []
            for col in df_idx_live.columns:
                m = compute_asset_metrics(ret_idx_live[col])
                m["Index"] = col
                rows.append(m)
            df_metrics = pd.DataFrame(rows).set_index("Index")
            st.dataframe(
                df_metrics.style.format({
                    "Annual Return": "{:.2%}",
                    "Annual Volatility": "{:.2%}",
                    "Sharpe Ratio": "{:.2f}",
                    "Max Drawdown": "{:.2%}",
                    "1-Day 95% VaR": "{:.2%}",
                }),
                use_container_width=True
            )

            # >>> Rolling Correlation — Strategies (PLACÉ JUSTE APRÈS LES MÉTRIQUES)
            st.markdown("""
                <div style='border:1px solid #00C29A; padding:10px; border-radius:10px; margin-top:16px;text-align:center;'>
                    <h3 style='color:#00C29A; margin:0;'>Rolling Correlation — Strategies (AiQR / Inverse Vol / GLD)</h3>
                </div>
            """, unsafe_allow_html=True)
            corr_win_strat0 = st.number_input("Corr window ", min_value=50, max_value=520, value=260, step=10, key="corr_strat_tab0")
            strat_cols0 = ["Ai for Quant Research Systematic Gold Index",
                           "Inverse Volatility Gold Index",
                           "Gold (GLD)"]
            ret_strat0 = df_idx[strat_cols0].pct_change().dropna()
            from itertools import combinations
            pairs0 = list(combinations(strat_cols0, 2))
            fig_corr_strat0 = fig_rolling_corr_strategies(ret_strat0, pairs0, corr_win_strat0,
                                                          title=f"Rolling Correlation — Strategies (window={corr_win_strat0}d)")
            st.plotly_chart(fig_corr_strat0, use_container_width=True)

    # ---------- Rolling volatility  ----------
    st.markdown("""
        <div style='border:1px solid #00C29A; padding:10px; border-radius:10px; margin-top:20px;text-align:center;'>
            <h3 style='color:#00C29A; margin:0;'>Rolling Volatility </h3>
        </div>
    """, unsafe_allow_html=True)
    if df_idx is not None:
        vol_window_idx = st.number_input("Vol window (indices PKL)", min_value=5, max_value=520, value=30, step=5, key="vol_idx_gold")
        ret_idx2 = df_idx.pct_change().dropna()
        rolling_vol_idx = ret_idx2.rolling(window=vol_window_idx).std() * np.sqrt(252)
        fig_vol_idx = px.line(rolling_vol_idx, title=f"Annualized {vol_window_idx}-day Rolling Volatility (PKL indices)")
        fig_vol_idx.update_layout(height=500, plot_bgcolor="white", paper_bgcolor="white",
                                  xaxis_showgrid=False, yaxis_showgrid=False, margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig_vol_idx, use_container_width=True)

    # ---------- EXTRAS BTC/GLD ----------
    px_df = get_close(selected_tickers, start=start_date, end=end_date)
    if not px_df.empty and px_df.shape[1] == 2:
        returns = px_df.pct_change().dropna()

        # Volatility scatter
        st.markdown("""
            <div style='border:1px solid #00C29A; padding:10px; border-radius:10px; margin-top:20px;text-align:center;'>
                <h3 style='color:#00C29A; margin:0;'>Volatility Scatter (BTC vs GLD)</h3>
            </div>
        """, unsafe_allow_html=True)
        vol_window = st.number_input("Vol window (extras)", min_value=10, max_value=520, value=30, step=5, key="vol_extras_gold")
        vol_df = (returns.rolling(window=vol_window).std() * np.sqrt(252)).dropna()
        if vol_df.shape[1] >= 2:
            x_name, y_name = selected_tickers[0], selected_tickers[1]
            fig_vol_scatter = px.scatter(vol_df.reset_index(), x=x_name, y=y_name, hover_data=["Date"],
                                         title=f"Realized Volatility : {x_name} vs {y_name} (window={vol_window})")
            fig_vol_scatter.update_traces(marker=dict(size=6, opacity=0.7))
            fig_vol_scatter.update_layout(height=520, plot_bgcolor="white", paper_bgcolor="white",
                                          xaxis_showgrid=False, yaxis_showgrid=False, margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig_vol_scatter, use_container_width=True)

        # Momentum scatter
        st.markdown("""
            <div style='border:1px solid #00C29A; padding:10px; border-radius:10px; margin-top:20px;text-align:center;'>
                <h3 style='color:#00C29A; margin:0;'>Momentum Scatter (BTC vs GLD)</h3>
            </div>
        """, unsafe_allow_html=True)
        mom_window = st.number_input("Mom window (extras)", min_value=21, max_value=520, value=63, step=7, key="mom_extras_gold")
        mom_df = ((1 + returns).rolling(window=mom_window).apply(np.prod, raw=True) - 1.0).dropna()
        if mom_df.shape[1] >= 2:
            x_name, y_name = selected_tickers[0], selected_tickers[1]
            fig_mom_scatter = px.scatter(mom_df.reset_index(), x=x_name, y=y_name, hover_data=["Date"],
                                         title=f"Realized Momentum : {x_name} vs {y_name} (window={mom_window})")
            fig_mom_scatter.update_traces(marker=dict(size=6, opacity=0.7))
            fig_mom_scatter.update_layout(height=520, plot_bgcolor="white", paper_bgcolor="white",
                                          xaxis_showgrid=False, yaxis_showgrid=False, margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig_mom_scatter, use_container_width=True)

        # Rolling Correlation (BTC vs GLD)
        st.markdown("""
            <div style='border:1px solid #00C29A; padding:10px; border-radius:10px; margin-top:20px;text-align:center;'>
                <h3 style='color:#00C29A; margin:0;'>Rolling Correlation (BTC vs GLD)</h3>
            </div>
        """, unsafe_allow_html=True)
        corr_window = st.number_input("Corr window (extras)", min_value=60, max_value=520, value=260, step=10, key="corr_extras_gold")
        s1 = returns[selected_tickers[0]]
        s2 = returns[selected_tickers[1]]
        cor_btc_gold = s1.rolling(corr_window).corr(s2).dropna()
        fig_corr_line = go.Figure()
        fig_corr_line.add_trace(go.Scatter(x=cor_btc_gold.index, y=cor_btc_gold.values,
                                           name=f"{selected_tickers[0]} vs {selected_tickers[1]} ({corr_window}j rolling corr)",
                                           mode="lines"))
        fig_corr_line.add_hline(y=0.0, line_dash="dash", line_width=1.5, line_color="red",
                                annotation_text="0", annotation_position="top left")
        fig_corr_line.update_layout(
            title=f"Corrélation {selected_tickers[0]} vs {selected_tickers[1]} ({corr_window} jours)",
            height=420, plot_bgcolor="white", paper_bgcolor="white",
            xaxis_showgrid=False, yaxis_showgrid=False, margin=dict(l=10, r=10, t=60, b=10)
        )
        st.plotly_chart(fig_corr_line, use_container_width=True)

# =============================
# TAB 1 — Precious Metal Index
# =============================
with tabs[1]:
    # ---------- Description ----------
    st.markdown("""
        <div style='
            border:1px solid #00C29A;
            padding:15px;
            border-radius:12px;
            margin-top:10px;
            background-color:#F8FFFC;
            text-align:justify;
            font-size:15px;
            line-height:1.6;
        '>
        The AIQR Precious Metal Index aims to deliver superior risk-adjusted exposure to the precious
        metals market through a systematic, volatility-controlled approach. The index combines exposure to
        precious metals and Bitcoin (BTC) as complementary assets. By dynamically allocating risk between
        traditional safe-haven metals and digital store-of-value assets, the index seeks to enhance long-term
        compounded returns while keeping drawdowns moderate.
        The strategy is long-only, rebalanced weekly, and benchmarked against the DBP Precious Metal ETF.
        Its design reflects AIQR’s research-driven objective to capture asymmetric performance while maintaining
        robust downside protection.
        </div>
    """, unsafe_allow_html=True)

    # ---------- PM Universe ----------
    pm_tickers = ["BTC-USD", "GLD", "SLV", "PPLT", "PALL"]
    start_date = "2016-01-01"
    end_date = pd.to_datetime("today").strftime("%Y-%m-%d")

    # ---------- Head chart using PKL ----------
    st.markdown("""
        <div style='border:1px solid #00C29A; padding:10px; border-radius:15px; margin-top:20px;text-align:center;'>
            <h3 style='color:#00C29A; margin:0;'>Indices — AiQR PM vs Inverse Vol PM vs DBP (base 100)</h3>
        </div>
    """, unsafe_allow_html=True)
    top_log_scale_pm = st.checkbox("Log scale", value=True, key="top_log_scale_pm")

    try:
        equity_all_pm = load_pickle_local("equity_all_precious_metal.pkl")
    except Exception as e:
        equity_all_pm = None
        st.warning(f"equity_all_precious_metal.pkl : {e}")

    try:
        dbp_dict = load_pickle_local("DBP.pkl")
    except Exception as e:
        dbp_dict = None
        st.warning(f"DBP.pkl : {e}")

    df_idx_pm = None
    if isinstance(equity_all_pm, dict) and isinstance(dbp_dict, dict):
        try:
            s_aiqr_pm = _get_from_any(equity_all_pm, 0.9, "0.9")
            s_inv_pm  = _get_from_any(equity_all_pm, 1, 1.0, "1", 0.1, "0.1")
            g_obj     = _get_from_any(dbp_dict, "DBP", "dbp")

            if s_aiqr_pm is not None and s_inv_pm is not None and g_obj is not None:
                s_aiqr_pm = ensure_datetime_index(s_aiqr_pm)
                s_inv_pm  = ensure_datetime_index(s_inv_pm)
                if isinstance(g_obj, pd.DataFrame):
                    s_dbp = g_obj["DBP"] if "DBP" in g_obj.columns else g_obj.iloc[:, 0]
                else:
                    s_dbp = pd.Series(g_obj)
                s_dbp = ensure_datetime_index(s_dbp)
                s_dbp = 100 * s_dbp / s_dbp.iloc[0] if not s_dbp.empty else s_dbp

                idx = pd.date_range(min(s_aiqr_pm.index.min(), s_inv_pm.index.min(), s_dbp.index.min()),
                                    max(s_aiqr_pm.index.max(), s_inv_pm.index.max(), s_dbp.index.max()), freq="B")
                df_idx_pm = pd.DataFrame({
                    "AiQR Precious Metal Index": s_aiqr_pm.reindex(idx).ffill(),
                    "Inverse Vol Precious Metal Index": s_inv_pm.reindex(idx).ffill(),
                    "DBP (PM ETF)": s_dbp.reindex(idx).ffill()
                }, index=idx)

                fig_top_pm = go.Figure()
                for col in df_idx_pm.columns:
                    fig_top_pm.add_trace(go.Scatter(x=df_idx_pm.index, y=df_idx_pm[col], mode="lines", name=col))
                layout_pm = dict(
                    title="AiQR PM vs Inverse Vol PM vs DBP (base 100)",
                    height=520, plot_bgcolor="white", paper_bgcolor="white",
                    xaxis_showgrid=False, yaxis_showgrid=False,
                    margin=dict(l=10, r=10, t=60, b=10),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
                )
                fig_top_pm.update_layout(yaxis_type="log" if top_log_scale_pm else "linear", **layout_pm)
                st.plotly_chart(fig_top_pm, use_container_width=True)
        except Exception as e:
            st.error(f"Erreur chargement PKL : {e}")
    else:
        st.info("Charge **equity_all_precious_metal.pkl** et **DBP.pkl** pour afficher le graphe en tête.")

    # ---------- Strat Metrics — Live window (PM) ----------
    st.markdown("""
        <div style='border:1px solid #00C29A; padding:10px; border-radius:15px; margin-top:0px;text-align:center;'>
            <h3 style='color:#00C29A; margin:0;'>Strat Metrics — Live window (PM)</h3>
        </div>
    """, unsafe_allow_html=True)
    if df_idx_pm is not None:
        min_d_pm = df_idx_pm.index.min().date()
        default_live_pm = max(datetime.date(2024, 1, 1), min_d_pm)
        live_start_date_pm = st.date_input(
            "Metrics start date (live — PM)",
            value=default_live_pm,
            min_value=min_d_pm,
            max_value=df_idx_pm.index.max().date(),
            help="Les métriques ci-dessous seront calculées à partir de cette date (indices PM)."
        )

        df_idx_live_pm = df_idx_pm.loc[pd.to_datetime(live_start_date_pm):]
        if df_idx_live_pm.empty:
            st.warning("Pas de données PM après la date choisie. Choisis une date plus ancienne.")
        else:
            ret_idx_live_pm = df_idx_live_pm.pct_change().dropna()
            rows_pm = []
            for col in df_idx_live_pm.columns:
                m = compute_asset_metrics(ret_idx_live_pm[col])
                m["Index"] = col
                rows_pm.append(m)
            df_metrics_pm = pd.DataFrame(rows_pm).set_index("Index")
            st.dataframe(
                df_metrics_pm.style.format({
                    "Annual Return": "{:.2%}",
                    "Annual Volatility": "{:.2%}",
                    "Sharpe Ratio": "{:.2f}",
                    "Max Drawdown": "{:.2%}",
                    "1-Day 95% VaR": "{:.2%}",
                }),
                use_container_width=True
            )

            # >>> Rolling Correlation — Strategies (PLACÉ JUSTE APRÈS LES MÉTRIQUES, avec légende à droite)
            st.markdown("""
                <div style='border:1px solid #00C29A; padding:10px; border-radius:10px; margin-top:16px;text-align:center;'>
                    <h3 style='color:#00C29A; margin:0;'>Rolling Correlation — Strategies (AiQR PM / Inverse Vol PM / DBP)</h3>
                </div>
            """, unsafe_allow_html=True)
            corr_win_strat1 = st.number_input("Corr window (strategies — Tab 1)", min_value=60, max_value=520, value=260, step=10, key="corr_strat_tab1")

            strat_cols1 = ["AiQR Precious Metal Index",
                           "Inverse Vol Precious Metal Index",
                           "DBP (PM ETF)"]
            ret_strat1 = df_idx_pm[strat_cols1].pct_change().dropna()
            from itertools import combinations
            pairs1 = list(combinations(strat_cols1, 2))
            fig_corr_strat1 = fig_rolling_corr_strategies(
                ret_strat1, pairs1, corr_win_strat1,
                title=f"Rolling Correlation — Strategies (window={corr_win_strat1}d)"
            )
            st.plotly_chart(fig_corr_strat1, use_container_width=True)

            

    # ---------- EXTRAS — VOL / MOM / CORR ----------
    px_pm = get_close(pm_tickers, start=start_date, end=end_date)
    if px_pm.empty:
        st.warning("Pas de données disponibles pour l’univers PM.")
    else:
        ret_pm = px_pm.pct_change().dropna()

        # Volatility Scatter Matrix
        st.markdown("""
            <div style='border:1px solid #00C29A; padding:10px; border-radius:10px; margin-top:20px;text-align:center;'>
                <h3 style='color:#00C29A; margin:0;'>Realized Volatility — Scatter Matrix</h3>
            </div>
        """, unsafe_allow_html=True)
        vol_window_pm = st.number_input("Vol window (extras — PM)", min_value=10, max_value=520, value=30, step=5, key="vol_extras_pm")
        vol_df_pm = (ret_pm.rolling(vol_window_pm).std() * np.sqrt(252)).dropna()
        try:
            fig_vol_matrix = px.scatter_matrix(vol_df_pm, dimensions=pm_tickers,
                                               title=f"Realized Volatility — Scatter Matrix (window={vol_window_pm})")
            fig_vol_matrix.update_traces(marker=dict(size=3, opacity=0.6))
            fig_vol_matrix.update_layout(height=800, plot_bgcolor="white", paper_bgcolor="white",
                                         margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig_vol_matrix, use_container_width=True)
        except Exception as e:
            st.error(f"Echec Scatter Matrix Vol: {e}")

        # Momentum Scatter Matrix
        st.markdown("""
            <div style='border:1px solid #00C29A; padding:10px; border-radius:10px; margin-top:20px;text-align:center;'>
                <h3 style='color:#00C29A; margin:0;'>Realized Momentum — Scatter Matrix</h3>
            </div>
        """, unsafe_allow_html=True)
        mom_window_pm = st.number_input("Mom window (extras — PM)", min_value=21, max_value=520, value=63, step=7, key="mom_extras_pm")
        mom_df_pm = ((1 + ret_pm).rolling(mom_window_pm).apply(np.prod, raw=True) - 1).dropna()
        try:
            fig_mom_matrix = px.scatter_matrix(mom_df_pm, dimensions=pm_tickers,
                                               title=f"Realized Momentum — Scatter Matrix (window={mom_window_pm})")
            fig_mom_matrix.update_traces(marker=dict(size=3, opacity=0.6))
            fig_mom_matrix.update_layout(height=800, plot_bgcolor="white", paper_bgcolor="white",
                                         margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig_mom_matrix, use_container_width=True)
        except Exception as e:
            st.error(f"Echec Scatter Matrix Momentum: {e}")

        # Rolling Correlations — 4 / 3 / 3 (pairs d'actifs)
        st.markdown("""
            <div style='border:1px solid #00C29A; padding:10px; border-radius:10px; margin-top:20px;text-align:center;'>
                <h3 style='color:#00C29A; margin:0;'>Rolling Correlations — Multiple Graphs</h3>
            </div>
        """, unsafe_allow_html=True)
        corr_window_pm = st.number_input("Corr window (extras — PM)", min_value=60, max_value=520, value=260, step=10, key="corr_extras_pm")

        from itertools import combinations
        pairs = list(combinations(pm_tickers, 2))  # 10 pairs for 5 assets
        pair_groups = [pairs[0:4], pairs[4:7], pairs[7:10]]  # 4 / 3 / 3

        for gi, group in enumerate(pair_groups, start=1):
            fig_corr_sub = go.Figure()
            for a, b in group:
                cs = ret_pm[a].rolling(corr_window_pm).corr(ret_pm[b]).dropna()
                if not cs.empty:
                    fig_corr_sub.add_trace(go.Scatter(
                        x=cs.index, y=cs.values, mode="lines",
                        name=f"{a} vs {b}", showlegend=True
                    ))
            fig_corr_sub.add_hline(y=0.0, line_dash="dot", line_color="gray", line_width=1)
            fig_corr_sub.update_layout(
                title=f"Rolling Correlation — Group {gi} (window={corr_window_pm}d)",
                height=420, plot_bgcolor="white", paper_bgcolor="white",
                xaxis_showgrid=False, yaxis_showgrid=False,
                margin=dict(l=10, r=10, t=60, b=10),
                legend=dict(orientation="h", yanchor="top", y=0.98, xanchor="left", x=0.01,
                            bgcolor="rgba(255,255,255,0.8)", bordercolor="rgba(0,0,0,0.2)", borderwidth=1, title="Pairs"),
                showlegend=True
            )
            st.plotly_chart(fig_corr_sub, use_container_width=True)

# =========================
# Placeholders for other tabs
# =========================
with tabs[2]:
    st.info("Crypto Factor — Very soon!")
with tabs[3]:
    st.info("Trend Following — Very soon!")
with tabs[4]:
    st.info("Systematic Index Trading — Very soon!")
with tabs[5]:
    st.info("Optimized Portfolio — Very soon!")
