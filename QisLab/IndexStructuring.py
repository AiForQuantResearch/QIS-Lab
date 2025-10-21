import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

class Index:
    """
    Classe de construction et de suivi d’un indice systématique basé sur des règles simples
    d’allocation (Equal Weight, Inverse Volatility, ou IVOL_MOM), en **long-only**.

    Fonctionnalités :
      - Backtest simulation (compute_index)
      - Prolongation live (compute_live)
      - Simulation + live (compute_all)
      - Historique & projection des poids (weights_evolution)
      - Visualisation des poids (plot_weights)
    """

    # ======================
    #  Initialisation
    # ======================
    def __init__(self, returns: pd.DataFrame, signal_window: int, backtest_window: int, allocation: str):
        """
        Parameters
        ----------
        returns : pd.DataFrame
            Rendements historiques (index = dates, colonnes = actifs).
        signal_window : int
            Taille de fenêtre pour estimer les signaux (vol, mom...).
        backtest_window : int
            Taille du bloc durant lequel on applique des poids figés.
        allocation : {'EQW','IVOL','IVOL_MOM'}
            Méthode d'allocation.
        """
        self.returns = returns.sort_index()
        self.signal_window = int(signal_window)
        self.backtest_window = int(backtest_window)
        self.allocation = allocation.upper()
        self._check_allocation()

    def _check_allocation(self):
        if self.allocation not in ('EQW', 'IVOL', 'IVOL_MOM'):
            raise ValueError("allocation doit être 'EQW', 'IVOL' ou 'IVOL_MOM'.")

    # ======================
    #  Calcul des poids
    # ======================
    def _compute_weights(self, hist_returns: pd.DataFrame, dates_signal: pd.DatetimeIndex, ivol_mom_weight: float) -> pd.Series:
        """
        Calcule les poids selon la méthode d’allocation spécifiée (long-only).

        Parameters
        ----------
        hist_returns : pd.DataFrame
        dates_signal : pd.DatetimeIndex
        ivol_mom_weight : float

        Returns
        -------
        pd.Series
            Poids (somme = 1, tous >= 0)
        """
        slice_sig = hist_returns.loc[dates_signal]
        if slice_sig.empty:
            return pd.Series(0.0, index=hist_returns.columns, dtype=float)

        if self.allocation == 'EQW':
            n_assets = len(hist_returns.columns)
            w = pd.Series(1.0 / n_assets, index=hist_returns.columns, dtype=float)

        elif self.allocation == 'IVOL':
            vol = slice_sig.std()
            inv_vol = (1.0 / vol.replace(0.0, np.nan)).fillna(0.0)
            s = inv_vol.sum()
            w = (inv_vol / s) if s > 0 else pd.Series(0.0, index=hist_returns.columns, dtype=float)

        elif self.allocation == 'IVOL_MOM':
            vol = slice_sig.std()
            mom = slice_sig.mean().clip(lower=0.0)  # momentum négatif = 0 pour long-only
            inv_vol = (1.0 / vol.replace(0.0, np.nan)).fillna(0.0)

            s1 = inv_vol.sum()
            s2 = mom.sum()
            w1 = (inv_vol / s1) if s1 > 0 else pd.Series(0.0, index=inv_vol.index, dtype=float)
            w2 = (mom / s2)     if s2 > 0 else pd.Series(0.0, index=mom.index,     dtype=float)

            w = ivol_mom_weight * w1 + (1.0 - ivol_mom_weight) * w2

        else:
            w = pd.Series(0.0, index=hist_returns.columns, dtype=float)

        # Contraintes long-only + normalisation
        w = w.clip(lower=0.0).fillna(0.0)
        total = w.sum()
        if total > 0:
            w = w / total
        else:
            w[:] = 0.0

        return w

    # ======================
    #  Backtest simulation
    # ======================
    def compute_index(self, as_equity: bool = False, ivol_mom_weight: float = 0.9, start_value: float = 1000.0) -> pd.Series:
        """
        Backtest de l’indice sur la période historique (simulation).

        Parameters
        ----------
        as_equity : bool
        ivol_mom_weight : float
        start_value : float

        Returns
        -------
        pd.Series
            Rendements pondérés OU courbe d’indice (base start_value).
        """
        rets = self.returns.copy()
        index_ret = pd.Series(dtype=float, index=rets.index)
        all_idx = rets.index
        n_days = len(all_idx)

        for start in range(0, n_days, self.backtest_window):
            sig_end = start + self.signal_window
            bt_end  = sig_end + self.backtest_window
            if bt_end > n_days:
                break

            dates_signal = all_idx[start:sig_end]
            dates_back   = all_idx[sig_end:bt_end]

            w = self._compute_weights(rets, dates_signal, ivol_mom_weight)

            block = rets.loc[dates_back].dropna(how='all')
            if not block.empty:
                weighted_ret = (block @ w).rename("IndexReturns")
                index_ret.loc[block.index] = weighted_ret.values

        if as_equity:
            return (1.0 + index_ret.fillna(0.0)).cumprod() * float(start_value)
        return index_ret

    # ======================
    #  Backtest live
    # ======================
    def compute_live(self, returns_live: pd.DataFrame, as_equity: bool = False, ivol_mom_weight: float = 0.9, start_value: float = 1000.0) -> pd.Series:
        """
        Prolonge la performance de l’indice sur des données "live" (par blocs).

        Parameters
        ----------
        returns_live : pd.DataFrame
        as_equity : bool
        ivol_mom_weight : float
        start_value : float

        Returns
        -------
        pd.Series
        """
        sim = self.returns.copy()
        live = returns_live.copy().sort_index()

        # Alignement de l’univers
        common_cols = sim.columns.intersection(live.columns)
        if len(common_cols) == 0:
            raise ValueError("Pas d'actifs communs entre 'returns' (simulation) et 'returns_live'.")
        sim = sim[common_cols]
        live = live[common_cols]

        hist = sim.copy()
        out_live = pd.Series(dtype=float, index=live.index)
        live_idx = live.index
        n_days_live = len(live_idx)

        pos = 0
        while pos < n_days_live:
            sig_end_hist = len(hist.index)
            start_sig = max(0, sig_end_hist - self.signal_window)
            dates_signal = hist.index[start_sig:sig_end_hist]

            back_end_pos = min(pos + self.backtest_window, n_days_live)
            dates_back = live_idx[pos:back_end_pos]

            w = self._compute_weights(hist, dates_signal, ivol_mom_weight)
            block = live.loc[dates_back].dropna(how='all')
            if not block.empty:
                weighted_ret = (block @ w).rename("IndexReturnsLive")
                out_live.loc[block.index] = weighted_ret.values
                hist = pd.concat([hist, block], axis=0)
            pos = back_end_pos

        if as_equity:
            return (1.0 + out_live.fillna(0.0)).cumprod() * float(start_value)
        return out_live

    # ======================
    #  Simulation + Live
    # ======================
    def compute_all(self, returns_live: pd.DataFrame = None, as_equity: bool = False, ivol_mom_weight: float = 0.9, start_value: float = 1000.0) -> pd.Series:
        """
        Combine les périodes de simulation et de live en une seule série.

        Parameters
        ----------
        returns_live : pd.DataFrame | None
        as_equity : bool
        ivol_mom_weight : float
        start_value : float

        Returns
        -------
        pd.Series
        """
        sim_part = self.compute_index(as_equity=False, ivol_mom_weight=ivol_mom_weight)
        if returns_live is None:
            return (1.0 + sim_part.fillna(0.0)).cumprod() * float(start_value) if as_equity else sim_part

        live_part = self.compute_live(returns_live, as_equity=False, ivol_mom_weight=ivol_mom_weight)
        all_part = pd.concat([sim_part, live_part], axis=0)
        all_part = all_part[~all_part.index.duplicated(keep='last')].sort_index()

        return (1.0 + all_part.fillna(0.0)).cumprod() * float(start_value) if as_equity else all_part

    # ======================
    #  Évolution & prévision des poids
    # ======================
    def weights_evolution(self, returns_live: pd.DataFrame = None, ivol_mom_weight: float = 0.9, horizon_days: int = 0, expand_daily: bool = True) -> dict:
        """
        Calcule :
          - l’évolution des poids à chaque rééquilibrage (simulation + live)
          - une version journalière des poids (forward-fill)
          - une projection future (forecast) pour horizon_days jours

        Parameters
        ----------
        returns_live : pd.DataFrame | None
        ivol_mom_weight : float
        horizon_days : int
        expand_daily : bool

        Returns
        -------
        dict
            {
              'weights_block'   : DataFrame (dates de rééquilibrage x actifs),
              'weights_daily'   : DataFrame journalier (si expand_daily=True),
              'weights_forecast': DataFrame (si horizon_days>0)
            }
        """
        # === 1. Simulation ===
        rets = self.returns.copy()
        all_idx = rets.index
        n_days = len(all_idx)
        block_weights_records = []

        for start in range(0, n_days, self.backtest_window):
            sig_end = start + self.signal_window
            bt_end  = sig_end + self.backtest_window
            if bt_end > n_days:
                break
            dates_signal = all_idx[start:sig_end]
            dates_back   = all_idx[sig_end:bt_end]
            if len(dates_back) == 0:
                continue
            w = self._compute_weights(rets, dates_signal, ivol_mom_weight)
            block_weights_records.append((dates_back[0], w.copy()))

        # === 2. Live (optionnel) ===
        if returns_live is not None:
            sim = self.returns.copy()
            live = returns_live.copy().sort_index()
            common_cols = sim.columns.intersection(live.columns)
            if len(common_cols) == 0:
                raise ValueError("Pas d'actifs communs entre simulation et live.")
            sim, live = sim[common_cols], live[common_cols]
            hist = sim.copy()
            live_idx = live.index
            pos = 0
            while pos < len(live_idx):
                start_sig = max(0, len(hist) - self.signal_window)
                dates_signal = hist.index[start_sig:]
                back_end_pos = min(pos + self.backtest_window, len(live_idx))
                dates_back = live_idx[pos:back_end_pos]
                w_live = self._compute_weights(hist, dates_signal, ivol_mom_weight)
                block_weights_records.append((dates_back[0], w_live.copy()))
                hist = pd.concat([hist, live.loc[dates_back]], axis=0)
                pos = back_end_pos

        # === 3. Assemblage ===
        if not block_weights_records:
            empty_df = pd.DataFrame(columns=self.returns.columns, dtype=float)
            return {'weights_block': empty_df, 'weights_daily': empty_df, 'weights_forecast': empty_df}

        weights_block = pd.DataFrame({d: w for d, w in block_weights_records}).T.sort_index()
        weights_block.index.name = "RebalanceDate"

        # === 4. Daily ===
        weights_daily = None
        if expand_daily:
            full_dates = self.returns.index if returns_live is None \
                else pd.Index(sorted(set(self.returns.index).union(returns_live.index)))
            weights_daily = (
                weights_block.reindex(full_dates.union(weights_block.index))
                             .sort_index()
                             .ffill()
                .loc[full_dates]
            )
            weights_daily.index.name = "Date"

        # === 5. Forecast ===
        weights_forecast = None
        if horizon_days > 0:
            last_weights = weights_block.iloc[-1]
            base_index = self.returns.index if returns_live is None \
                else pd.Index(sorted(set(self.returns.index).union(returns_live.index)))
            freq = pd.infer_freq(base_index) or 'D'
            start_forecast = (base_index.max() if len(base_index) else pd.Timestamp.today()).normalize() \
                             + pd.tseries.frequencies.to_offset(freq)
            future_dates = pd.date_range(start=start_forecast, periods=horizon_days, freq=freq)
            future_dates = future_dates[future_dates.weekday < 5][:horizon_days]
            weights_forecast = pd.DataFrame(
                np.tile(last_weights.values, (len(future_dates), 1)),
                index=future_dates,
                columns=last_weights.index
            )
            weights_forecast.index.name = "Date"

        return {'weights_block': weights_block, 'weights_daily': weights_daily, 'weights_forecast': weights_forecast}

    # ======================
    #  Visualisation des poids
    # ======================
    def plot_weights(self,
                     returns_live: pd.DataFrame = None,
                     ivol_mom_weight: float = 0.9,
                     use_daily: bool = True,
                     horizon_days: int = 0,
                     figsize=(14, 6),
                     title: str = None,
                     savepath: str = None) -> pd.DataFrame:
        """
        Affiche l’évolution des poids sous forme de **stacked area chart** (matplotlib).

        Parameters
        ----------
        returns_live : pd.DataFrame | None
            Pour inclure la partie live dans l’historique des poids.
        ivol_mom_weight : float
            Poids de la composante IVOL (si 'IVOL_MOM').
        use_daily : bool
            - True  : trace les poids **journaliers** (forward-fill entre rééquilibrages)
            - False : trace les poids uniquement aux **dates de rééquilibrage**
        horizon_days : int
            Si > 0, projette et trace aussi les poids futurs (figés).
        figsize : tuple
            Taille de la figure matplotlib.
        title : str | None
            Titre personnalisé. Par défaut, construit automatiquement.
        savepath : str | None
            Si renseigné, sauvegarde la figure à ce chemin.

        Returns
        -------
        pd.DataFrame
            Le DataFrame de poids effectivement tracé (utile pour export).
        """
        res = self.weights_evolution(returns_live=returns_live,
                                     ivol_mom_weight=ivol_mom_weight,
                                     horizon_days=horizon_days,
                                     expand_daily=use_daily)

        if use_daily:
            df = res['weights_daily']
        else:
            df = res['weights_block']

        if df is None or df.empty:
            raise ValueError("Aucun poids à tracer. Vérifie l'historique et les paramètres.")

        # Concatène la projection si demandée
        if horizon_days > 0 and res['weights_forecast'] is not None:
            df = pd.concat([df, res['weights_forecast']]).sort_index()

        # Plot
        plt.figure(figsize=figsize)
        ax = df.plot.area(figsize=figsize)  # couleurs par défaut de matplotlib
        ax.set_ylim(0, 1)
        ax.set_ylabel("Poids")
        if title is None:
            mode = "Daily" if use_daily else "Rebalance-only"
            title = f"Évolution des poids ({self.allocation}) – {mode}"
            if horizon_days > 0:
                title += f" + Forecast ({horizon_days}j)"
        ax.set_title(title, loc='left', fontsize=12)
        ax.legend(loc='upper left', ncol=2, fontsize=9, frameon=False)
        plt.tight_layout()

        if savepath:
            plt.savefig(savepath, dpi=150, bbox_inches='tight')

        plt.show()
        return df
