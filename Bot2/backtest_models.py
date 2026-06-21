"""
backtest_models_optimized.py

OPTIMISATIONS APPLIQUÉES:
1. [SIGNAL] Signal -1 = FLAT (pas de position) — bug corrigé, plus de SELL forcé
2. [FILTRE] Filtre de tendance MA200 appliqué avant le backtest
3. [MÉTRIQUES] Métriques étendues: win_rate, avg_win, avg_loss, n_trades, flat_pct
4. [RAPPORT] Rapport mensuel automatique + résumé des signaux
5. [SEUILS] Seuils buy/sell chargés depuis ensemble_meta (cohérence avec l'entraînement)
6. [SAUVEGARDE] CSV enrichi avec colonne 'position' (1/−1/0) pour analyse ultérieure
7. [ROBUSTESSE] Vérification de cohérence features avant inférence
"""


import os
import joblib
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import load_model

from run import (
    CONFIG, fetch_mt5, add_features_full, add_labels,
    X_3d_RNN, backtest_with_costs, preload_models,
    build_predict_fns_from_models, ensemble_vote_majority,
    apply_trend_filter,
)


def print_section(title):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print('='*55)


def run_historical_backtest():
    print_section("DÉMARRAGE DU BACKTEST HISTORIQUE OPTIMISÉ")

    # 1. Initialisation MT5
    if not mt5.initialize():
        print("Erreur initialisation MT5")
        return
    si = mt5.symbol_info(CONFIG["SYMBOL"])

    # 2. Chargement des ressources
    model_dir    = CONFIG["MODEL_DIR"]
    scaler       = joblib.load(os.path.join(model_dir, "scaler.joblib"))
    feature_cols = joblib.load(os.path.join(model_dir, "feature_cols.joblib"))
    ensemble_meta = joblib.load(os.path.join(model_dir, "ensemble_meta.joblib"))
    model_paths  = ensemble_meta.get("selected", [])

    # OPT-5: charger les seuils depuis ensemble_meta (cohérence)
    thresh_buy  = ensemble_meta.get("thresholds", {}).get("buy",  CONFIG["THRESH_BUY"])
    thresh_sell = ensemble_meta.get("thresholds", {}).get("sell", CONFIG["THRESH_SELL"])
    print(f"[CONFIG] Seuils chargés → BUY > {thresh_buy} | SELL < {thresh_sell}")

    # OPT-7: vérifier si le filtre de tendance est activé
    use_trend_filter = CONFIG.get("USE_TREND_FILTER", True)
    print(f"[CONFIG] Filtre tendance MA200 : {'ACTIF' if use_trend_filter else 'INACTIF'}")

    # OPT-7: vérifier la présence de ma200 dans les features
    if use_trend_filter and "ma200" not in feature_cols:
        print("[WARN] ma200 absent des feature_cols — filtre tendance sur close/ma200 du df_prices")

    # 3. Chargement des modèles
    models      = preload_models(model_paths)
    predict_fns = build_predict_fns_from_models(models, CONFIG["LOOKBACK"], len(feature_cols))
    print(f"[INFO] {len(models)} modèles chargés, {len(predict_fns)} fonctions d'inférence")

    # OPT-7: vérification cohérence features
    print(f"[INFO] Features utilisées ({len(feature_cols)}): {feature_cols}")

    # 4. Récupération des données historiques
    print_section("CHARGEMENT DES DONNÉES")
    df = fetch_mt5(CONFIG["SYMBOL"], CONFIG["TIMEFRAME"], "2026-01-01", datetime.now())
    df = add_features_full(df)
    df = df.dropna()
    print(f"[DATA] {len(df)} barres disponibles après feature engineering")

    # 5. Préparation des séquences 3D
    X_2d = scaler.transform(df[feature_cols].values)
    X_3d, _ = X_3d_RNN(X_2d, np.zeros(len(X_2d)), CONFIG["LOOKBACK"])

    # Aligner les prix avec les séquences (on perd les LOOKBACK premières barres)
    df_prices = df.iloc[CONFIG["LOOKBACK"]:].copy().reset_index(drop=True)
    print(f"[DATA] {len(X_3d)} séquences 3D générées ({len(df_prices)} barres alignées)")

    # 6. Génération des signaux historiques
    print_section("GÉNÉRATION DES SIGNAUX")
    print(f"[INFO] Vote majoritaire sur {len(predict_fns)} modèles (unanimité)")
    historical_signals = []

    for i in range(len(X_3d)):
        X_window = X_3d[i:i+1]
        vote, _  = ensemble_vote_majority(
            predict_fns, X_window,
            prob_threshold=0.5,
            required_majority=len(predict_fns)  # OPT-8: unanimité
        )
        historical_signals.append(vote)

        if (i + 1) % 500 == 0:
            n_buy  = historical_signals.count(1)
            n_sell = historical_signals.count(0)
            n_flat = historical_signals.count(-1)
            print(f"  [{i+1}/{len(X_3d)}] BUY={n_buy} SELL={n_sell} FLAT={n_flat}")

    historical_signals = np.array(historical_signals)

    # Résumé distribution des signaux bruts
    n_buy  = (historical_signals == 1).sum()
    n_sell = (historical_signals == 0).sum()
    n_flat = (historical_signals == -1).sum()
    total  = len(historical_signals)
    print(f"\n[SIGNAUX BRUTS] BUY: {n_buy} ({n_buy/total*100:.1f}%) | "
          f"SELL: {n_sell} ({n_sell/total*100:.1f}%) | "
          f"FLAT: {n_flat} ({n_flat/total*100:.1f}%)")

    # OPT-2: Appliquer le filtre de tendance MA200
    if use_trend_filter:
        historical_signals = apply_trend_filter(historical_signals, df_prices, CONFIG)
        n_buy2  = (historical_signals == 1).sum()
        n_sell2 = (historical_signals == 0).sum()
        n_flat2 = (historical_signals == -1).sum()
        print(f"[SIGNAUX FILTRÉS] BUY: {n_buy2} ({n_buy2/total*100:.1f}%) | "
              f"SELL: {n_sell2} ({n_sell2/total*100:.1f}%) | "
              f"FLAT: {n_flat2} ({n_flat2/total*100:.1f}%)")

    # OPT-1: Ne pas convertir -1 en 0 — backtest gère flat correctement
    # (dans l'ancien code: clean_signals = np.where(historical_signals == -1, 0, historical_signals) ← BUG)
    clean_signals = historical_signals  # -1 = flat, géré par backtest_with_costs

    # 7. Calcul des performances
    print_section("CALCUL DES PERFORMANCES")
    results = backtest_with_costs(df_prices, clean_signals, si, CONFIG)

    # 8. Métriques étendues (OPT-3)
    net_ret  = results["net_ret"]
    wins     = net_ret[net_ret > 0]
    losses   = net_ret[net_ret < 0]
    win_rate = len(wins) / (len(wins) + len(losses) + 1e-9) * 100
    avg_win  = wins.mean()  if len(wins)   > 0 else 0.0
    avg_loss = losses.mean() if len(losses) > 0 else 0.0
    rr_ratio = avg_win / (abs(avg_loss) + 1e-9)  # Risk/Reward ratio

    # 9. Rapport mensuel (OPT-4)
    df_prices["signal"]   = historical_signals
    df_prices["position"] = np.where(historical_signals == 1, 1,
                             np.where(historical_signals == 0, -1, 0))
    df_prices["net_pnl"]  = net_ret
    df_prices["cum_pnl"]  = results["cum"]

    if "time" in df_prices.columns:
        df_prices["time"] = pd.to_datetime(df_prices["time"])
        df_prices["month"] = df_prices["time"].dt.to_period("M")
    else:
        df_prices.index = pd.to_datetime(df_prices.index)
        df_prices["month"] = df_prices.index.to_period("M")

    monthly = df_prices.groupby("month")["net_pnl"].sum()

    # 10. Affichage complet
    print_section("RÉSULTATS DU BACKTEST")
    print(f"  Profit Total      : {results['total']:+.4f}")
    print(f"  Drawdown Max      : {results['max_dd']:.4f}")
    print(f"  Ratio Sharpe      : {results['sharpe']:.3f}")
    print(f"  Profit Factor     : {results['pf']:.3f}")
    print(f"  Ratio Rdt/Risque  : {results['ratio']:.3f}")
    print(f"  Nb Trades         : {results.get('n_trades', 'N/A')}")
    print(f"  % Temps en Flat   : {results.get('flat_pct', 0):.1f}%")
    print(f"  Win Rate          : {win_rate:.1f}%")
    print(f"  Gain Moyen        : {avg_win:.6f}")
    print(f"  Perte Moyenne     : {avg_loss:.6f}")
    print(f"  Ratio G/P         : {rr_ratio:.2f}x")

    print_section("PERFORMANCE MENSUELLE")
    for period, pnl in monthly.items():
        bar  = "█" * int(abs(pnl) * 500)
        sign = "+" if pnl >= 0 else ""
        print(f"  {period} : {sign}{pnl:.4f}  {bar}")

    # 11. Sauvegarde CSV enrichi (OPT-6)
    output_path = "backtest_results_detailed_optimized.csv"
    cols_to_save = [c for c in [
        "time","open","high","low","close","tick_volume",
        "atr14","rsi14","ma50","ma200","signal","position","net_pnl","cum_pnl"
    ] if c in df_prices.columns]
    df_prices[cols_to_save].to_csv(output_path, index=False)
    print(f"\n[SAUVEGARDE] '{output_path}' généré ({len(df_prices)} lignes)")

    # Résumé compact pour copier-coller
    print_section("RÉSUMÉ COMPACT")
    print(f"  PnL={results['total']:+.4f} | DD={results['max_dd']:.4f} | "
          f"Sharpe={results['sharpe']:.2f} | PF={results['pf']:.2f} | "
          f"WR={win_rate:.1f}% | G/P={rr_ratio:.2f}x | "
          f"Trades={results.get('n_trades','?')} | Flat={results.get('flat_pct',0):.0f}%")

    return results, df_prices


if __name__ == "__main__":
    run_historical_backtest()
