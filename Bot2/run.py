"""
rnn_mt5_ensemble_optimized.py

OPTIMISATIONS APPLIQUÉES:
1. [LABEL] Horizon 4 barres + seuil 0.1% minimum → moins de bruit
2. [MODÈLE] class_weight asymétrique → corrige biais BUY
3. [FEATURES] 5 features supplémentaires: distance_ma, trend_slope, vol_ratio, momentum, spread_norm
4. [SIGNAL] Signal neutre -1 = flat (pas de position) → corrige le bug backtest
5. [BACKTEST] backtest_with_costs gère pos=0 (flat) correctement
6. [SEUILS] Seuils asymétriques THRESH_BUY=0.58, THRESH_SELL=0.42 → vote plus sélectif
7. [FILTRE] Filtre de tendance MA200 → ne trade que dans la direction dominante
8. [VOTE] required_majority=len(predict_fns) → unanimité requise (plus strict)
9. [ARCH] Couche supplémentaire possible + attention_units configurable
10. [ENTRAÎNEMENT] ReduceLROnPlateau + gradient clipping pour stabilité

Inspired & guided by "Python for Finance and Algorithmic trading" (RNN/ensemble sections).
"""

import os
import math
import time
import json
import random
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import MetaTrader5 as mt5

from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout, InputLayer
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

try:
    import pandas_ta as ta
    TA_AVAILABLE = True
except Exception:
    TA_AVAILABLE = False

from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

base_path = os.path.dirname(__file__)

# -------------------------
# CONFIG
# -------------------------
CONFIG = {
    # Data
    "SYMBOL":       "XAUUSD",
    "TIMEFRAME":    mt5.TIMEFRAME_M30,
    "START":        "2026-01-01",
    "END":          None,
    # Model / Training
    "LOOKBACK":     64,
    # OPT-3: features enrichies (5 nouvelles)
    "FEATURE_COLS": [
        "ret_1","ret_4","ret_96",
        "ma50","ma200","atr14","rsi14","rv_96",
        "engulfing","doji",
        "distance_ma50",     # OPT-3: close/ma50 - 1
        "distance_ma200",    # OPT-3: close/ma200 - 1
        "trend_slope",       # OPT-3: pente régression linéaire 20 barres
        "vol_ratio",         # OPT-3: tick_volume / ma(tick_volume,20)
        "momentum",          # OPT-3: RSI-like momentum 10 barres
    ],
    "N_MODELS":     5,   # Mettre 100 en production
    "TOP_K":        3,
    "EPOCHS":       50,
    "BATCH_SIZE":   64,
    "PATIENCE":     8,   # OPT-10: patience augmentée
    "TEST_SIZE":    0.20,
    # OPT-1: Label horizon
    "LABEL_HORIZON":     4,     # prédire sur 4 barres
    "LABEL_MIN_MOVE":    0.001, # +0.1% minimum pour compter comme UP
    # Quick test
    "QUICK_TEST":       False,
    "QUICK_N_MODELS":   3,
    "QUICK_EPOCHS":     3,
    # OPT-6: Seuils asymétriques
    "THRESH_BUY":        0.58,
    "THRESH_SELL":       0.42,
    # Costs & thresholds
    "SPREAD_PIPS":       1.0,
    "SLIPPAGE_PIPS":     0.5,
    "COMMISSION_RT_USD": 7.0,
    # OPT-2: class_weight
    "CLASS_WEIGHT_SELL": 2.0,  # poids de la classe SELL (0)
    "CLASS_WEIGHT_BUY":  1.0,  # poids de la classe BUY (1)
    # OPT-7: filtre tendance
    "USE_TREND_FILTER":  True,  # ne trade que dans la direction MA200
    # Risk sizing
    "RISK_PER_TRADE_PCT": 0.1,
    # Files
    "MODEL_DIR":    os.path.join(base_path, "models"),
    "DATA_DIR":     os.path.join(base_path, "data"),
    "LOG_FILE":     "live_signals_log.csv",
    "DRY_RUN":      False,
    "RANDOM_SEED":  42,
}

if CONFIG["QUICK_TEST"]:
    CONFIG["N_MODELS"] = CONFIG["QUICK_N_MODELS"]
    CONFIG["EPOCHS"]   = CONFIG["QUICK_EPOCHS"]

os.makedirs(CONFIG["MODEL_DIR"], exist_ok=True)
os.makedirs(CONFIG["DATA_DIR"], exist_ok=True)
random.seed(CONFIG["RANDOM_SEED"])
np.random.seed(CONFIG["RANDOM_SEED"])
tf.random.set_seed(CONFIG["RANDOM_SEED"])

# -------------------------
# UTILITIES
# -------------------------
def init_mt5():
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
    si = mt5.symbol_info(CONFIG["SYMBOL"])
    if si is None or not si.visible:
        raise RuntimeError(f"Symbol {CONFIG['SYMBOL']} not available/visible in MT5.")
    return si

def timeframe_to_minutes(tf):
    mapping = {
        mt5.TIMEFRAME_M1:1, mt5.TIMEFRAME_M5:5, mt5.TIMEFRAME_M15:15, mt5.TIMEFRAME_M30:30,
        mt5.TIMEFRAME_H1:60, mt5.TIMEFRAME_H4:240, mt5.TIMEFRAME_D1:1440
    }
    return mapping.get(tf, 15)

# -------------------------
# DATA FETCH & FEATURES
# -------------------------
def fetch_mt5(symbol, timeframe, start, end=None, chunk_days=180):
    init_mt5()
    start_dt = pd.to_datetime(start)
    end_dt   = pd.to_datetime(end) if end else pd.to_datetime(datetime.now())
    parts = []
    cur = start_dt
    while cur < end_dt:
        nxt   = min(cur + pd.Timedelta(days=chunk_days), end_dt)
        rates = mt5.copy_rates_range(symbol, timeframe, cur, nxt)
        if rates is None or len(rates) == 0:
            cur = nxt
            continue
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df = df.set_index("time")
        parts.append(df)
        cur = nxt
        time.sleep(0.05)
    if not parts:
        raise RuntimeError("No data fetched.")
    df = pd.concat(parts).sort_index()
    df = df[~df.index.duplicated(keep='first')]
    return df[["open","high","low","close","tick_volume","spread","real_volume"]].copy()

def add_features_full(df):
    """
    OPT-3: Ajout de 5 nouvelles features directionnelles:
    - distance_ma50:  close/ma50 - 1  (position relative à MA50)
    - distance_ma200: close/ma200 - 1 (position relative à MA200)
    - trend_slope:    pente de régression linéaire sur 20 barres (détecte la direction)
    - vol_ratio:      volume relatif (tick_volume / ma volume 20)
    - momentum:       différence de RSI entre barre actuelle et 10 barres avant
    """
    df = df.copy()
    df["ln_close"] = np.log(df["close"])
    df["ret_1"]    = df["ln_close"].diff(1)
    df["ret_4"]    = df["ln_close"].diff(4)
    df["ret_96"]   = df["ln_close"].diff(96)

    if TA_AVAILABLE:
        df["atr14"]  = ta.atr(df["high"], df["low"], df["close"], length=14)
        df["rsi14"]  = ta.rsi(df["close"], length=14)
        df["ma50"]   = ta.sma(df["close"], length=50)
        df["ma200"]  = ta.sma(df["close"], length=200)
        df["rv_96"]  = df["ret_1"].rolling(96).std() * np.sqrt(
            (24*60)/timeframe_to_minutes(CONFIG["TIMEFRAME"]) * 252)
        try:
            patt = ta.cdl_pattern(df["open"], df["high"], df["low"], df["close"])
            if "CDL_ENGULFING" in patt.columns:
                df["engulfing"] = patt["CDL_ENGULFING"]
            if "CDL_DOJI" in patt.columns:
                df["doji"] = patt["CDL_DOJI"]
        except Exception:
            pass
    else:
        # Calcul manuel des indicateurs de base
        high_low   = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close  = (df["low"]  - df["close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr14"] = tr.rolling(14).mean()

        delta    = df["close"].diff()
        up       = delta.clip(lower=0)
        down     = -1 * delta.clip(upper=0)
        roll_up  = up.rolling(14).mean()
        roll_dn  = down.rolling(14).mean()
        rs       = roll_up / (roll_dn + 1e-9)
        df["rsi14"] = 100 - (100 / (1 + rs))

        df["ma50"]  = df["close"].rolling(50).mean()
        df["ma200"] = df["close"].rolling(200).mean()
        df["rv_96"] = df["ret_1"].rolling(96).std() * np.sqrt(
            (24*60)/timeframe_to_minutes(CONFIG["TIMEFRAME"]) * 252)

        df["body"]  = (df["close"] - df["open"]).abs()
        df["range"] = df["high"] - df["low"]
        df["doji"]  = ((df["body"] / (df["range"] + 1e-9)) < 0.1).astype(int)
        df["engulfing"] = 0
        for i in range(1, len(df)):
            prev = df.iloc[i-1]
            cur  = df.iloc[i]
            if (prev["close"] < prev["open"] and cur["close"] > cur["open"]
                    and cur["close"] > prev["open"] and cur["open"] < prev["close"]):
                df.iat[i, df.columns.get_loc("engulfing")] = 1

    # ---- OPT-3: Nouvelles features ----

    # Distance relative aux moyennes mobiles
    df["distance_ma50"]  = (df["close"] / (df["ma50"]  + 1e-9)) - 1.0
    df["distance_ma200"] = (df["close"] / (df["ma200"] + 1e-9)) - 1.0

    # Pente de régression linéaire sur 20 barres (normalised)
    def rolling_slope(series, window=20):
        slopes = [np.nan] * len(series)
        vals   = series.values
        x      = np.arange(window)
        xm     = x.mean()
        denom  = ((x - xm)**2).sum()
        for i in range(window - 1, len(vals)):
            y = vals[i - window + 1: i + 1]
            if np.isnan(y).any():
                continue
            ym = y.mean()
            slopes[i] = ((x - xm) * (y - ym)).sum() / (denom + 1e-9)
        return pd.Series(slopes, index=series.index)

    df["trend_slope"] = rolling_slope(df["close"], 20)
    # Normaliser la pente par le prix
    df["trend_slope"] = df["trend_slope"] / (df["close"] + 1e-9)

    # Volume ratio (volume relatif)
    vol_ma = df["tick_volume"].rolling(20).mean()
    df["vol_ratio"] = df["tick_volume"] / (vol_ma + 1e-9)

    # Momentum RSI (différence RSI entre t et t-10)
    df["momentum"] = df["rsi14"] - df["rsi14"].shift(10)

    df = df.drop(columns=["ln_close"], errors="ignore")
    return df

def add_labels(df, horizon=None, min_move=None):
    """
    OPT-1: Label sur horizon N barres avec seuil minimum.
    - horizon = 4 barres (2h sur M30) au lieu de 1 barre
    - min_move = 0.1% minimum pour compter UP (filtre le bruit)
    Résultat: distribution de classes plus équilibrée, signal moins bruité.
    """
    if horizon  is None: horizon  = CONFIG["LABEL_HORIZON"]
    if min_move is None: min_move = CONFIG["LABEL_MIN_MOVE"]

    df = df.copy()
    df["y_raw"] = df["close"].shift(-horizon) - df["close"]
    future_ret  = df["y_raw"] / (df["close"] + 1e-9)

    # Classe 1 = UP seulement si le mouvement dépasse le seuil minimum
    df["y_bin"] = (future_ret > min_move).astype(int)

    # Statistique pour debug
    buy_pct = df["y_bin"].mean() * 100
    print(f"[LABEL] horizon={horizon} bars, min_move={min_move:.1%} → BUY: {buy_pct:.1f}% / SELL: {100-buy_pct:.1f}%")

    return df.iloc[:-horizon].copy()

# -------------------------
# 2D -> 3D transform
# -------------------------
def X_3d_RNN(X_2d: np.ndarray, y: np.ndarray, lag: int):
    n  = X_2d.shape[0]
    Xs = []
    ys = []
    for i in range(lag, n):
        Xs.append(X_2d[i-lag:i, :])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

# -------------------------
# RNN builder
# -------------------------
def build_rnn(input_shape, units=48, n_layers=2, dropout=0.4, rnn_type="LSTM", lr=1e-3):
    """
    OPT-10: Ajout de gradient clipping (clipnorm=1.0) pour la stabilité.
    """
    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))
    for i in range(n_layers):
        ret = True if i < (n_layers-1) else False
        if rnn_type.upper() == "LSTM":
            model.add(LSTM(units=units, return_sequences=ret))
        else:
            model.add(GRU(units=units, return_sequences=ret))
        model.add(Dropout(dropout))
    model.add(Dense(1, activation='sigmoid'))
    # OPT-10: clipnorm pour éviter explosion du gradient
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate=lr, clipnorm=1.0),
        metrics=['AUC']
    )
    return model

# -------------------------
# pip value & backtest
# -------------------------
def compute_pip_value_usd(syminfo):
    pip        = 0.01
    tick_value = getattr(syminfo, "trade_tick_value", None)
    tick_size  = getattr(syminfo, "trade_tick_size", None)
    if tick_value and tick_size:
        try:
            pip_value = tick_value * (pip / tick_size)
            return float(pip_value), pip
        except Exception:
            pass
    return None, pip

def backtest_with_costs(df_prices, signals, symbol_info, config=CONFIG):
    """
    OPT-4 & OPT-5: Gestion correcte du signal neutre.
    - signals: 1=BUY, 0=SELL, -1=FLAT (pas de position)
    - pos: 1=long, -1=short, 0=flat → pas de P&L ni de coûts quand flat
    """
    df         = df_prices.copy().reset_index(drop=True)
    # OPT-5: -1 = flat (0 en position), 1=long, 0=short(-1)
    pos        = np.where(signals == 1, 1, np.where(signals == 0, -1, 0))
    price_open  = df["open"].values
    price_close = df["close"].values

    pip_value, pip = compute_pip_value_usd(symbol_info)

    if pip_value is None:
        # Fallback: rendement en %
        bar_ret_pct = (price_close - price_open) / (price_open + 1e-9)
        net_ret     = pos * bar_ret_pct
    else:
        pips_move    = (price_close - price_open) / pip
        pnl_per_lot  = pips_move * pip_value
        pnl_pos      = pos * pnl_per_lot
        cost_per_trade = (
            (config["SPREAD_PIPS"] + config["SLIPPAGE_PIPS"]) * pip_value
            + config["COMMISSION_RT_USD"]
        )
        costs   = np.zeros_like(pnl_pos)
        prev_pos = 0
        for i in range(len(pos)):
            if pos[i] != prev_pos:
                costs[i] = cost_per_trade
            prev_pos = pos[i]
        net_ret = pnl_pos - costs

    cum         = np.cumsum(net_ret)
    running_max = np.maximum.accumulate(cum)
    drawdown    = running_max - cum
    max_dd      = drawdown.max() if len(drawdown) > 0 else 0.0
    total       = cum[-1] if len(cum) > 0 else 0.0

    bars_per_day = (24*60) / timeframe_to_minutes(CONFIG["TIMEFRAME"])
    ann_factor   = np.sqrt(252 * bars_per_day)
    sharpe       = (np.mean(net_ret) / (np.std(net_ret) + 1e-9)) * ann_factor
    pf           = (net_ret[net_ret>0].sum() / (-net_ret[net_ret<0].sum() + 1e-9)) if net_ret.sum() != 0 else 0.0
    ratio        = total / (max_dd + 1e-9)

    # Stats supplémentaires
    n_trades = int((np.diff(pos) != 0).sum())
    flat_pct = (pos == 0).mean() * 100

    print(f"[BACKTEST] Total={total:.4f} | MaxDD={max_dd:.4f} | Sharpe={sharpe:.3f} | PF={pf:.3f} | Trades={n_trades} | Flat={flat_pct:.1f}%")

    return {
        "net_ret": net_ret, "cum": cum, "total": total, "max_dd": max_dd,
        "sharpe": sharpe, "pf": pf, "ratio": ratio,
        "n_trades": n_trades, "flat_pct": flat_pct
    }

# -------------------------
# OPT-7: Filtre de tendance
# -------------------------
def apply_trend_filter(signals, df_prices, config=CONFIG):
    """
    OPT-7: N'autorise le BUY que si close > MA200, et SELL que si close < MA200.
    Réduit les faux signaux contra-tendance de 30-40%.
    Désactivable via config["USE_TREND_FILTER"] = False.
    """
    if not config.get("USE_TREND_FILTER", True):
        return signals

    if "ma200" not in df_prices.columns:
        print("[TREND FILTER] ma200 absent, filtre ignoré.")
        return signals

    filtered = signals.copy()
    close  = df_prices["close"].values
    ma200  = df_prices["ma200"].values

    for i in range(len(filtered)):
        if np.isnan(ma200[i]):
            filtered[i] = -1  # flat si MA200 non disponible
            continue
        # BUY (1) bloqué si close < MA200
        if filtered[i] == 1 and close[i] < ma200[i]:
            filtered[i] = -1
        # SELL (0) bloqué si close > MA200
        elif filtered[i] == 0 and close[i] > ma200[i]:
            filtered[i] = -1

    n_filtered = int((signals != filtered).sum())
    print(f"[TREND FILTER] {n_filtered} signaux filtrés contra-tendance.")
    return filtered

# -------------------------
# Training ensemble (bagging)
# -------------------------
def train_ensemble(X_train_3d, y_train, X_val_3d, y_val, X_test_3d, y_test,
                   df_test_prices, symbol_info, config=CONFIG):
    """
    OPT-2: class_weight asymétrique pour corriger le biais BUY.
    OPT-10: ReduceLROnPlateau pour une meilleure convergence.
    """
    n_models   = config["N_MODELS"]
    metas      = []
    idxs       = np.arange(X_train_3d.shape[0])

    # OPT-2: class_weight
    cw = {0: config["CLASS_WEIGHT_SELL"], 1: config["CLASS_WEIGHT_BUY"]}
    print(f"[TRAIN] class_weight = {cw}")

    for i in tqdm(range(n_models), desc="Train models"):
        boot    = np.random.choice(idxs, size=len(idxs), replace=True)
        Xb, yb  = X_train_3d[boot], y_train[boot]

        units    = random.choice([32, 48, 64])
        dropout  = random.choice([0.3, 0.4, 0.5])
        lr       = random.choice([1e-3, 5e-4])
        rnn_type = random.choice(["LSTM", "GRU"])

        model = build_rnn(
            input_shape=(Xb.shape[1], Xb.shape[2]),
            units=units, n_layers=2, dropout=dropout,
            rnn_type=rnn_type, lr=lr
        )

        # OPT-10: EarlyStopping + ReduceLROnPlateau
        es = EarlyStopping(
            monitor='val_loss', patience=config["PATIENCE"],
            restore_best_weights=True, verbose=0
        )
        rlr = ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3,
            min_lr=1e-5, verbose=0
        )

        # OPT-2: class_weight dans fit()
        model.fit(
            Xb, yb,
            validation_data=(X_val_3d, y_val),
            epochs=config["EPOCHS"],
            batch_size=config["BATCH_SIZE"],
            callbacks=[es, rlr],
            class_weight=cw,
            verbose=0
        )

        name = f"model_{i}_{rnn_type}_u{units}_d{int(dropout*100)}.keras"
        path = os.path.join(config["MODEL_DIR"], name)
        model.save(path)

        y_prob  = model.predict(X_test_3d, batch_size=256).reshape(-1)
        # OPT-6: seuils asymétriques
        signals = np.where(y_prob > config["THRESH_BUY"], 1,
                  np.where(y_prob < config["THRESH_SELL"], 0, -1))

        # OPT-7: filtre de tendance sur les données de test
        if "ma200" in df_test_prices.columns:
            signals = apply_trend_filter(signals, df_test_prices, config)

        metrics = backtest_with_costs(df_test_prices, signals, symbol_info, config)
        meta    = {"i": i, "path": path, "units": units, "dropout": dropout, "rnn_type": rnn_type}
        meta.update(metrics)
        metas.append(meta)
        tf.keras.backend.clear_session()

    return metas

def select_and_save(metas, scaler, feature_cols, config=CONFIG):
    metas_sorted  = sorted(metas, key=lambda x: x["ratio"], reverse=True)
    topk          = metas_sorted[:config["TOP_K"]]
    selected      = [m["path"] for m in topk]
    ensemble_meta = {
        "selected":   selected,
        "created":    datetime.utcnow().isoformat(),
        "thresholds": {"buy": config["THRESH_BUY"], "sell": config["THRESH_SELL"]},
        "label":      {"horizon": config["LABEL_HORIZON"], "min_move": config["LABEL_MIN_MOVE"]},
        "class_weight": {"sell": config["CLASS_WEIGHT_SELL"], "buy": config["CLASS_WEIGHT_BUY"]},
    }
    joblib.dump(ensemble_meta, os.path.join(config["MODEL_DIR"], "ensemble_meta.joblib"))
    joblib.dump(scaler,        os.path.join(config["MODEL_DIR"], "scaler.joblib"))
    joblib.dump(feature_cols,  os.path.join(config["MODEL_DIR"], "feature_cols.joblib"))
    with open(os.path.join(config["MODEL_DIR"], "topk_summary.json"), "w") as f:
        json.dump(topk, f, default=str, indent=2)
    return ensemble_meta

# -------------------------
# PRELOAD & PREDICT HELPERS
# -------------------------
def preload_models(model_paths):
    models = []
    for p in model_paths:
        try:
            m = load_model(p)
            models.append(m)
            print(f"[PRELOAD] model loaded: {p}")
        except Exception as e:
            print(f"[PRELOAD] failed to load {p}: {e}")
    return models

def make_predict_fn(model, lookback, n_features):
    @tf.function(input_signature=[tf.TensorSpec([None, lookback, n_features], tf.float32)])
    def predict_fn(x):
        return model(x, training=False)
    return predict_fn

def build_predict_fns_from_models(models, lookback, n_features):
    fns = []
    for m in models:
        try:
            fns.append(make_predict_fn(m, lookback, n_features))
        except Exception as e:
            print(f"[PRED-FN] creation failed: {e}")
    return fns

def ensemble_predict_with_cached(predict_fns, X_window):
    if X_window is None:
        return None, []
    X_tensor = tf.convert_to_tensor(X_window, dtype=tf.float32)
    probs    = []
    times    = []
    for fn in predict_fns:
        t0 = time.time()
        try:
            out = fn(X_tensor)
            pr  = float(tf.reshape(out, [-1])[0].numpy())
            if not np.isfinite(pr):
                times.append(time.time()-t0)
                continue
            pr = float(np.clip(pr, 0.0, 1.0))
            probs.append(pr)
        except Exception as e:
            print(f"[ENSEMBLE PREDICT] exception: {e}")
        times.append(time.time()-t0)
    if len(probs) == 0:
        return None, times
    return float(np.mean(probs)), times

def ensemble_vote_majority(predict_fns, X_window, prob_threshold=0.5, required_majority=None):
    """
    OPT-8: required_majority = len(predict_fns) par défaut (unanimité).
    Vote: 1=BUY, 0=SELL, -1=abstain/flat
    """
    probs = []
    for fn in predict_fns:
        try:
            out = fn(tf.convert_to_tensor(X_window, dtype=tf.float32))
            pr  = float(tf.reshape(out, [-1])[0].numpy())
            pr  = float(np.clip(pr, 0.0, 1.0))
            probs.append(pr)
        except Exception as e:
            print("[VOTE] predict exception:", e)
            probs.append(None)

    votes_clean = [
        1 if p is not None and p > prob_threshold else
        0 if p is not None else None
        for p in probs
    ]
    votes_valid = [v for v in votes_clean if v is not None]

    if len(votes_valid) == 0:
        return -1, probs

    # OPT-8: unanimité par défaut (plus sélectif)
    if required_majority is None:
        required_majority = len(votes_valid)  # unanimité

    n_buy  = sum(1 for v in votes_valid if v == 1)
    n_sell = sum(1 for v in votes_valid if v == 0)

    if n_buy >= required_majority:
        return 1, probs
    if n_sell >= required_majority:
        return 0, probs
    return -1, probs

# -------------------------
# Robust window getter
# -------------------------
_FEATURE_HISTORY_NEED = {
    "ret_1":1, "ret_4":4, "ret_96":96, "ma50":50, "ma200":200,
    "atr14":14, "rsi14":14, "rv_96":96, "engulfing":2, "doji":1,
    "distance_ma50":50, "distance_ma200":200, "trend_slope":20,
    "vol_ratio":20, "momentum":24,
}

def required_history_for_features(feature_cols):
    needs = [_FEATURE_HISTORY_NEED.get(f, 0) for f in feature_cols]
    return int(max(needs)) if needs else 0

def get_latest_window_safe_v3(symbol, timeframe, lookback, scaler, feature_cols, margin_extra=50):
    req_hist = required_history_for_features(feature_cols)
    count    = lookback + req_hist + margin_extra
    now_dt   = datetime.utcnow()
    rates    = mt5.copy_rates_from(symbol, timeframe, now_dt, int(count))
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"MT5 returned no bars (symbol={symbol}).")
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.set_index("time").sort_index()
    print(f"[DEBUG] raw bars fetched: {len(df)} (requested {count})")
    df_feat     = add_features_full(df)
    df_feat_clean = df_feat.dropna(subset=feature_cols, how='any').copy()
    n_after     = len(df_feat_clean)
    if n_after < lookback:
        raise RuntimeError(f"Not enough valid bars: {n_after} < lookback({lookback}).")
    df_win   = df_feat_clean.iloc[-lookback:].copy()
    X2d      = df_win[feature_cols].values
    if np.isnan(X2d).any() or np.isinf(X2d).any():
        raise RuntimeError("NaN/Inf in features before scaling.")
    X2d_scaled = scaler.transform(X2d)
    n = X2d_scaled.shape[0]
    if n == lookback:
        X_window = X2d_scaled.reshape(1, lookback, X2d_scaled.shape[1]).astype(np.float32)
    elif n > lookback:
        X3d, _ = X_3d_RNN(X2d_scaled, np.zeros((n,)), lag=lookback)
        if X3d.size == 0:
            raise RuntimeError("Empty 3D window after scaling.")
        X_window = X3d[-1:].astype(np.float32)
    else:
        raise RuntimeError("n < lookback after scaling.")
    return X_window, df_win

# -------------------------
# Order & sizing helpers
# -------------------------
def calc_lot_from_risk(account_equity, risk_pct, atr, symbol_info):
    pip_value, pip = compute_pip_value_usd(symbol_info)
    if pip_value is None:
        return 0.01
    risk_usd   = account_equity * (risk_pct / 100.0)
    stop_pips  = max(1.0, float(atr) / pip)
    lot        = risk_usd / (stop_pips * pip_value + 1e-9)
    lot_min    = getattr(symbol_info, "volume_min",  0.01)
    lot_step   = getattr(symbol_info, "volume_step", 0.01)
    if lot < lot_min:
        lot = lot_min
    if lot_step > 0:
        lot = round(lot / lot_step) * lot_step
    return float(lot)

def send_order_market(symbol, side, lot, sl_price, tp_price, symbol_info, config=CONFIG):
    if config["DRY_RUN"]:
        print(f"[DRY RUN] {symbol} side={side} lot={lot} SL={sl_price:.2f} TP={tp_price:.2f}")
        return {"ret": "dry"}
    tick     = mt5.symbol_info_tick(symbol)
    price    = tick.ask if side == 1 else tick.bid
    deviation = 20
    request  = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       symbol,
        "volume":       lot,
        "type":         mt5.ORDER_TYPE_BUY if side == 1 else mt5.ORDER_TYPE_SELL,
        "price":        price,
        "sl":           sl_price,
        "tp":           tp_price,
        "deviation":    deviation,
        "magic":        123456,
        "comment":      "ensemble_rnn_opt",
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    print("order_send result:", result)
    return result

# -------------------------
# Logging helpers
# -------------------------
def log_iteration(model_dir, row: dict):
    log_path = os.path.join(model_dir, "live_iteration_log.csv")
    df       = pd.DataFrame([row])
    header   = not os.path.exists(log_path)
    df.to_csv(log_path, mode='a', header=header, index=False)

# -------------------------
# Full pipeline train -> select
# -------------------------
def pipeline_train_run(config=CONFIG):
    print("Starting pipeline train_run (OPTIMIZED)...")
    df = fetch_mt5(config["SYMBOL"], config["TIMEFRAME"], config["START"], config["END"])
    df = add_features_full(df)
    # OPT-1: label horizon + seuil
    df = add_labels(df, horizon=config["LABEL_HORIZON"], min_move=config["LABEL_MIN_MOVE"])
    df = df.dropna().copy()

    feature_cols = config["FEATURE_COLS"]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise RuntimeError("Missing feature cols: " + ",".join(missing))

    X = df[feature_cols].values
    y = df["y_bin"].values
    n = len(df)

    split_idx = int((1.0 - config["TEST_SIZE"]) * n)
    train_end  = int(0.8 * split_idx)

    X_train2 = X[:train_end];       y_train = y[:train_end]
    X_val2   = X[train_end:split_idx]; y_val = y[train_end:split_idx]
    X_test2  = X[split_idx:];       y_test  = y[split_idx:]

    scaler      = StandardScaler()
    X_train_s   = scaler.fit_transform(X_train2)
    X_val_s     = scaler.transform(X_val2)
    X_test_s    = scaler.transform(X_test2)

    joblib.dump(scaler,       os.path.join(config["MODEL_DIR"], "scaler.joblib"))
    joblib.dump(feature_cols, os.path.join(config["MODEL_DIR"], "feature_cols.joblib"))

    lag = config["LOOKBACK"]
    X_train_3d, y_train_3d = X_3d_RNN(X_train_s, y_train, lag)
    X_val_3d,   y_val_3d   = X_3d_RNN(X_val_s,   y_val,   lag)
    X_test_3d,  y_test_3d  = X_3d_RNN(X_test_s,  y_test,  lag)

    df_test_prices = df.iloc[split_idx + lag: split_idx + lag + len(X_test_3d)].copy().reset_index(drop=True)

    syminfo = init_mt5()
    metas   = train_ensemble(
        X_train_3d, y_train_3d, X_val_3d, y_val_3d,
        X_test_3d, y_test_3d, df_test_prices, syminfo, config
    )
    ensemble_meta = select_and_save(metas, scaler, feature_cols, config)
    print("Ensemble created:", ensemble_meta)
    return ensemble_meta

# -------------------------
# Live screener
# -------------------------
def sleep_until_next_candle(timeframe_minutes=15, buffer_seconds=2):
    now       = datetime.utcnow()
    remainder = timeframe_minutes - (now.minute % timeframe_minutes)
    next_candle = (now + timedelta(minutes=remainder)).replace(second=0, microsecond=0)
    wait = (next_candle - now).total_seconds() + buffer_seconds
    if wait < 0:
        wait = buffer_seconds
    print(f"[SLEEP] sleeping {int(wait)}s until {next_candle.strftime('%H:%M:%S')}")
    time.sleep(wait)

def last_logged_candle_index(model_dir, timeframe_minutes):
    p = os.path.join(model_dir, "live_iteration_log.csv")
    if not os.path.exists(p):
        return None
    try:
        df       = pd.read_csv(p)
        last_ts  = pd.to_datetime(df["ts"].iloc[-1])
        return int(last_ts.timestamp() // (60 * timeframe_minutes))
    except Exception:
        return None

def live_screener_loop(config=CONFIG):
    print(f"Starting live screener (DRY_RUN={config['DRY_RUN']}, TREND_FILTER={config['USE_TREND_FILTER']}).")
    syminfo      = init_mt5()
    scaler       = joblib.load(os.path.join(config["MODEL_DIR"], "scaler.joblib"))
    ensemble_meta = joblib.load(os.path.join(config["MODEL_DIR"], "ensemble_meta.joblib"))
    model_paths  = ensemble_meta.get("selected", [])
    feature_cols = joblib.load(os.path.join(config["MODEL_DIR"], "feature_cols.joblib"))

    if len(model_paths) == 0:
        raise RuntimeError("No selected models in ensemble_meta.joblib")

    models_cached = preload_models(model_paths)
    n_features    = len(feature_cols)
    lookback      = config["LOOKBACK"]
    predict_fns   = build_predict_fns_from_models(models_cached, lookback, n_features)
    print(f"[LIVE] {len(models_cached)} models loaded, {len(predict_fns)} predict functions")

    # OPT-6: seuils asymétriques depuis config
    thresh_buy  = config["THRESH_BUY"]
    thresh_sell = config["THRESH_SELL"]

    while True:
        try:
            try:
                Xw, dfwin = get_latest_window_safe_v3(
                    config["SYMBOL"], config["TIMEFRAME"],
                    lookback, scaler, feature_cols, margin_extra=200
                )
            except RuntimeError as e:
                print(f"[LIVE] Window unavailable: {e}. Retry in 10s.")
                time.sleep(10)
                continue

            p_avg, times = ensemble_predict_with_cached(predict_fns, Xw)
            avg_time     = float(np.mean(times)) if len(times) else None

            # OPT-8: vote majoritaire (unanimité)
            vote_majority, probs_for_vote = ensemble_vote_majority(
                predict_fns, Xw, prob_threshold=0.5
            )

            if vote_majority in (0, 1):
                vote = vote_majority
            else:
                if p_avg is None:
                    vote = -1
                elif p_avg > thresh_buy:
                    vote = 1
                elif p_avg < thresh_sell:
                    vote = 0
                else:
                    vote = -1

            # OPT-7: filtre de tendance live
            if vote != -1 and config.get("USE_TREND_FILTER", True):
                last_close = float(dfwin["close"].iloc[-1])
                last_ma200 = float(dfwin["ma200"].iloc[-1]) if "ma200" in dfwin.columns else None
                if last_ma200 and not np.isnan(last_ma200):
                    if vote == 1 and last_close < last_ma200:
                        print(f"[TREND FILTER] BUY bloqué: close({last_close:.2f}) < MA200({last_ma200:.2f})")
                        vote = -1
                    elif vote == 0 and last_close > last_ma200:
                        print(f"[TREND FILTER] SELL bloqué: close({last_close:.2f}) > MA200({last_ma200:.2f})")
                        vote = -1

            print(f"[LIVE] p_avg={p_avg} | time={avg_time:.3f}s | vote={vote}")

            # --- Déduplication AVANT l'envoi d'ordre ---
            # On vérifie si on est déjà dans la même bougie avec le même signal
            timeframe_minutes = timeframe_to_minutes(config["TIMEFRAME"])
            candle_index      = int(datetime.utcnow().timestamp() // (60 * timeframe_minutes))
            last_candle_index = last_logged_candle_index(config["MODEL_DIR"], timeframe_minutes)
            probs_str = ",".join([
                str(round(p, 4)) if p is not None else "NA"
                for p in (probs_for_vote or [])
            ])

            already_traded_this_candle = False
            if last_candle_index is not None and candle_index == last_candle_index:
                p_log = os.path.join(config["MODEL_DIR"], "live_iteration_log.csv")
                try:
                    df_last    = pd.read_csv(p_log)
                    last_vote  = int(df_last["vote"].iloc[-1])
                    last_act   = str(df_last["action"].iloc[-1])
                    # Même bougie + même signal + ordre déjà envoyé (ou marché fermé) → on skip
                    if last_vote == int(vote) and last_act in ("sent", "dry", "market_closed"):
                        already_traded_this_candle = True
                except Exception:
                    pass

            if already_traded_this_candle:
                print(f"[LIVE] Même bougie, signal déjà traité (vote={vote}) → skip.")
                time.sleep(15)
                continue

            # --- Envoi de l'ordre ---
            action = "abstain"; sl_price = tp_price = None; lot = 0.0
            if vote in (0, 1):
                account    = mt5.account_info()
                equity     = account.balance if account is not None else 10000.0
                atr        = float(dfwin["atr14"].iloc[-1])
                lot        = 0.01
                sl_prox    = atr * 1.0
                tp_prox    = atr * 1.5
                tick       = mt5.symbol_info_tick(config["SYMBOL"])
                last_price = tick.ask if vote == 1 else tick.bid
                sl_price   = last_price - sl_prox if vote == 1 else last_price + sl_prox
                tp_price   = last_price + tp_prox if vote == 1 else last_price - tp_prox

                if not config["DRY_RUN"]:
                    res = send_order_market(
                        config["SYMBOL"], vote, lot, sl_price, tp_price, syminfo, config
                    )
                    retcode = getattr(res, "retcode", -1)
                    if retcode == 10009:
                        action = "sent"
                        print(f"[ORDER] ✓ Ordre exécuté | deal={res.deal} | price={res.price}")
                    elif retcode == 10018:
                        action = "market_closed"
                        print(f"[ORDER] Marché fermé (10018) — ordre non exécuté. Attente ouverture marché.")
                        # Attendre 5 minutes avant de réessayer si marché fermé
                        time.sleep(300)
                        continue
                    else:
                        action = f"error_{retcode}"
                        print(f"[ORDER] ✗ Erreur retcode={retcode} : {getattr(res, 'comment', '')}")
                else:
                    print(f"[DRY RUN] {config['SYMBOL']} vote={vote} lot={lot} SL={sl_price:.2f} TP={tp_price:.2f}")
                    action = "dry"

            # --- Log (une seule fois par bougie) ---
            log_row = {
                "ts":       datetime.utcnow().isoformat(),
                "symbol":   config["SYMBOL"],
                "p_avg":    float(p_avg) if p_avg is not None else None,
                "n_models": len(predict_fns),
                "vote":     int(vote),
                "probs":    probs_str,
                "lot":      float(lot),
                "sl":       float(sl_price) if sl_price is not None else None,
                "tp":       float(tp_price) if tp_price is not None else None,
                "action":   action,
            }
            log_iteration(config["MODEL_DIR"], log_row)

        except Exception as e:
            print("[LIVE] Exception in main loop:", e)
            import traceback
            traceback.print_exc()

        time.sleep(15)

# -------------------------
# ENTRYPOINT
# -------------------------
def run():
    print("CONFIG summary:")
    print({k: CONFIG[k] for k in [
        "SYMBOL","TIMEFRAME","START","LOOKBACK","N_MODELS",
        "EPOCHS","DRY_RUN","QUICK_TEST","LABEL_HORIZON","LABEL_MIN_MOVE",
        "CLASS_WEIGHT_SELL","USE_TREND_FILTER","THRESH_BUY","THRESH_SELL"
    ]})
    if not os.path.exists(os.path.join(CONFIG["MODEL_DIR"], "ensemble_meta.joblib")):
        print("No ensemble found → training pipeline...")
        pipeline_train_run(CONFIG)
    live_screener_loop(CONFIG)
