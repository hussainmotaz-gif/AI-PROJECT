from MT5 import *
import os
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import talib
import joblib
import time
from datetime import datetime
import pytz
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from backtesting import Backtest, Strategy

import warnings
warnings.filterwarnings("ignore")

m5: MT5 = MT5()

# ---------------------------
# 1. Configuration générale
# ---------------------------
TRADING_LIVE = True  # Si False, on n'exécute pas les ordres, on fais seulement l'entraînement
BACKTESTING = False
SYMBOL = "XAUUSD"
TIMEFRAME = mt5.TIMEFRAME_M15   # H1, M15, D1, etc.
#HISTORY_CANDLES = 5000
SEQ_LEN = 30            # longueur des séquences temporelles (lag)
# Séquence de 30 + la plus longue période d'indicateur (MACD slow = 26)
max_indicator_lookback = 50
LOOKBACK = SEQ_LEN + max_indicator_lookback
N_MODELS = 25          # nombre de modèles à entraîner pour le bagging
TOP_K = 3               # nombre de meilleurs modèles à retenir pour le vote
EPOCHS = 50
SEUIL_DYNAMIQUE = 0.5  # ← tu peux ajuster après test
# RISK_PER_TRADE = 1      # % de capital à risquer
SL_PIPS = 100
TP_PIPS = 150
LOT_SIZE = 0.01  # taille de lot fixe pour simplifier, à ajuster selon votre stratégie

base_path = os.path.dirname(__file__)  # Répertoire du script actuel
# Emplacement pour sauvegarder les modèles et le scaler
SCALER_PATH = os.path.join(base_path, "scaler.save")
MODEL_DIR = os.path.join(base_path, "models")

timezone = pytz.timezone("Etc/UTC")
# TRAINING_DATE_FROM = "2020-01-01 00:00"
# TRAINING_DATE_TO = "2024-12-31 23:59"
TRAINING_DATE_FROM = datetime(year=2020, month=1, day=1, hour=0, minute=0, second=0, tzinfo=timezone)
TRAINING_DATE_TO = datetime(year=2024, month=12, day=31, hour=23, minute=59, second=59, tzinfo=timezone)

# ---------------------------
# 2. Fonctions utilitaires
# ---------------------------
def initialize_mt5():
    if not mt5.initialize():
        raise RuntimeError(f"MT5 init failed: {mt5.last_error()}")
    print("MT5 initialized")


def fetch_data(symbol=SYMBOL, timeframe=TIMEFRAME, n=5000) -> pd.DataFrame:
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n)
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df

def fetch_data_range(symbol: str=SYMBOL, timeframe: int=TIMEFRAME, date_from: datetime=TRAINING_DATE_FROM, date_to: datetime=TRAINING_DATE_TO) -> pd.DataFrame:
    """
    Récupère les données OHLCV pour `symbol` et `timeframe`
    entre deux dates.
    Parameters
    ----------
    symbol : str
        Symbole tel que "XAUUSD", "EURUSD", etc.
    timeframe : int
        Constante MT5 (mt5.TIMEFRAME_M1, M5, M15, H1, …).
    date_from : str
        Date de début au format 'YYYY-MM-DD' ou 'YYYY-MM-DD HH:MM:SS'.
    date_to : str
        Date de fin   au format 'YYYY-MM-DD' ou 'YYYY-MM-DD HH:MM:SS'.
    Returns
    -------
    pd.DataFrame
        DataFrame indexé en datetime, colonnes
        ['open','high','low','close','tick_volume','spread','real_volume'].
    """
    # Conversion en datetime Python
    # dt_from = pd.to_datetime(date_from)
    # dt_to   = pd.to_datetime(date_to)
    # dt_from = datetime.strptime(date_from, '%Y-%m-%d %H:%M')
    # dt_to = datetime.strptime(date_to, '%Y-%m-%d %H:%M')

    # Appel MT5
    rates = mt5.copy_rates_range(symbol, timeframe, date_from, date_to)
    # Construction du DataFrame
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    # print(df_prices)
    # exit()
    df.set_index('time', inplace=True)
    return df

def add_technical_indicators(df):
    # Chapter 10 & 16: features engineering
    df['return'] = df['close'].pct_change()
    df['sma_20'] = talib.SMA(df['close'], timeperiod=20)
    df['rsi_14'] = talib.RSI(df['close'], timeperiod=14)
    df['macd'], df['macd_signal'], _ = talib.MACD(
        df['close'], fastperiod=12, slowperiod=26, signalperiod=9
    )
    df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    # df.dropna(inplace=True)
    return df

def generate_target(df, future_window=5, threshold=0.005):
    # Chapter 10: target labeling
    df['future_return'] = df['close'].shift(-future_window) / df['close'] - 1
    df['target'] = (df['future_return'] > threshold).astype(int)
    df.dropna(inplace=True)
    return df


def create_sequences(X, y, seq_len=SEQ_LEN):
    # Chapter 14.2.1: transformer 2D -> 3D
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len])
    return np.array(Xs), np.array(ys)


def build_model(input_shape):
    # Chapter 14.2.2: LSTM + Dropout + Dense sigmoid
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape, return_sequences=False))
    model.add(Dropout(0.2))               # éviter overfitting
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def evaluate_strategy(df_test, signals_col='signal'):
    # simple backtest: cumprod des retours
    df_test['strat_ret'] = df_test[signals_col].shift(1) * df_test['return']
    df_test['cumul_strat'] = (1 + df_test['strat_ret']).cumprod()
    df_test['buy_hold'] = (1 + df_test['return']).cumprod()
    total_ret = df_test['cumul_strat'].iloc[-1] - 1
    max_dd = (df_test['cumul_strat'].cummax() - df_test['cumul_strat']).max()
    ratio = total_ret / max_dd if max_dd>0 else np.nan
    return ratio


# def calculate_lot(balance, risk_pct=RISK_PER_TRADE, sl_pips=SL_PIPS, symbol=SYMBOL):
#     info = mt5.symbol_info(symbol)
#     tick_value = info.trade_tick_value
#     tick_size = info.trade_tick_size
#     pip_value = tick_value / tick_size
#     risk_usd = balance * (risk_pct/100)
#     lot = risk_usd / (sl_pips * pip_value)
#     return round(lot, 2)

# ---------------------------
# 3. Préparation des données
# ---------------------------
def prepare_dataset_for_train():
    initialize_mt5()
    df = fetch_data_range()
    df = add_technical_indicators(df)
    df = generate_target(df)

    # Ici on nettoie : on retire toutes les lignes qui n'ont pas
    # à la fois un target et toutes les features
    cols_to_keep = ['return','sma_20','rsi_14','macd','atr','target']
    df = df.dropna(subset=cols_to_keep)

    features = ['sma_20','rsi_14','macd','atr','return']  
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features].values)
    joblib.dump(scaler, SCALER_PATH)

    X, y = create_sequences(X_scaled, df['target'].values)
    split = int(0.8 * len(X))
    return X[:split], y[:split], X[split:], y[split:], df.iloc[-len(y):]

# ---------------------------
# 4. Entraînement et sélection par Bagging
# ---------------------------
def train_and_select():
    X_train, y_train, X_test, y_test, df_full = prepare_dataset_for_train()
    input_shape = (X_train.shape[1], X_train.shape[2])

    models = []
    scores = []

    # On isole la partie de df_full qui correspond aux X_test
    df_test = df_full.iloc[-len(y_test):].copy()

    # entraînement de N_MODELS modèles indépendants
    for i in range(N_MODELS):
        model = build_model(input_shape)
        es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        # model.fit(X_train, y_train, validation_data=(X_test,y_test),
        #           epochs=EPOCHS, batch_size=32, callbacks=[es], verbose=0)
        class_weights = {0: 1.0, 1: 5.0}  # Ajuste selon ton déséquilibre
        model.fit(X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=EPOCHS, batch_size=32,
                callbacks=[es],
                class_weight=class_weights,
                verbose=0)

        # générer signaux et évaluer
        y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()
        df_test_copy = df_test.copy()
        df_test_copy['signal'] = y_pred[:len(df_test_copy)]  # Assure matching length
        ratio = evaluate_strategy(df_test_copy)
        models.append(model)
        scores.append(ratio)
        print(f"Model {i+1}/{N_MODELS} ratio(R/DD)={ratio:.3f}")

    os.makedirs(MODEL_DIR, exist_ok=True)    # <- Crée "models/" si nécessaire
    # sélection des TOP_K meilleurs
    top_idx = np.argsort(scores)[-TOP_K:]
    best_models = [models[i] for i in top_idx]
    for idx, m in enumerate(best_models):
        m.save(f"{MODEL_DIR}/model_{idx+1}.keras")
    print(f"Top {TOP_K} models saved: {top_idx}")
    return best_models

# ---------------------------
# 5. Prédiction en ensemble + Vote majoritaire
# ---------------------------
class EnsembleTrader:
    def __init__(self, models, scaler, features, threshold=SEUIL_DYNAMIQUE):
        self.models = models
        self.scaler = scaler
        self.features = features
        self.threshold = threshold

    # def predict(self, df_latest):
    #     # préparer séquence
    #     df = add_technical_indicators(df_latest.copy())
    #     X = self.scaler.transform(df[self.features].values)

    #     if len(X) < SEQ_LEN:
    #         raise ValueError(f"Pas assez de données ({len(X)}) pour construire une séquence de longueur {SEQ_LEN}")
    #     seq = X[-SEQ_LEN:].reshape(1, SEQ_LEN, len(self.features))

    #     # prédictions individuelles
    #     preds = [m.predict(seq)[0][0] for m in self.models]
    #     # vote majoritaire (bagging)
    #     votes = [1 if p>0.5 else 0 for p in preds]
    #     return int(sum(votes) > len(votes)/2)

    def predict_from_df(self, df_window):
        """
        Prédit le signal (0/1) sur une fenêtre brute de prix.
        df_window doit contenir ['open','high','low','close','tick_volume'].
        """
        # 1) Calcul des indicateurs (sans dropna global)
        df_ind = add_technical_indicators(
            df_window.rename(columns={'tick_volume':'volume'})
        )
        # 2) On enlève seulement les lignes où un **feature** est NaN
        df_ind = df_ind.dropna(subset=self.features)
        # 3) Vérifier la longueur pour la séquence
        if len(df_ind) < SEQ_LEN:
            raise ValueError(f"Pas assez de données après dropna(features) : {len(df_ind)} < {SEQ_LEN}")
        # 4) Normalisation + extraction de la dernière fenêtre
        X = self.scaler.transform(df_ind[self.features].values)
        X_seq = X[-SEQ_LEN:]
        seq = X_seq.reshape(1, SEQ_LEN, len(self.features))
        # # 1) Calculer tous les indicateurs techniques sur la fenêtre brute
        # df_ind = df_window.rename(columns={'tick_volume':'volume'})
        # df_ind = add_technical_indicators(df_ind)
        # # 2) S'assurer qu'on a assez de bougies après dropna()
        # if len(df_ind) < SEQ_LEN:
        #     raise ValueError(f"Pas assez de données après indicateurs : {len(df_ind)} < {SEQ_LEN}")
        # # 3) Normaliser puis extraire la séquence
        # X = self.scaler.transform(df_ind[self.features].values)
        # X_seq = X[-SEQ_LEN:]  # dernière tranche de longueur SEQ_LEN
        # seq = X_seq.reshape(1, SEQ_LEN, len(self.features))
        # 5) Prédictions + vote majoritaire

        # preds = [m.predict(seq)[0][0] for m in self.models]
        # vote  = int(sum(p > 0.5 for p in preds) > len(preds)/2)
        # return vote
        preds = [m.predict(seq)[0][0] for m in self.models]
        # votes = [1 if p > threshold else 0 for p in preds]
        # vote = int(sum(votes) > len(votes) / 2)
        avg = np.mean(preds)
        vote = int(avg > self.threshold)
        return vote

    def add_signal_column(self, df_prices):
        """
        Ajoute la colonne 'signal' à un DataFrame de prix bruts.
        df_prices doit contenir ['open','high','low','close','tick_volume'].
        """
        df = df_prices.rename(columns={'tick_volume':'volume'})  # juste pour la forme
        # df = add_technical_indicators(df.copy())
        signals = [np.nan] * len(df)
        # Pour chaque index à partir de SEQ_LEN, on prend la tranche brute
        for i in range(LOOKBACK, len(df)):
            window = df_prices.iloc[i-LOOKBACK:i]            # PAS de dropna ici
            signals[i] = self.predict_from_df(window)      # qui gère indicateurs + scaler
        df['signal'] = signals
        return df

    def execute_live(self, symbol=SYMBOL):
        initialize_mt5()
        account = mt5.account_info()
        balance = account.balance
        #lot = calculate_lot(balance)
        lot = LOT_SIZE

        # Get filling mode
        filling_mode = mt5.symbol_info(symbol).filling_mode - 1

        df_latest = fetch_data(symbol, TIMEFRAME, LOOKBACK)
        # signal = self.predict(df_latest)
        signal = self.predict_from_df(df_latest)
        action = "BUY" if signal==1 else "SELL"
        # exécution order
        price = mt5.symbol_info_tick(symbol).ask if action=="BUY" else mt5.symbol_info_tick(symbol).bid
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": mt5.ORDER_TYPE_BUY if action=="BUY" else mt5.ORDER_TYPE_SELL,
            "price": price,
            "sl": price - SL_PIPS * mt5.symbol_info(symbol).point if action=="BUY" else price + SL_PIPS * mt5.symbol_info(symbol).point,
            "tp": price + TP_PIPS * mt5.symbol_info(symbol).point if action=="BUY" else price - TP_PIPS * mt5.symbol_info(symbol).point,
            "deviation": 10,
            "magic": 123456,
            "comment": "Ensemble_LSTM_Bot",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling_mode
        }
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Order failed: {result.retcode} - {result.comment}")
        else:
            print(f"Order {action} executed @ {price}")
    
    def execute_live2(self, symbol=SYMBOL):
        df_latest = fetch_data(symbol, TIMEFRAME, LOOKBACK)
        # signal = self.predict(df_latest)
        signal = self.predict_from_df(df_latest)
        action = mt5.ORDER_TYPE_BUY if signal==1 else mt5.ORDER_TYPE_SELL
        m5.run(
            symbol=symbol,
            long=action == mt5.ORDER_TYPE_BUY,
            short=action == mt5.ORDER_TYPE_SELL,
            lot=LOT_SIZE
        )

    # def compute_optimal_threshold(self, df_prices, step=0.01):
    #     """
    #     Calcule automatiquement le meilleur seuil (entre 0 et 1)
    #     qui maximise le ratio rendement / drawdown.
    #     """
    #     scores = []
    #     thresholds = np.arange(0.1, 0.9, step)

    #     for t in thresholds:
    #         signals = [np.nan] * len(df_prices)
    #         for i in range(SEQ_LEN, len(df_prices)):
    #             window = df_prices.iloc[i-SEQ_LEN:i]
    #             try:
    #                 preds = [m.predict(self._prepare_sequence(window))[0][0] for m in self.models]
    #                 vote = int(np.mean(preds) > t)
    #                 signals[i] = vote
    #             except:
    #                 continue
    #         df_test = df_prices.copy()
    #         df_test['signal'] = signals
    #         ratio = evaluate_strategy(df_test)  # ta fonction existante
    #         scores.append(ratio)

    #     best_idx = np.nanargmax(scores)
    #     best_thresh = thresholds[best_idx]
    #     print(f"🔍 Meilleur seuil automatique trouvé : {best_thresh:.3f} (ratio R/DD = {scores[best_idx]:.3f})")
    #     return best_thresh

    # def _prepare_sequence(self, df_window):
    #     df = add_technical_indicators(df_window.copy())
    #     if len(df) < SEQ_LEN:
    #         raise ValueError("Trop peu de données")
    #     X = self.scaler.transform(df[self.features].values)
    #     X_seq = X[-SEQ_LEN:]
    #     return X_seq.reshape(1, SEQ_LEN, len(self.features))


# class RNNEnsembleStrategy(Strategy):
#     def init(self):
#         self.signal = self.data.signal

#     def next(self):
#         if self.signal[-1] == 1 and not self.position:
#             self.buy()
#         elif self.signal[-1] == 0 and self.position:
#             self.position.close()

# class RNNEnsembleStrategy(Strategy):
#     def init(self):
#         self.signal = self.data.signal

#     def next(self):
#         current_signal = self.signal[-1]

#         if current_signal == 1 and not self.position:
#             self.buy()
#         elif current_signal == 0 and not self.position:
#             self.sell()
#         elif self.position.is_long and current_signal == 0:
#             self.position.close()
#         elif self.position.is_short and current_signal == 1:
#             self.position.close()

class RNNEnsembleStrategy(Strategy):
    def init(self):
        self.signal = self.data.signal

    def next(self):
        sig = self.signal[-1]

        if pd.isna(sig):
            return

        sig = int(sig)  # <-- force 0.0 → 0, ou 1.0 → 1

        # Si pas de position ouverte
        if not self.position:
            if sig == 1:
                self.buy()
                print("BUY")
            elif sig == 0:
                self.sell()
                print("SELL")
        # Si position LONG est ouverte
        elif self.position.is_long:
            if sig == 0:
                self.position.close()
                self.sell()
                print("CLOSE LONG + SELL")
        # Si position SHORT est ouverte
        elif self.position.is_short:
            if sig == 1:
                self.position.close()
                self.buy()
                print("CLOSE SHORT + BUY")

        print(f"[{self.data.index[-1]}] Signal: {sig}, Position: {self.position}, Price: {self.data.Close[-1]}")

        # Ferme si inverse
        # if self.position:
        #     if sig == 1 and self.position.is_short:
        #         print("Closing short, opening long")
        #         self.position.close()
        #         self.buy()
        #     elif sig == 0 and self.position.is_long:
        #         print("Closing long, opening short")
        #         self.position.close()
        #         self.sell()
        # else:
        #     if sig == 1:
        #         print("BUY")
        #         self.buy()
        #     elif sig == 0:
        #         print("SELL")
        #         self.sell()


        # Fermer toute position si le signal change
        # if self.position:
        #     if sig == 1 and self.position.is_short:
        #         self.position.close()
        #     elif sig == 0 and self.position.is_long:
        #         self.position.close()

        # # Ouvrir nouvelle position uniquement si aucune n'est ouverte
        # if not self.position:
        #     if sig == 1:
        #         print(f"BUY executed at {self.data.Close[-1]}")
        #         self.buy()
        #     elif sig == 0:
        #         print(f"SELL executed at {self.data.Close[-1]}")
        #         self.sell()

        # BUY
        # if sig == 1:
        #     if not self.position:
        #         self.buy()
        #     elif self.position.is_short:
        #         self.position.close()
        #         self.buy()

        # # SELL
        # elif sig == 0:
        #     if not self.position:
        #         self.sell()
        #     elif self.position.is_long:
        #         self.position.close()
        #         self.sell()

# ---------------------------
# 6. Exécution principale
# ---------------------------
def run():
    if TRADING_LIVE:
        # === PHASE 1 : chargement des modèles pré-entraînés ===
        best_models = [load_model(f"{MODEL_DIR}/model_{i+1}.keras") for i in range(TOP_K)]
        scaler = joblib.load(SCALER_PATH)
        features = ['sma_20','rsi_14','macd','atr','return']
        trader = EnsembleTrader(best_models, scaler, features)

        # === PHASE 2 : trading continu ===
        mt5.initialize()
        try:
            while True:
                trader.execute_live(SYMBOL)
                # trader.execute_live2(SYMBOL)
                time.sleep(5)  # Pause pour éviter les appels trop fréquents
        finally:
            mt5.shutdown()
    elif BACKTESTING:
        mt5.initialize()

        # Phase 1 : on a déjà entraîné et chargé les 3 meilleurs modèles
        best_models = [load_model(f"{MODEL_DIR}/model_{i+1}.keras") for i in range(TOP_K)]
        scaler      = joblib.load(SCALER_PATH)
        features    = ['sma_20','rsi_14','macd','atr','return']
        trader      = EnsembleTrader(best_models, scaler, features)


        date_from = datetime(year=2025, month=7, day=1, hour=0, minute=0, second=0, tzinfo=timezone)
        date_to = datetime(year=2025, month=7, day=3, hour=23, minute=59, second=59, tzinfo=timezone)
        # 1) On récupère le DataFrame de prix complet
        df_prices = fetch_data_range(SYMBOL, TIMEFRAME, date_from=date_from, date_to=date_to)
        # print(df_prices)
        # exit()

        # trader.threshold = trader.compute_optimal_threshold(df_prices)

        # 2) On prépare la colonne 'signal' via la classe
        df_signals = trader.add_signal_column(df_prices)



        # print(df_signals['signal'].value_counts(dropna=False))
        # print(df_signals[['signal']].tail(10))
        # # Après avoir créé df_prices et chargé ton trader :
        # raw_scores = []
        # for i in range(LOOKBACK, len(df_prices)):
        #     window = df_prices.iloc[i-LOOKBACK : i]
        #     # Normalise+predictions
        #     df_ind = window.rename(columns={'tick_volume':'volume'})
        #     df_ind = add_technical_indicators(df_ind)
        #     df_ind = df_ind.dropna(subset=features)
        #     X = scaler.transform(df_ind[features].values)
        #     X_seq = X[-SEQ_LEN:].reshape(1, SEQ_LEN, len(features))
        #     # Concatène les 3 scores
        #     preds = [m.predict(X_seq)[0][0] for m in best_models]
        #     raw_scores.append(np.mean(preds))  # moyenne des 3 modèles
        # # Afficher un résumé
        # plt.hist(raw_scores, bins=20)
        # plt.title("Distribution des scores moyens")
        # plt.show()
        # print(f"Min score = {min(raw_scores):.3f}, Max score = {max(raw_scores):.3f}")
        # exit()

        print(df_signals['signal'].value_counts(dropna=False))
        print(df_signals.tail(20)[['signal', 'close']])

        df_bt = df_signals.rename(columns={
            'open':  'Open',
            'high':  'High',
            'low':   'Low',
            'close': 'Close',
            'volume':'Volume'
        })[
            ['Open','High','Low','Close','Volume','signal']
        ]

        bt = Backtest(
            df_bt,
            RNNEnsembleStrategy,
            cash=10000,
            commission=0.001,
            trade_on_close=True
            # exclusive_orders=True
        )
        
        # 1) Lancer le backtest
        stats = bt.run()
        # 2) Afficher toutes les stats
        print(stats)
        print("=== Backtest Results ===")
        # 3) Afficher les plus importantes sous forme formatée
        print(f"Total Return     : {stats['Return [%]']:.2f}%")
        print(f"Max Drawdown     : {stats['Max. Drawdown [%]']:.2f}%")
        print(f"Sharpe Ratio     : {stats['Sharpe Ratio']:.2f}")
        print(f"Sortino Ratio    : {stats['Sortino Ratio']:.2f}")
        print(f"Number of Trades : {stats['# Trades']}")
        print(f"Win Rate         : {stats['Win Rate [%]']:.2f}%")
        bt.plot()

    else:
        # entraîner et sélectionner
        best_models = train_and_select()
       