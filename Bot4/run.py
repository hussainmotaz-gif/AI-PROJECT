from MT5 import *
import numpy as np
import pandas as pd
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")
import time
from datetime import datetime, timedelta
import pickle
import ta
from joblib import dump, load
import os
from sklearn.preprocessing import StandardScaler
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tqdm import tqdm # Library to estimate the time until the end of the loop
import matplotlib as mpl
import matplotlib.pyplot as plt

m5: MT5 = MT5()

FEATURE_COLUMNS = ['SMA 15', 'SMA 60', 'MSD 10', 'MSD 30', 'rsi']

symbol: str = "XAUUSD"
lotSize: float = 0.01
training: bool = False
backtesting: bool = False
execution: bool = True
lag: int = 15 #les jours précédents, ex 15 jours précédents, les 15 premieres lignes du dataframe
nb_neurons: int = 10
nb_hidden_layer: int = 1
epochs: int = 1
dropout: float = 0.20 # pour eviter le problème d'overfitting
nb_days_for_tranning: int = 1826 # 5 ans
nb_days_for_execution: int = 700

nb_model_training: int = 100
base_path = os.path.dirname(__file__)  # Répertoire du script actuel
#rnn_models1_name: str = os.path.join(base_path, "Weights_RNN", symbol, "RNN n°0.weights.h5")
#rnn_models2_name: str = os.path.join(base_path, "Weights_RNN", symbol, "RNN n°2.weights.h5")
#rnn_models3_name: str = os.path.join(base_path, "Weights_RNN", symbol, "RNN n°3.weights.h5")

rnn_models1_name: str = os.path.join(base_path, "Weights_RNN", symbol, "RNN n°99.weights.h5")
rnn_models2_name: str = os.path.join(base_path, "Weights_RNN", symbol, "RNN n°95.weights.h5")
rnn_models3_name: str = os.path.join(base_path, "Weights_RNN", symbol, "RNN n°58.weights.h5")

def feature_engineering(df):
    """Feature engineering pour le modèle RNN"""
    """ DON'T PUT THE SHIFT HERE"""

    # We copy the dataframe to avoid interferences in the data
    df_copy = df.copy()

    # Calcul du rendement et de la variable cible (dummy)
    df_copy["returns"] = df_copy["close"].pct_change(1)
    df_copy["dummy"] = np.round(df_copy["returns"] + 0.5)

    # Moyennes mobiles SMAs
    df_copy["SMA 15"] = df_copy["close"].rolling(15).mean()
    df_copy["SMA 60"] = df_copy["close"].rolling(60).mean()

    # Volatilités
    df_copy["MSD 10"] = df_copy["returns"].rolling(10).std()
    df_copy["MSD 30"] = df_copy["returns"].rolling(30).std()

    # RSI
    rsi = ta.momentum.RSIIndicator(df_copy["close"], window=14, fillna=False)
    df_copy["rsi"] = rsi.rsi()

    # STANDARDIZATION

    # Suppression des NaN
    df_copy = df_copy.dropna()

    # Standardisation des features principales
    scaler = StandardScaler()
    df_copy[FEATURE_COLUMNS] = scaler.fit_transform(df_copy[FEATURE_COLUMNS])

    return df_copy

    # sc = StandardScaler()
    # df_copy = df_copy[["SMA 15", "SMA 60", "MSD 10", "MSD 30", "rsi"]].dropna()
    # df_copy_sc = sc.fit_transform(df_copy)
    #
    # return df_copy_sc


def X_3d_RNN(X_s, y_s):
    # Simple verification
    if len(X_s) != len(y_s):
        print("Warnings")

    # Create the X_train
    X_train = []
    for variable in range(0, X_s.shape[1]):
        X = []
        for i in range(lag, X_s.shape[0]):
            X.append(X_s[i - lag:i, variable])
        X_train.append(X)
    X_train, np.array(X_train)
    X_train = np.swapaxes(np.swapaxes(X_train, 0, 1), 1, 2)

    # Create the y_train
    y_train = []
    for i in range(lag, y_s.shape[0]):
        y_train.append(y_s[i, :].reshape(-1, 1).transpose())
    y_train = np.concatenate(y_train, axis=0)
    return X_train, y_train

def RNN_weights(input_shape):
    # INITIALIZATION OF THE MODEL
    classifier = Sequential()

    # ADD LSTM LAYER
    #classifier.add(LSTM(units=10, return_sequences=True, input_shape=(15, 5,)))
    classifier.add(LSTM(units=nb_neurons, return_sequences=True,
                        input_shape=input_shape))

    # LOOP WHICH ADD LSTM LAYER
    for _ in range(nb_hidden_layer):
        classifier.add(LSTM(units=nb_neurons, return_sequences=True))
        classifier.add(Dropout(dropout))  # pour eviter le problème d'overfitting

    # LAST LSTM LAYER BUT WITH return_sequences = False
    classifier.add(LSTM(units=nb_neurons, return_sequences=False))

    # OUTPUT DENSE LAYER
    classifier.add(Dense(1, activation="sigmoid"))

    # COMPILE THE MODEL
    classifier.compile(loss="binary_crossentropy", optimizer="adam")

    return classifier


def RNN_cl_sig(symbol):
    """ Function for predict the value of tommorow using DNN model"""

    # Take the lastest percentage of change
    df = m5.get_data(symbol, nb_days_for_execution)

    # Features engeeniring
    data = feature_engineering(df)
    # X_data, _ = X_3d_RNN(data, np.zeros([700, 1]), 15)
    X_data, _ = X_3d_RNN(data[FEATURE_COLUMNS].values, np.zeros([len(data), 1]))
    X = X_data[-1:, :, :]
    print(np.shape(X))

    # Create the weights if there is not in the folder
    rnn_1 = RNN_weights((X_data.shape[1], X_data.shape[2]))
    rnn_2 = RNN_weights((X_data.shape[1], X_data.shape[2]))
    rnn_3 = RNN_weights((X_data.shape[1], X_data.shape[2]))

    # Import trained weights
    rnn_1.load_weights(rnn_models1_name)
    rnn_2.load_weights(rnn_models2_name)
    rnn_3.load_weights(rnn_models3_name)

    # Bagging
    pr1 = np.where(rnn_1.predict(X) == 0, -1, 1)
    pr2 = np.where(rnn_2.predict(X) == 0, -1, 1)
    pr3 = np.where(rnn_3.predict(X) == 0, -1, 1)

    # Find the signal
    buy = (pr1 + pr2 + pr3)[0][0] >= 1
    sell = not buy

    return buy, sell


def RNN_train(symbol, nb_model=None):
    # Take the lastest percentage of change
    df = m5.get_data(symbol, nb_days_for_tranning)
    dfc = feature_engineering(df)

    # Percentage train set
    split = int(0.80 * len(dfc))
    split_val = int(0.90 * len(dfc))

    # Train set creation
    X_train = dfc[FEATURE_COLUMNS].iloc[:split, :]
    y_train = dfc[["dummy"]].iloc[:split]

    # Test set creation
    X_test = dfc[FEATURE_COLUMNS].iloc[split:, :]
    y_test = dfc[["dummy"]].iloc[split:]

    # STANDARDISATION

    sc = StandardScaler()

    X_train_sc = sc.fit_transform(X_train)
    X_test_sc = sc.transform(X_test)

    # Transform 2-dimensional data to 3-dimensional data
    X_train_3d, y_train_3d = X_3d_RNN(X_train_sc, y_train.values)
    X_test_3d, y_test_3d = X_3d_RNN(X_test_sc, y_test.values)

    classifier = RNN_weights((X_train_3d.shape[1], X_train_3d.shape[2]))

    # TRAINING
    early_stop = EarlyStopping(verbose=1, patience=5)
    classifier.fit(X_train_3d, y_train_3d, validation_data=(X_test_3d, y_test_3d),
                   epochs=epochs, callbacks=[early_stop])

    # Create predictions for the whole dataset
    y_pred_train = np.concatenate((np.zeros([lag, 1]), classifier.predict(X_train_3d)),
                                  axis=0)

    y_pred_test = np.concatenate((np.zeros([lag, 1]), classifier.predict(X_test_3d)),
                                 axis=0)

    dfc["prediction"] = np.concatenate((y_pred_train, y_pred_test),
                                       axis=0)

    dfc["prediction"] = np.where(dfc["prediction"] < 0.5, -1, 1) # -1: sell; 1: buy

    # Compute the strategy
    dfc["strategy"] = np.sign(dfc["prediction"]) * dfc["returns"]

    # Extraire les rendements pour le backtesting
    test_returns = dfc["strategy"].iloc[split + lag:split_val]
    val_returns = dfc["strategy"].iloc[split_val:]

    if nb_model is not None:
        classifier.save_weights(f"Weights_RNN/{symbol}/RNN n°{nb_model}.weights.h5")

    return test_returns, val_returns, dfc  # Retourner aussi le DataFrame complet

current_account_info = mt5.account_info()
print("------------------------------------------------------------------")
print("Date: ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print(f"Balance: {current_account_info.balance} USD, \t"
      f"Equity: {current_account_info.equity} USD, \t"
      f"Profit: {current_account_info.profit} USD")
print("------------------------------------------------------------------")

def training_model():
    # Trouver les trois model les plus
    # Empty lists to contain the returns
    returns_test = []
    returns_val = []
    # Compute the returns during the test and validation periods
    for i in tqdm(range(nb_model_training)):
        test, val, __ = RNN_train(symbol, nb_model=i)
        returns_test.append(test)
        returns_val.append(val)

    ##################################################################################
    # Organize well all the data
    RNN_returns = pd.DataFrame(returns_test, index=[f"RNN n°{i}" for i in range(len(returns_test))]).transpose()
    ##################################################################################

    ##################################################################################
    # Adapt the size of the graph
    plt.figure(figsize=(15, 8))
    # Plot the returns
    plt.plot(RNN_returns.cumsum(axis=0), alpha=0.6)
    # Plot break-even line
    plt.axhline(0, color="red")
    plt.title("All Models")
    ##################################################################################

    ##################################################################################
    # Create empty dataframe
    values = pd.DataFrame(index=["ret/dd", "ret", "dd"])
    # Compute return and max drawdawn for each ANN
    for col in RNN_returns.columns:
        # Compute the returns and the max drawdown for one ANN
        ret, dd = RNN_returns[col].cumsum().iloc[-1], -np.min(drawdown_function(RNN_returns[col]))
        # Create a column containing the ret/dd ratio, the returns and the drawdown
        values[col] = ret / dd, ret, dd

    RNN_models = list(values.transpose().sort_values(by="ret/dd", ascending=False).index[0:3])

    print(RNN_models)
    ##################################################################################

    ##################################################################################
    #Use the 3best models on the test set to trade on the validation set
    # Organize the data
    RNN_val = pd.DataFrame(returns_val, index=[f"RNN n°{i}" for i in range(len(returns_val))]).transpose()
    # Adapt the size
    plt.figure(figsize=(15, 8))
    # Plot the portfolio method containing the 10 best strategies
    plt.plot(RNN_val[RNN_models].cumsum(axis=0), alpha=0.6)
    plt.axhline(0, color="red")
    plt.title("3best models on the test set to trade on the validation set")
    ##################################################################################

    ##################################################################################
    #Use the 3best models on the test set to trade on the test set
    # Organize the data
    # RNN_val = pd.DataFrame(returns_val, index=[f"RNN n°{i}" for i in range(len(returns_val))]).transpose()
    # Adapt the size
    plt.figure(figsize=(15, 8))
    # Plot the portfolio method containing the 10 best strategies
    plt.plot(RNN_returns[RNN_models].cumsum(axis=0), alpha=0.6)
    plt.axhline(0, color="red")
    plt.title("3best models on the test set to trade on the test set")
    ##################################################################################

    # show the graph
    plt.show()

def drawdown_function(serie):
  # We compute Cumsum of the returns
  cum = serie.dropna().cumsum() + 1

  # We compute max of the cumsum on the period (accumulate max) # (1,3,5,3,1) --> (1,3,5,5,5)
  running_max = np.maximum.accumulate(cum)

  # We compute drawdown
  drawdown = cum/running_max - 1
  return drawdown

def execute_strategy():
    start = datetime.now().strftime("%H:%M:%S")
    while True:
        # Verfication for launch
        # if datetime.now().weekday() not in (5, 6):
        #     is_time = datetime.now().strftime("%H:%M:%S") == start  # "23:59:59"
        # else:
        #     is_time = False
        is_time = True

        # Launch the algorithm
        if is_time:
            # Open the trades
            # Create the signals
            buy, sell = RNN_cl_sig(symbol)

            # Run the algorithm
            if not training:
                m5.run(symbol, buy, sell, lotSize)

            print(f"Symbol: {symbol}\t"
                  f"Buy: {buy}\t"
                  f"Sell: {sell}")
        time.sleep(1)


def BackTest(returns_serie, annualized_scalar=252, benchmark_symbol="^GSPC"):
    """
    Fonction de backtesting pour évaluer une stratégie de trading
    
    Parameters:
    -----------
    returns_serie : pd.Series
        Série des rendements de la stratégie (déjà calculés, pas les prix)
    annualized_scalar : int
        Nombre de jours de trading par an (252 par défaut)
    benchmark_symbol : str
        Symbole du benchmark à comparer (S&P 500 par défaut)
    
    Returns:
    --------
    dict : Dictionnaire avec les métriques de performance
    """
    
    # Vérifications d'entrée
    if not isinstance(returns_serie, pd.Series):
        raise ValueError("returns_serie doit être une pandas Series")
    
    if returns_serie.empty:
        raise ValueError("La série de rendements est vide")
    
    # Nettoyer la série (enlever les NaN)
    returns_serie = returns_serie.dropna()
    
    try:
        # Télécharger les données du benchmark pour la même période
        start_date = returns_serie.index[0] - timedelta(days=5)  # Marge de sécurité
        end_date = returns_serie.index[-1] + timedelta(days=1)
        
        benchmark_data = yf.download(benchmark_symbol, start=start_date, end=end_date)
        
        if benchmark_data.empty:
            print(f"Attention: Impossible de télécharger les données du benchmark {benchmark_symbol}")
            benchmark_returns = None
        else:
            benchmark_returns = benchmark_data["Close"].pct_change(1)
            benchmark_returns.name = "Benchmark"
            
            # Aligner les dates avec la stratégie
            benchmark_returns = benchmark_returns.reindex(returns_serie.index, method='ffill')
    
    except Exception as e:
        print(f"Erreur lors du téléchargement du benchmark: {e}")
        benchmark_returns = None
    
    # Calculer les métriques de performance
    metrics = {}
    
    # Rendements
    total_return = (1 + returns_serie).prod() - 1
    annualized_return = (1 + total_return) ** (annualized_scalar / len(returns_serie)) - 1
    
    # Volatilité
    volatility = returns_serie.std() * np.sqrt(annualized_scalar)
    
    # Sharpe Ratio (en supposant un taux sans risque de 0)
    sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
    
    # Drawdown
    cumulative_returns = (1 + returns_serie).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns / running_max - 1) * 100
    max_drawdown = drawdown.min()
    
    # Sortino Ratio
    negative_returns = returns_serie[returns_serie < 0]
    if len(negative_returns) > 0:
        downside_deviation = negative_returns.std() * np.sqrt(annualized_scalar)
        sortino_ratio = annualized_return / downside_deviation if downside_deviation > 0 else 0
    else:
        sortino_ratio = float('inf')  # Pas de rendements négatifs
    
    # Métriques vs benchmark
    if benchmark_returns is not None:
        # Aligner les données
        combined_data = pd.concat([returns_serie, benchmark_returns], axis=1).dropna()
        
        if not combined_data.empty and len(combined_data.columns) == 2:
            strategy_col = combined_data.columns[0]
            benchmark_col = combined_data.columns[1]
            
            # Beta
            covariance = np.cov(combined_data[strategy_col], combined_data[benchmark_col])[0, 1]
            benchmark_variance = np.var(combined_data[benchmark_col])
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
            
            # Alpha
            benchmark_return = combined_data[benchmark_col].mean() * annualized_scalar
            alpha = annualized_return - beta * benchmark_return
            
            metrics['beta'] = beta
            metrics['alpha'] = alpha
            metrics['benchmark_total_return'] = (1 + combined_data[benchmark_col]).prod() - 1
    
    # Stocker les métriques
    metrics.update({
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': abs(max_drawdown),
        'win_rate': len(returns_serie[returns_serie > 0]) / len(returns_serie),
        'number_of_trades': len(returns_serie)
    })
    
    # Créer les graphiques
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Analyse de Performance de la Stratégie', fontsize=16, fontweight='bold')
    
    # 1. Rendements cumulés
    ax1 = axes[0, 0]
    cumulative_returns_pct = (cumulative_returns - 1) * 100
    ax1.plot(cumulative_returns_pct.index, cumulative_returns_pct.values, 
             color='#2E8B57', linewidth=2, label='Stratégie')
    
    if benchmark_returns is not None and 'benchmark_total_return' in metrics:
        benchmark_cumulative = (1 + benchmark_returns.reindex(returns_serie.index, method='ffill')).cumprod()
        benchmark_cumulative_pct = (benchmark_cumulative - 1) * 100
        ax1.plot(benchmark_cumulative_pct.index, benchmark_cumulative_pct.values, 
                 color='#B8860B', linewidth=2, label='Benchmark', alpha=0.8)
    
    ax1.set_title('Rendements Cumulés (%)')
    ax1.set_ylabel('Rendement (%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Drawdown
    ax2 = axes[0, 1]
    ax2.fill_between(drawdown.index, 0, drawdown.values, 
                     color='#DC143C', alpha=0.7, label='Drawdown')
    ax2.set_title('Drawdown (%)')
    ax2.set_ylabel('Drawdown (%)')
    ax2.grid(True, alpha=0.3)
    
    # 3. Distribution des rendements
    ax3 = axes[1, 0]
    returns_pct = returns_serie * 100
    ax3.hist(returns_pct, bins=50, color='#4682B4', alpha=0.7, edgecolor='black')
    ax3.axvline(returns_pct.mean(), color='red', linestyle='--', linewidth=2, label='Moyenne')
    ax3.set_title('Distribution des Rendements Quotidiens')
    ax3.set_xlabel('Rendement (%)')
    ax3.set_ylabel('Fréquence')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Rendements mobiles
    ax4 = axes[1, 1]
    rolling_returns = returns_serie.rolling(window=30).mean() * 100
    ax4.plot(rolling_returns.index, rolling_returns.values, color='#9932CC', linewidth=2)
    ax4.set_title('Rendements Moyens Mobiles (30 jours)')
    ax4.set_ylabel('Rendement Moyen (%)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Afficher les métriques
    print("\n" + "="*60)
    print("RAPPORT DE PERFORMANCE")
    print("="*60)
    print(f"Période analysée: {returns_serie.index[0].strftime('%Y-%m-%d')} à {returns_serie.index[-1].strftime('%Y-%m-%d')}")
    print(f"Nombre de jours: {len(returns_serie)}")
    print("\nRENDEMENTS:")
    print(f"  Rendement total: {metrics['total_return']:.2%}")
    print(f"  Rendement annualisé: {metrics['annualized_return']:.2%}")
    print(f"  Volatilité annualisée: {metrics['volatility']:.2%}")
    print("\nMÉTRIQUES DE RISQUE:")
    print(f"  Ratio de Sharpe: {metrics['sharpe_ratio']:.3f}")
    print(f"  Ratio de Sortino: {metrics['sortino_ratio']:.3f}")
    print(f"  Drawdown maximum: {metrics['max_drawdown']:.2f}%")
    print(f"  Taux de réussite: {metrics['win_rate']:.2%}")
    
    if 'alpha' in metrics and 'beta' in metrics:
        print(f"\nCOMPARAISON AU BENCHMARK:")
        print(f"  Alpha: {metrics['alpha']:.2%}")
        print(f"  Beta: {metrics['beta']:.3f}")
        print(f"  Rendement du benchmark: {metrics['benchmark_total_return']:.2%}")
    
    print("="*60)
    
    return metrics

def test_single_model_performance(symbol):
    """Tester la performance d'un seul modèle"""
    
    # Entraîner le modèle
    test_returns, val_returns, __  = RNN_train(symbol)
    
    print("=== PERFORMANCE SUR PÉRIODE DE TEST ===")
    test_metrics = BackTest(test_returns)
    
    print("\n=== PERFORMANCE SUR PÉRIODE DE VALIDATION ===")
    val_metrics = BackTest(val_returns)
    
    return test_metrics, val_metrics

def run():
    if training:
        training_model()
    elif backtesting:
        test_metrics, val_metrics = test_single_model_performance(symbol)
    elif execution:
        execute_strategy()