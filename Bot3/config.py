# config.py
"""
Configuration centralisée pour le système de trading LSTM
"""

import os
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class TradingConfig:
    """Configuration principale du système de trading"""
    
    # Paramètres de trading
    SYMBOL: str = "XAUUSD"
    LOT_SIZE: float = 0.01
    MAX_POSITIONS: int = 1
    RISK_PER_TRADE: float = 0.02  # 2% du capital par trade
    
    # Paramètres du modèle
    LOOKBACK_WINDOW: int = 20  # Fenêtre d'observation (optimisée)
    PREDICTION_HORIZON: int = 1  # Prédire le prochain jour
    FEATURE_COLUMNS: List[str] = None
    
    # Architecture LSTM
    LSTM_UNITS: int = 64
    HIDDEN_LAYERS: int = 2
    DROPOUT_RATE: float = 0.3
    LEARNING_RATE: float = 0.001
    
    # Entraînement
    EPOCHS: int = 100
    BATCH_SIZE: int = 32
    VALIDATION_SPLIT: float = 0.2
    PATIENCE: int = 10  # Early stopping
    
    # Données
    TRAINING_DAYS: int = 1826  # 5 ans
    EXECUTION_DAYS: int = 252  # 1 an pour les prédictions
    MIN_DATA_POINTS: int = 500
    
    # Ensemble de modèles
    N_MODELS: int = 5
    ENSEMBLE_THRESHOLD: float = 0.6  # 60% des modèles doivent être d'accord
    
    # Backtesting
    TRAIN_RATIO: float = 0.6
    VALIDATION_RATIO: float = 0.2
    TEST_RATIO: float = 0.2
    
    # Chemins de fichiers
    BASE_PATH: str = os.path.dirname(__file__)
    MODELS_DIR: str = "models"
    SCALERS_DIR: str = "scalers"
    LOGS_DIR: str = "logs"
    
    def __post_init__(self):
        """Initialisation post-création"""
        if self.FEATURE_COLUMNS is None:
            self.FEATURE_COLUMNS = [
                'sma_10', 'sma_20', 'sma_50',
                'ema_12', 'ema_26',
                'rsi_14', 'rsi_21',
                'bb_upper', 'bb_lower', 'bb_width',
                'macd', 'macd_signal', 'macd_histogram',
                'atr_14', 'volatility_10', 'volatility_20',
                'volume_sma_10', 'price_to_sma_ratio'
            ]
        
        # Créer les dossiers nécessaires
        for directory in [self.MODELS_DIR, self.SCALERS_DIR, self.LOGS_DIR]:
            full_path = os.path.join(self.BASE_PATH, directory, self.SYMBOL)
            os.makedirs(full_path, exist_ok=True)
    
    def get_model_path(self, model_id: int, model_type: str = "weights") -> str:
        """Retourne le chemin du modèle"""
        filename = f"{self.SYMBOL}_model_{model_id}.{model_type}.h5"
        return os.path.join(self.BASE_PATH, self.MODELS_DIR, self.SYMBOL, filename)
    
    def get_scaler_path(self, scaler_type: str = "features") -> str:
        """Retourne le chemin du scaler"""
        filename = f"{self.SYMBOL}_{scaler_type}_scaler.joblib"
        return os.path.join(self.BASE_PATH, self.SCALERS_DIR, self.SYMBOL, filename)


@dataclass
class BacktestConfig:
    """Configuration pour le backtesting"""
    
    INITIAL_CAPITAL: float = 10000.0
    COMMISSION: float = 0.0003  # 0.03% par trade
    SLIPPAGE: float = 0.0001   # 0.01% de slippage
    
    # Stop Loss et Take Profit
    STOP_LOSS_PCT: float = 0.02    # 2%
    TAKE_PROFIT_PCT: float = 0.04  # 4%
    TRAILING_STOP: bool = True
    
    # Métriques à calculer
    CALCULATE_SHARPE: bool = True
    CALCULATE_SORTINO: bool = True
    CALCULATE_CALMAR: bool = True
    RISK_FREE_RATE: float = 0.02  # 2% annuel


# Instance globale de configuration
config = TradingConfig()
backtest_config = BacktestConfig()

# Constantes de trading
class TradingSignals:
    BUY = 1
    SELL = -1
    HOLD = 0

class ModelStatus:
    UNTRAINED = "untrained"
    TRAINING = "training"
    TRAINED = "trained"
    ERROR = "error"

# Logging configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'detailed': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        },
        'simple': {
            'format': '%(asctime)s - %(levelname)s - %(message)s'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'simple'
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': os.path.join(config.BASE_PATH, config.LOGS_DIR, 'trading.log'),
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5,
            'level': 'DEBUG',
            'formatter': 'detailed'
        }
    },
    'loggers': {
        'trading_system': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
            'propagate': False
        }
    }
}