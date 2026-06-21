# data_processor.py
"""
Module de traitement des données et feature engineering
Corrige tous les problèmes de data leakage et de standardisation
"""

import numpy as np
import pandas as pd
import ta
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.utils import check_array
from typing import Tuple, Optional, Dict, Any
import logging
import warnings
from joblib import dump, load
import os

from config import config, TradingSignals

warnings.filterwarnings("ignore")
logger = logging.getLogger('trading_system.data_processor')

class DataProcessor:
    """
    Classe pour le traitement des données avec correction des biais temporels
    """
    
    def __init__(self):
        self.feature_scaler: Optional[StandardScaler] = None
        self.target_scaler: Optional[RobustScaler] = None
        self.is_fitted = False
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Création des features techniques sans data leakage
        
        Args:
            df: DataFrame avec colonnes OHLCV
            
        Returns:
            DataFrame avec features techniques
        """
        logger.info("Création des features techniques...")
        
        if len(df) < 100:
            raise ValueError(f"Pas assez de données: {len(df)} < 100")
        
        df_features = df.copy()
        
        # 1. Moyennes mobiles
        df_features['sma_10'] = df['close'].rolling(window=10, min_periods=10).mean()
        df_features['sma_20'] = df['close'].rolling(window=20, min_periods=20).mean()
        df_features['sma_50'] = df['close'].rolling(window=50, min_periods=50).mean()
        
        # 2. Moyennes mobiles exponentielles
        df_features['ema_12'] = df['close'].ewm(span=12).mean()
        df_features['ema_26'] = df['close'].ewm(span=26).mean()
        
        # 3. RSI (Relative Strength Index)
        df_features['rsi_14'] = ta.momentum.RSIIndicator(
            df['close'], window=14, fillna=False
        ).rsi()
        df_features['rsi_21'] = ta.momentum.RSIIndicator(
            df['close'], window=21, fillna=False
        ).rsi()
        
        # 4. Bandes de Bollinger
        bb_indicator = ta.volatility.BollingerBands(
            df['close'], window=20, window_dev=2, fillna=False
        )
        df_features['bb_upper'] = bb_indicator.bollinger_hband()
        df_features['bb_lower'] = bb_indicator.bollinger_lband()
        df_features['bb_width'] = (df_features['bb_upper'] - df_features['bb_lower']) / df_features['sma_20']
        
        # 5. MACD
        macd_indicator = ta.trend.MACD(
            df['close'], window_slow=26, window_fast=12, window_sign=9, fillna=False
        )
        df_features['macd'] = macd_indicator.macd()
        df_features['macd_signal'] = macd_indicator.macd_signal()
        df_features['macd_histogram'] = macd_indicator.macd_diff()
        
        # 6. ATR (Average True Range)
        df_features['atr_14'] = ta.volatility.AverageTrueRange(
            df['high'], df['low'], df['close'], window=14, fillna=False
        ).average_true_range()
        
        # 7. Volatilité (écart-type des rendements)
        returns = df['close'].pct_change()
        df_features['volatility_10'] = returns.rolling(window=10, min_periods=10).std()
        df_features['volatility_20'] = returns.rolling(window=20, min_periods=20).std()
        
        # 8. Volume features (si disponible)
        if 'volume' in df.columns:
            df_features['volume_sma_10'] = df['volume'].rolling(window=10, min_periods=10).mean()
        else:
            df_features['volume_sma_10'] = 1.0  # Valeur par défaut
        
        # 9. Features de ratio
        df_features['price_to_sma_ratio'] = df['close'] / df_features['sma_20']
        
        # Suppression des NaN
        initial_len = len(df_features)
        df_features = df_features.dropna()
        dropped_rows = initial_len - len(df_features)
        
        if dropped_rows > 0:
            logger.warning(f"{dropped_rows} lignes supprimées à cause des NaN")
        
        if len(df_features) < config.MIN_DATA_POINTS:
            raise ValueError(f"Pas assez de données après nettoyage: {len(df_features)}")
        
        logger.info(f"Features créées: {len(df_features)} lignes, {len(config.FEATURE_COLUMNS)} features")
        return df_features
    
    def create_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Création de la variable cible FUTURE (correction du data leakage)
        
        Args:
            df: DataFrame avec prix de clôture
            
        Returns:
            DataFrame avec variable cible
        """
        df_target = df.copy()
        
        # CORRECTION MAJEURE: Prédire le futur, pas le présent
        future_returns = df['close'].pct_change(periods=config.PREDICTION_HORIZON).shift(-config.PREDICTION_HORIZON)
        
        # Classification binaire: 1 si hausse, 0 si baisse
        df_target['target'] = (future_returns > 0).astype(int)
        
        # Suppression des dernières lignes (pas de target futur disponible)
        df_target = df_target[:-config.PREDICTION_HORIZON]
        
        logger.info(f"Variable cible créée: {df_target['target'].sum()} hausses sur {len(df_target)} observations")
        return df_target
    
    def prepare_sequences(self, features: np.ndarray, targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Préparation des séquences temporelles pour LSTM
        
        Args:
            features: Features standardisées
            targets: Variables cibles
            
        Returns:
            Tuple (X_sequences, y_sequences)
        """
        if len(features) != len(targets):
            raise ValueError(f"Tailles incompatibles: features={len(features)}, targets={len(targets)}")
        
        X_sequences = []
        y_sequences = []
        
        for i in range(config.LOOKBACK_WINDOW, len(features)):
            # Séquence de features
            sequence = features[i-config.LOOKBACK_WINDOW:i]
            X_sequences.append(sequence)
            
            # Target correspondant
            y_sequences.append(targets[i])
        
        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)
        
        logger.info(f"Séquences créées: {X_sequences.shape}, targets: {y_sequences.shape}")
        return X_sequences, y_sequences
    
    def fit_scalers(self, train_features: pd.DataFrame, train_targets: pd.Series) -> None:
        """
        Ajustement des scalers UNIQUEMENT sur les données d'entraînement
        
        Args:
            train_features: Features d'entraînement
            train_targets: Targets d'entraînement
        """
        logger.info("Ajustement des scalers...")
        
        # Scaler pour les features (StandardScaler)
        self.feature_scaler = StandardScaler()
        self.feature_scaler.fit(train_features[config.FEATURE_COLUMNS])
        
        # Scaler pour les targets (RobustScaler pour gérer les outliers)
        self.target_scaler = RobustScaler()
        target_reshaped = train_targets.values.reshape(-1, 1)
        self.target_scaler.fit(target_reshaped)
        
        self.is_fitted = True
        logger.info("Scalers ajustés avec succès")
    
    def transform_features(self, features: pd.DataFrame) -> np.ndarray:
        """
        Transformation des features avec le scaler ajusté
        
        Args:
            features: Features à transformer
            
        Returns:
            Features transformées
        """
        if not self.is_fitted:
            raise ValueError("Les scalers doivent être ajustés avant la transformation")
        
        return self.feature_scaler.transform(features[config.FEATURE_COLUMNS])
    
    def transform_targets(self, targets: pd.Series) -> np.ndarray:
        """
        Transformation des targets avec le scaler ajusté
        
        Args:
            targets: Targets à transformer
            
        Returns:
            Targets transformées
        """
        if not self.is_fitted:
            raise ValueError("Les scalers doivent être ajustés avant la transformation")
        
        target_reshaped = targets.values.reshape(-1, 1)
        return self.target_scaler.transform(target_reshaped).flatten()
    
    def split_data_temporal(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split temporel des données (CORRECTION du split aléatoire)
        
        Args:
            df: DataFrame complet
            
        Returns:
            Tuple (train_df, val_df, test_df)
        """
        total_len = len(df)
        
        # Calcul des indices de split temporel
        train_end = int(total_len * config.TRAIN_RATIO)
        val_end = int(total_len * (config.TRAIN_RATIO + config.VALIDATION_RATIO))
        
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()
        
        logger.info(f"Split temporel: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        return train_df, val_df, test_df
    
    def save_scalers(self) -> None:
        """Sauvegarde des scalers"""
        if not self.is_fitted:
            raise ValueError("Les scalers doivent être ajustés avant la sauvegarde")
        
        feature_scaler_path = config.get_scaler_path("features")
        target_scaler_path = config.get_scaler_path("targets")
        
        dump(self.feature_scaler, feature_scaler_path)
        dump(self.target_scaler, target_scaler_path)
        
        logger.info(f"Scalers sauvegardés: {feature_scaler_path}, {target_scaler_path}")
    
    def load_scalers(self) -> bool:
        """
        Chargement des scalers sauvegardés
        
        Returns:
            True si chargement réussi, False sinon
        """
        feature_scaler_path = config.get_scaler_path("features")
        target_scaler_path = config.get_scaler_path("targets")
        
        try:
            if os.path.exists(feature_scaler_path) and os.path.exists(target_scaler_path):
                self.feature_scaler = load(feature_scaler_path)
                self.target_scaler = load(target_scaler_path)
                self.is_fitted = True
                logger.info("Scalers chargés avec succès")
                return True
            else:
                logger.warning("Fichiers de scalers non trouvés")
                return False
        except Exception as e:
            logger.error(f"Erreur lors du chargement des scalers: {e}")
            return False
    
    def process_pipeline(self, df: pd.DataFrame, fit_scalers: bool = True) -> Dict[str, Any]:
        """
        Pipeline complet de traitement des données
        
        Args:
            df: DataFrame brut
            fit_scalers: Si True, ajuste les scalers (mode entraînement)
            
        Returns:
            Dictionnaire avec données traitées
        """
        logger.info("Début du pipeline de traitement des données")
        
        # 1. Création des features
        df_with_features = self.create_features(df)
        
        # 2. Création des targets
        df_with_target = self.create_target(df_with_features)
        
        # 3. Split temporel
        train_df, val_df, test_df = self.split_data_temporal(df_with_target)
        
        # 4. Ajustement des scalers (seulement sur train)
        if fit_scalers:
            self.fit_scalers(train_df, train_df['target'])
            self.save_scalers()
        elif not self.is_fitted:
            if not self.load_scalers():
                raise ValueError("Impossible de charger les scalers et fit_scalers=False")
        
        # 5. Transformation des données
        train_features_scaled = self.transform_features(train_df)
        val_features_scaled = self.transform_features(val_df)
        test_features_scaled = self.transform_features(test_df)
        
        # 6. Préparation des séquences
        X_train, y_train = self.prepare_sequences(train_features_scaled, train_df['target'].values)
        X_val, y_val = self.prepare_sequences(val_features_scaled, val_df['target'].values)
        X_test, y_test = self.prepare_sequences(test_features_scaled, test_df['target'].values)
        
        logger.info("Pipeline de traitement terminé avec succès")
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test,
            'train_df': train_df, 'val_df': val_df, 'test_df': test_df,
            'original_df': df_with_target
        }