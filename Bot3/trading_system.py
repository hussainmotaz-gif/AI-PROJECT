# trading_system.py
"""
Système de trading principal intégrant tous les composants
"""

import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, Any, Optional, Tuple
import logging
import logging.config
from datetime import datetime, timedelta
import warnings
import os
import time

# Imports locaux
from config import config, backtest_config, LOGGING_CONFIG, TradingSignals
from data_processor import DataProcessor
from ensemble_manager import EnsembleManager
from backtester import AdvancedBacktester

# Configuration des warnings et logging
warnings.filterwarnings("ignore")
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger('trading_system.main')

class TradingSystemError(Exception):
    """Exception personnalisée pour le système de trading"""
    pass

class TradingSystem:
    """
    Système de trading principal orchestrant tous les composants
    """
    
    def __init__(self):
        self.data_processor = DataProcessor()
        self.ensemble_manager = EnsembleManager()
        self.backtester = AdvancedBacktester()
        
        # État du système
        self.is_trained = False
        self.last_data_update = None
        self.current_data: Optional[pd.DataFrame] = None
        
        logger.info("Système de trading initialisé")
    
    def fetch_data(self, symbol: str = None, period: str = None) -> pd.DataFrame:
        """
        Récupération des données de marché avec gestion d'erreurs
        
        Args:
            symbol: Symbole à récupérer (défaut: config.SYMBOL)
            period: Période de données (défaut: calculé selon config)
            
        Returns:
            DataFrame avec données OHLCV
        """
        if symbol is None:
            symbol = config.SYMBOL
        
        if period is None:
            # Conversion des jours en période yfinance
            total_days = config.TRAINING_DAYS + config.EXECUTION_DAYS
            if total_days <= 365:
                period = "1y"
            elif total_days <= 365 * 2:
                period = "2y"
            elif total_days <= 365 * 5:
                period = "5y"
            else:
                period = "max"
        
        logger.info(f"Récupération des données pour {symbol} (période: {period})")
        
        try:
            # Récupération avec yfinance
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval="1d")
            
            if data.empty:
                raise TradingSystemError(f"Aucune donnée récupérée pour {symbol}")
            
            # Nettoyage des colonnes
            data.columns = [col.lower() for col in data.columns]
            
            # Vérification des données minimales
            if len(data) < config.MIN_DATA_POINTS:
                raise TradingSystemError(f"Pas assez de données: {len(data)} < {config.MIN_DATA_POINTS}")
            
            logger.info(f"Données récupérées: {len(data)} lignes de {data.index[0]} à {data.index[-1]}")
            
            self.current_data = data
            self.last_data_update = datetime.now()
            
            return data
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des données: {e}")
            raise TradingSystemError(f"Impossible de récupérer les données: {e}")
    
    def train_models(self, data: pd.DataFrame = None, force_retrain: bool = False) -> Dict[str, Any]:
        """
        Entraînement complet des modèles
        
        Args:
            data: Données à utiliser (optionnel)
            force_retrain: Forcer le réentraînement même si modèles existants
            
        Returns:
            Résultats d'entraînement
        """
        logger.info("=" * 50)
        logger.info("DÉBUT DE L'ENTRAÎNEMENT DES MODÈLES")
        logger.info("=" * 50)
        
        # Vérification des modèles existants
        if not force_retrain and self._check_existing_models():
            logger.info("Modèles existants trouvés, chargement...")
            if self.load_models():
                logger.info("Modèles chargés avec succès")
                return {"status": "loaded", "message": "Modèles existants chargés"}
        
        # Récupération des données si non fournies
        if data is None:
            data = self.fetch_data()
        
        try:
            # 1. Traitement des données
            logger.info("Étape 1/3: Traitement des données...")
            processed_data = self.data_processor.process_pipeline(data, fit_scalers=True)
            
            # 2. Création et entraînement de l'ensemble
            logger.info("Étape 2/3: Entraînement de l'ensemble...")
            self.ensemble_manager.create_models(config.N_MODELS)
            training_results = self.ensemble_manager.train_ensemble(processed_data)
            
            # 3. Sauvegarde
            logger.info("Étape 3/3: Sauvegarde des modèles...")
            save_success = self.ensemble_manager.save_ensemble()
            
            if save_success:
                self.is_trained = True
                logger.info("Entraînement terminé avec succès")
                
                # Résumé des résultats
                logger.info("\n" + self.ensemble_manager.get_ensemble_summary())
                
                return {
                    "status": "success",
                    "training_results": training_results,
                    "ensemble_metrics": self.ensemble_manager.ensemble_metrics,
                    "selected_models": self.ensemble_manager.selected_models
                }
            else:
                raise TradingSystemError("Erreur lors de la sauvegarde des modèles")
                
        except Exception as e:
            logger.error(f"Erreur lors de l'entraînement: {e}")
            raise TradingSystemError(f"Échec de l'entraînement: {e}")
    
    def _check_existing_models(self) -> bool:
        """
        Vérification de l'existence de modèles pré-entraînés
        
        Returns:
            True si modèles existants trouvés
        """
        models_dir = os.path.join(config.BASE_PATH, config.MODELS_DIR, config.SYMBOL)
        
        if not os.path.exists(models_dir):
            return False
        
        # Vérification des fichiers de modèles
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.h5')]
        ensemble_metadata = os.path.join(models_dir, "ensemble_metadata.json")
        
        return len(model_files) >= config.N_MODELS and os.path.exists(ensemble_metadata)
    
    def load_models(self) -> bool:
        """
        Chargement des modèles pré-entraînés
        
        Returns:
            True si chargement réussi
        """
        logger.info("Chargement des modèles pré-entraînés...")
        
        try:
            # Création de l'ensemble
            self.ensemble_manager.create_models(config.N_MODELS)
            
            # Chargement des modèles et scalers
            if self.ensemble_manager.load_ensemble() and self.data_processor.load_scalers():
                self.is_trained = True
                logger.info("Modèles chargés avec succès")
                return True
            else:
                logger.warning("Échec du chargement des modèles")
                return False
                
        except Exception as e:
            logger.error(f"Erreur lors du chargement: {e}")
            return False
    
    def run_backtest(self, data: pd.DataFrame = None, 
                    start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """
        Exécution d'un backtest complet
        
        Args:
            data: Données à utiliser (optionnel)
            start_date: Date de début au format 'YYYY-MM-DD'
            end_date: Date de fin au format 'YYYY-MM-DD'
            
        Returns:
            Résultats du backtest
        """
        logger.info("=" * 50)
        logger.info("DÉBUT DU BACKTEST")
        logger.info("=" * 50)
        
        if not self.is_trained:
            raise TradingSystemError("Les modèles doivent être entraînés avant le backtest")
        
        # Récupération des données
        if data is None:
            data = self.fetch_data()
        
        try:
            # Traitement des données (sans réajuster les scalers)
            processed_data = self.data_processor.process_pipeline(data, fit_scalers=False)
            
            # Génération des signaux sur les données de test
            X_test = processed_data['X_test']
            test_df = processed_data['test_df']
            
            logger.info(f"Génération des signaux sur {len(X_test)} échantillons...")
            signals = []
            
            for i in range(len(X_test)):
                try:
                    # Prédiction avec l'ensemble
                    prediction_result = self.ensemble_manager.predict_ensemble(
                        X_test[i:i+1], use_confidence_threshold=True
                    )
                    signals.append(prediction_result['signal'])
                    
                except Exception as e:
                    logger.warning(f"Erreur prédiction échantillon {i}: {e}")
                    signals.append(TradingSignals.HOLD)
            
            # Conversion des dates pour le backtest
            start_dt = pd.to_datetime(start_date) if start_date else None
            end_dt = pd.to_datetime(end_date) if end_date else None
            
            # Alignement des données avec les signaux
            test_data_aligned = test_df.iloc[config.LOOKBACK_WINDOW:].copy()
            
            if len(signals) != len(test_data_aligned):
                min_len = min(len(signals), len(test_data_aligned))
                signals = signals[:min_len]
                test_data_aligned = test_data_aligned.iloc[:min_len]
            
            # Exécution du backtest
            logger.info(f"Exécution du backtest sur {len(test_data_aligned)} points...")
            backtest_results = self.backtester.run_backtest(
                test_data_aligned, signals, start_dt, end_dt
            )
            
            # Génération du rapport
            report = self.backtester.generate_report()
            
            logger.info("Backtest terminé avec succès")
            logger.info(f"\nRésumé: Rendement total = {backtest_results['metrics'].get('total_return', 0):.2f}%")
            
            return {
                "status": "success",
                "backtest_results": backtest_results,
                "report": report,
                "signals_generated": len(signals)
            }
            
        except Exception as e:
            logger.error(f"Erreur lors du backtest: {e}")
            raise TradingSystemError(f"Échec du backtest: {e}")
    
    def generate_current_signal(self, data: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Génération du signal de trading actuel
        
        Args:
            data: Données récentes (optionnel)
            
        Returns:
            Signal et informations associées
        """
        if not self.is_trained:
            raise TradingSystemError("Les modèles doivent être entraînés avant la génération de signaux")
        
        # Récupération des données récentes
        if data is None:
            data = self.fetch_data()
        
        try:
            # Traitement des données (dernière fenêtre)
            df_with_features = self.data_processor.create_features(data)
            
            # Préparation de la dernière séquence
            recent_features = df_with_features[config.FEATURE_COLUMNS].tail(config.LOOKBACK_WINDOW)
            
            if len(recent_features) < config.LOOKBACK_WINDOW:
                raise TradingSystemError(f"Pas assez de données récentes: {len(recent_features)}")
            
            # Transformation avec les scalers existants
            features_scaled = self.data_processor.transform_features(
                pd.DataFrame(recent_features, columns=config.FEATURE_COLUMNS)
            )
            
            # Reshape pour LSTM (1 échantillon)
            X_current = features_scaled.reshape(1, config.LOOKBACK_WINDOW, len(config.FEATURE_COLUMNS))
            
            # Prédiction avec l'ensemble
            prediction_result = self.ensemble_manager.predict_ensemble(
                X_current, use_confidence_threshold=True
            )
            
            # Informations contextuelles
            current_price = data['close'].iloc[-1]
            current_time = data.index[-1]
            
            signal_info = {
                "timestamp": current_time,
                "current_price": float(current_price),
                "signal": prediction_result['signal'],
                "confidence": prediction_result['confidence'],
                "buy_votes": prediction_result['buy_votes'],
                "sell_votes": prediction_result['sell_votes'],
                "models_used": prediction_result['models_used'],
                "ensemble_probability": prediction_result['avg_probability']
            }
            
            # Interprétation du signal
            signal_map = {
                TradingSignals.BUY: "BUY",
                TradingSignals.SELL: "SELL", 
                TradingSignals.HOLD: "HOLD"
            }
            
            logger.info(f"Signal généré: {signal_map.get(prediction_result['signal'], 'UNKNOWN')} "
                       f"(Confiance: {prediction_result['confidence']:.1%})")
            
            return signal_info
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération du signal: {e}")
            raise TradingSystemError(f"Impossible de générer le signal: {e}")
    
    def run_live_trading_simulation(self, duration_minutes: int = 60) -> None:
        """
        Simulation de trading en temps réel
        
        Args:
            duration_minutes: Durée de la simulation en minutes
        """
        if not self.is_trained:
            raise TradingSystemError("Les modèles doivent être entraînés pour le trading live")
        
        logger.info(f"Début de la simulation de trading live ({duration_minutes} minutes)")
        
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)
        
        signal_history = []
        
        try:
            while datetime.now() < end_time:
                try:
                    # Génération du signal actuel
                    signal_info = self.generate_current_signal()
                    signal_history.append(signal_info)
                    
                    # Affichage du signal
                    signal_str = ["SELL", "HOLD", "BUY"][signal_info['signal'] + 1]
                    logger.info(f"[{signal_info['timestamp']}] "
                              f"Prix: ${signal_info['current_price']:.2f} | "
                              f"Signal: {signal_str} | "
                              f"Confiance: {signal_info['confidence']:.1%}")
                    
                    # Attente avant le prochain signal (simulation)
                    time.sleep(60)  # 1 minute entre chaque signal
                    
                except KeyboardInterrupt:
                    logger.info("Simulation interrompue par l'utilisateur")
                    break
                except Exception as e:
                    logger.error(f"Erreur durant la simulation: {e}")
                    time.sleep(30)  # Attendre avant de réessayer
        
        except Exception as e:
            logger.error(f"Erreur fatale durant la simulation: {e}")
        
        finally:
            logger.info(f"Simulation terminée. {len(signal_history)} signaux générés.")
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        État actuel du système
        
        Returns:
            Informations sur l'état du système
        """
        status = {
            "is_trained": self.is_trained,
            "last_data_update": self.last_data_update.isoformat() if self.last_data_update else None,
            "current_data_points": len(self.current_data) if self.current_data is not None else 0,
            "config": {
                "symbol": config.SYMBOL,
                "lookback_window": config.LOOKBACK_WINDOW,
                "n_models": config.N_MODELS,
                "ensemble_threshold": config.ENSEMBLE_THRESHOLD
            }
        }
        
        if self.is_trained:
            status["ensemble_info"] = {
                "selected_models": self.ensemble_manager.selected_models,
                "ensemble_metrics": self.ensemble_manager.ensemble_metrics
            }
        
        return status
    
    def cleanup(self) -> None:
        """Nettoyage des ressources"""
        logger.info("Nettoyage des ressources du système")
        # Libération de la mémoire des modèles TensorFlow
        try:
            import tensorflow as tf
            tf.keras.backend.clear_session()
        except:
            pass


def main():
    """
    Fonction principale pour démonstration
    """
    try:
        # Initialisation du système
        trading_system = TradingSystem()
        
        print("=" * 60)
        print("SYSTÈME DE TRADING LSTM - DÉMONSTRATION")
        print("=" * 60)
        
        # 1. Entraînement des modèles
        print("\n1. Entraînement des modèles...")
        training_results = trading_system.train_models()
        print(f"Statut d'entraînement: {training_results['status']}")
        
        # 2. Backtest
        print("\n2. Exécution du backtest...")
        backtest_results = trading_system.run_backtest()
        
        # Affichage des résultats
        metrics = backtest_results['backtest_results']['metrics']
        print(f"\nRésultats du backtest:")
        print(f"- Rendement total: {metrics['total_return']:+.2f}%")
        print(f"- Sharpe ratio: {metrics['sharpe_ratio']:.3f}")
        print(f"- Max drawdown: {metrics['max_drawdown']:.2f}%")
        print(f"- Taux de réussite: {metrics['win_rate']:.1f}%")
        
        # 3. Signal actuel
        print("\n3. Génération du signal actuel...")
        current_signal = trading_system.generate_current_signal()
        signal_names = {-1: "SELL", 0: "HOLD", 1: "BUY"}
        print(f"Signal actuel: {signal_names[current_signal['signal']]} "
              f"(Confiance: {current_signal['confidence']:.1%})")
        
        # 4. Visualisation (optionnel)
        try:
            trading_system.backtester.plot_results(show=False)
            print("\nGraphiques de performance générés")
        except Exception as e:
            print(f"Erreur lors de la génération des graphiques: {e}")
        
        print("\nDémonstration terminée avec succès!")
        
    except Exception as e:
        logger.error(f"Erreur dans la démonstration: {e}")
        print(f"Erreur: {e}")
    
    finally:
        try:
            trading_system.cleanup()
        except:
            pass


if __name__ == "__main__":
    main()