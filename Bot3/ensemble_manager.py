# ensemble_manager.py
"""
Gestionnaire d'ensemble de modèles LSTM avec vote intelligent
Corrige les problèmes de logique de vote et de sélection de modèles
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
import os
from datetime import datetime
import json

from config import config, TradingSignals, ModelStatus
from lstm_model import LSTMModel
from data_processor import DataProcessor

logger = logging.getLogger('trading_system.ensemble_manager')

class EnsembleManager:
    """
    Gestionnaire sophistiqué d'ensemble de modèles LSTM
    """
    
    def __init__(self):
        self.models: List[LSTMModel] = []
        self.model_performances: Dict[int, Dict[str, float]] = {}
        self.selected_models: List[int] = []
        self.ensemble_metrics: Dict[str, float] = {}
        self.data_processor = DataProcessor()
    
    def create_models(self, n_models: int = None) -> None:
        """
        Création de l'ensemble de modèles
        
        Args:
            n_models: Nombre de modèles à créer (défaut: config.N_MODELS)
        """
        if n_models is None:
            n_models = config.N_MODELS
        
        logger.info(f"Création de {n_models} modèles LSTM")
        
        self.models = []
        for i in range(n_models):
            model = LSTMModel(model_id=i)
            self.models.append(model)
        
        logger.info(f"Ensemble de {len(self.models)} modèles créé")
    
    def train_ensemble(self, data_dict: Dict[str, Any], 
                      use_parallel: bool = True) -> Dict[str, Any]:
        """
        Entraînement de l'ensemble avec traitement parallèle
        
        Args:
            data_dict: Données préprocessées
            use_parallel: Utiliser l'entraînement parallèle
            
        Returns:
            Résultats d'entraînement
        """
        logger.info(f"Début de l'entraînement de l'ensemble ({len(self.models)} modèles)")
        
        X_train = data_dict['X_train']
        y_train = data_dict['y_train']
        X_val = data_dict['X_val']
        y_val = data_dict['y_val']
        X_test = data_dict['X_test']
        y_test = data_dict['y_test']
        
        results = {}
        
        if use_parallel and len(self.models) > 1:
            # Entraînement parallèle
            results = self._train_parallel(X_train, y_train, X_val, y_val, X_test, y_test)
        else:
            # Entraînement séquentiel
            results = self._train_sequential(X_train, y_train, X_val, y_val, X_test, y_test)
        
        # Sélection des meilleurs modèles
        self._select_best_models(X_test, y_test)
        
        # Évaluation de l'ensemble
        ensemble_metrics = self._evaluate_ensemble(X_test, y_test)
        results['ensemble_metrics'] = ensemble_metrics
        
        logger.info(f"Entraînement de l'ensemble terminé. Modèles sélectionnés: {self.selected_models}")
        return results
    
    def _train_sequential(self, X_train: np.ndarray, y_train: np.ndarray,
                         X_val: np.ndarray, y_val: np.ndarray,
                         X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Entraînement séquentiel des modèles"""
        results = {'models': {}, 'training_times': {}}
        
        for model in self.models:
            start_time = datetime.now()
            
            try:
                # Entraînement
                history = model.train(X_train, y_train, X_val, y_val)
                
                # Évaluation
                metrics = model.evaluate(X_test, y_test)
                
                # Sauvegarde
                model.save_model()
                
                # Stockage des résultats
                training_time = (datetime.now() - start_time).total_seconds()
                results['models'][model.model_id] = {
                    'history': history,
                    'metrics': metrics,
                    'status': model.status
                }
                results['training_times'][model.model_id] = training_time
                
                self.model_performances[model.model_id] = metrics
                
                logger.info(f"Modèle {model.model_id} entraîné en {training_time:.1f}s - "
                           f"Accuracy: {metrics['accuracy']:.4f}")
                
            except Exception as e:
                logger.error(f"Erreur lors de l'entraînement du modèle {model.model_id}: {e}")
                results['models'][model.model_id] = {
                    'error': str(e),
                    'status': ModelStatus.ERROR
                }
        
        return results
    
    def _train_parallel(self, X_train: np.ndarray, y_train: np.ndarray,
                       X_val: np.ndarray, y_val: np.ndarray,
                       X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Entraînement parallèle des modèles"""
        results = {'models': {}, 'training_times': {}}
        
        def train_single_model(model: LSTMModel) -> Tuple[int, Dict[str, Any]]:
            """Fonction pour entraîner un seul modèle"""
            start_time = datetime.now()
            
            try:
                # Entraînement
                history = model.train(X_train, y_train, X_val, y_val)
                
                # Évaluation
                metrics = model.evaluate(X_test, y_test)
                
                # Sauvegarde
                model.save_model()
                
                training_time = (datetime.now() - start_time).total_seconds()
                
                return model.model_id, {
                    'history': history,
                    'metrics': metrics,
                    'training_time': training_time,
                    'status': model.status
                }
                
            except Exception as e:
                logger.error(f"Erreur modèle {model.model_id}: {e}")
                return model.model_id, {
                    'error': str(e),
                    'status': ModelStatus.ERROR
                }
        
        # Exécution parallèle
        max_workers = min(len(self.models), 4)  # Limiter le nombre de threads
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Soumission des tâches
            future_to_model = {
                executor.submit(train_single_model, model): model.model_id 
                for model in self.models
            }
            
            # Récupération des résultats
            for future in as_completed(future_to_model):
                model_id, result = future.result()
                results['models'][model_id] = result
                
                if 'metrics' in result:
                    self.model_performances[model_id] = result['metrics']
                    logger.info(f"Modèle {model_id} terminé - "
                              f"Accuracy: {result['metrics']['accuracy']:.4f}")
                
                if 'training_time' in result:
                    results['training_times'][model_id] = result['training_time']
        
        return results
    
    def _select_best_models(self, X_test: np.ndarray, y_test: np.ndarray,
                           selection_criteria: str = 'f1_score') -> None:
        """
        Sélection des meilleurs modèles basée sur les performances
        
        Args:
            X_test: Données de test
            y_test: Labels de test
            selection_criteria: Critère de sélection ('f1_score', 'auc_roc', 'accuracy')
        """
        logger.info(f"Sélection des meilleurs modèles basée sur {selection_criteria}")
        
        # Créer un DataFrame des performances
        performances_df = pd.DataFrame(self.model_performances).T
        
        if len(performances_df) == 0:
            logger.warning("Aucune performance de modèle disponible")
            return
        
        # Trier par critère de sélection
        if selection_criteria in performances_df.columns:
            sorted_models = performances_df.sort_values(
                by=selection_criteria, ascending=False
            )
        else:
            logger.warning(f"Critère {selection_criteria} non trouvé, utilisation de 'accuracy'")
            sorted_models = performances_df.sort_values(
                by='accuracy', ascending=False
            )
        
        # Sélectionner les meilleurs modèles (au moins 3, au maximum la moitié)
        n_select = max(3, min(len(self.models) // 2, len(sorted_models)))
        self.selected_models = sorted_models.head(n_select).index.tolist()
        
        # Log des modèles sélectionnés
        logger.info(f"Modèles sélectionnés: {self.selected_models}")
        for model_id in self.selected_models:
            metrics = self.model_performances[model_id]
            logger.info(f"  Modèle {model_id}: {selection_criteria}={metrics.get(selection_criteria, 'N/A'):.4f}")
    
    def _evaluate_ensemble(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Évaluation de l'ensemble de modèles
        
        Args:
            X_test: Données de test
            y_test: Labels de test
            
        Returns:
            Métriques de l'ensemble
        """
        if not self.selected_models:
            logger.warning("Aucun modèle sélectionné pour l'évaluation de l'ensemble")
            return {}
        
        logger.info("Évaluation de l'ensemble de modèles")
        
        # Prédictions de chaque modèle sélectionné
        all_predictions = []
        all_probabilities = []
        
        for model_id in self.selected_models:
            model = self.models[model_id]
            if model.status == ModelStatus.TRAINED:
                predictions, probabilities = model.predict(X_test)
                all_predictions.append(predictions)
                all_probabilities.append(probabilities)
        
        if not all_predictions:
            logger.error("Aucune prédiction disponible pour l'évaluation de l'ensemble")
            return {}
        
        # Conversion en arrays numpy
        predictions_array = np.array(all_predictions)  # Shape: (n_models, n_samples)
        probabilities_array = np.array(all_probabilities)
        
        # Vote majoritaire (CORRECTION de la logique de vote)
        ensemble_predictions = np.where(
            np.mean(predictions_array, axis=0) >= 0.5, 1, 0
        )
        
        # Probabilités moyennes
        ensemble_probabilities = np.mean(probabilities_array, axis=0)
        
        # Calcul des métriques
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        ensemble_metrics = {
            'ensemble_accuracy': float(accuracy_score(y_test, ensemble_predictions)),
            'ensemble_precision': float(precision_score(y_test, ensemble_predictions)),
            'ensemble_recall': float(recall_score(y_test, ensemble_predictions)),
            'ensemble_f1': float(f1_score(y_test, ensemble_predictions)),
            'ensemble_auc': float(roc_auc_score(y_test, ensemble_probabilities)),
            'n_models_used': len(self.selected_models),
            'agreement_threshold': config.ENSEMBLE_THRESHOLD
        }
        
        self.ensemble_metrics = ensemble_metrics
        
        logger.info(f"Ensemble - Accuracy: {ensemble_metrics['ensemble_accuracy']:.4f}, "
                   f"F1: {ensemble_metrics['ensemble_f1']:.4f}, "
                   f"AUC: {ensemble_metrics['ensemble_auc']:.4f}")
        
        return ensemble_metrics
    
    def predict_ensemble(self, X: np.ndarray, 
                        use_confidence_threshold: bool = True) -> Dict[str, Any]:
        """
        Prédiction avec l'ensemble de modèles (CORRECTION de la logique)
        
        Args:
            X: Données d'entrée
            use_confidence_threshold: Utiliser le seuil de confiance
            
        Returns:
            Résultats de prédiction
        """
        if not self.selected_models:
            raise ValueError("Aucun modèle sélectionné pour les prédictions")
        
        # Prédictions de chaque modèle
        model_predictions = []
        model_probabilities = []
        
        for model_id in self.selected_models:
            model = self.models[model_id]
            if model.status == ModelStatus.TRAINED:
                try:
                    predictions, probabilities = model.predict(X)
                    model_predictions.append(predictions)
                    model_probabilities.append(probabilities)
                except Exception as e:
                    logger.warning(f"Erreur prédiction modèle {model_id}: {e}")
        
        if not model_predictions:
            raise ValueError("Aucune prédiction disponible")
        
        # Conversion en arrays
        predictions_array = np.array(model_predictions)
        probabilities_array = np.array(model_probabilities)
        
        # Calcul du consensus
        buy_votes = np.sum(predictions_array == 1, axis=0)
        sell_votes = np.sum(predictions_array == 0, axis=0)
        total_votes = len(model_predictions)
        
        # Probabilité moyenne
        avg_probability = np.mean(probabilities_array, axis=0)
        
        # Décision finale avec seuil de confiance
        confidence_ratio = buy_votes / total_votes
        
        if use_confidence_threshold:
            # Signal fort seulement si confiance suffisante
            final_signal = np.where(
                confidence_ratio >= config.ENSEMBLE_THRESHOLD, TradingSignals.BUY,
                np.where(confidence_ratio <= (1 - config.ENSEMBLE_THRESHOLD), 
                        TradingSignals.SELL, TradingSignals.HOLD)
            )
        else:
            # Vote majoritaire simple
            final_signal = np.where(buy_votes > sell_votes, 
                                  TradingSignals.BUY, TradingSignals.SELL)
        
        return {
            'signal': final_signal[0] if len(final_signal) == 1 else final_signal,
            'confidence': float(confidence_ratio[0]) if len(confidence_ratio) == 1 else confidence_ratio.tolist(),
            'buy_votes': int(buy_votes[0]) if len(buy_votes) == 1 else buy_votes.tolist(),
            'sell_votes': int(sell_votes[0]) if len(sell_votes) == 1 else sell_votes.tolist(),
            'avg_probability': float(avg_probability[0]) if len(avg_probability) == 1 else avg_probability.tolist(),
            'models_used': self.selected_models,
            'total_models': total_votes
        }
    
    def load_ensemble(self) -> bool:
        """
        Chargement de l'ensemble sauvegardé
        
        Returns:
            True si chargement réussi
        """
        logger.info("Chargement de l'ensemble de modèles")
        
        success_count = 0
        
        # Chargement des modèles individuels
        for model in self.models:
            if model.load_model():
                success_count += 1
                # Récupération des métriques
                if model.metrics:
                    self.model_performances[model.model_id] = model.metrics
        
        # Chargement des métadonnées de l'ensemble
        ensemble_metadata_path = os.path.join(
            config.BASE_PATH, config.MODELS_DIR, config.SYMBOL, "ensemble_metadata.json"
        )
        
        if os.path.exists(ensemble_metadata_path):
            try:
                with open(ensemble_metadata_path, 'r') as f:
                    metadata = json.load(f)
                    self.selected_models = metadata.get('selected_models', [])
                    self.ensemble_metrics = metadata.get('ensemble_metrics', {})
                logger.info("Métadonnées de l'ensemble chargées")
            except Exception as e:
                logger.warning(f"Erreur lors du chargement des métadonnées: {e}")
        
        # Chargement du processeur de données
        self.data_processor.load_scalers()
        
        logger.info(f"Ensemble chargé: {success_count}/{len(self.models)} modèles, "
                   f"sélectionnés: {self.selected_models}")
        
        return success_count > 0
    
    def save_ensemble(self) -> bool:
        """
        Sauvegarde de l'ensemble complet
        
        Returns:
            True si sauvegarde réussie
        """
        logger.info("Sauvegarde de l'ensemble de modèles")
        
        success_count = 0
        
        # Sauvegarde des modèles individuels
        for model in self.models:
            if model.save_model():
                success_count += 1
        
        # Sauvegarde des métadonnées de l'ensemble
        ensemble_metadata = {
            'selected_models': self.selected_models,
            'ensemble_metrics': self.ensemble_metrics,
            'model_performances': self.model_performances,
            'config': {
                'n_models': len(self.models),
                'ensemble_threshold': config.ENSEMBLE_THRESHOLD,
                'selection_criteria': 'f1_score'
            },
            'timestamp': datetime.now().isoformat()
        }
        
        ensemble_metadata_path = os.path.join(
            config.BASE_PATH, config.MODELS_DIR, config.SYMBOL, "ensemble_metadata.json"
        )
        
        try:
            with open(ensemble_metadata_path, 'w') as f:
                json.dump(ensemble_metadata, f, indent=2)
            logger.info("Métadonnées de l'ensemble sauvegardées")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des métadonnées: {e}")
        
        # Sauvegarde du processeur de données
        self.data_processor.save_scalers()
        
        logger.info(f"Ensemble sauvegardé: {success_count}/{len(self.models)} modèles")
        return success_count > 0
    
    def get_ensemble_summary(self) -> str:
        """
        Résumé complet de l'ensemble
        
        Returns:
            Résumé textuel
        """
        summary_lines = []
        summary_lines.append("=" * 50)
        summary_lines.append("RÉSUMÉ DE L'ENSEMBLE DE MODÈLES")
        summary_lines.append("=" * 50)
        
        summary_lines.append(f"Nombre total de modèles: {len(self.models)}")
        summary_lines.append(f"Modèles sélectionnés: {len(self.selected_models)}")
        summary_lines.append(f"Seuil de confiance: {config.ENSEMBLE_THRESHOLD}")
        
        if self.ensemble_metrics:
            summary_lines.append("\n=== MÉTRIQUES DE L'ENSEMBLE ===")
            for metric, value in self.ensemble_metrics.items():
                if isinstance(value, float):
                    summary_lines.append(f"{metric}: {value:.4f}")
                else:
                    summary_lines.append(f"{metric}: {value}")
        
        if self.selected_models:
            summary_lines.append("\n=== MODÈLES SÉLECTIONNÉS ===")
            for model_id in self.selected_models:
                if model_id in self.model_performances:
                    metrics = self.model_performances[model_id]
                    summary_lines.append(f"Modèle {model_id}: F1={metrics.get('f1_score', 'N/A'):.4f}, "
                                        f"AUC={metrics.get('auc_roc', 'N/A'):.4f}")
        
        return "\n".join(summary_lines)