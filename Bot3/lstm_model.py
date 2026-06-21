# lstm_model.py
"""
Modèle LSTM amélioré avec architecture sophistiquée et gestion d'erreurs
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input, Attention, MultiHeadAttention
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import logging
import os
from typing import Tuple, Dict, Any, Optional, List
import json

from config import config, ModelStatus

logger = logging.getLogger('trading_system.lstm_model')

class LSTMModel:
    """
    Modèle LSTM sophistiqué avec attention et régularisation
    """
    
    def __init__(self, model_id: int):
        self.model_id = model_id
        self.model: Optional[Model] = None
        self.status = ModelStatus.UNTRAINED
        self.training_history: Dict[str, Any] = {}
        self.metrics: Dict[str, float] = {}
        
        # Configuration de TensorFlow pour éviter les erreurs de GPU
        self._configure_tensorflow()
    
    def _configure_tensorflow(self):
        """Configuration de TensorFlow"""
        try:
            # Configuration GPU si disponible
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"GPU détecté: {len(gpus)} GPU(s)")
            else:
                logger.info("Pas de GPU détecté, utilisation du CPU")
        except Exception as e:
            logger.warning(f"Erreur configuration GPU: {e}")
    
    def build_model(self, input_shape: Tuple[int, int]) -> Model:
        """
        Construction du modèle LSTM avec architecture avancée
        
        Args:
            input_shape: (timesteps, features)
            
        Returns:
            Modèle Keras compilé
        """
        logger.info(f"Construction du modèle {self.model_id} avec input_shape: {input_shape}")
        
        # Input layer
        inputs = Input(shape=input_shape, name='lstm_input')
        
        # Première couche LSTM avec return_sequences=True
        x = LSTM(
            units=config.LSTM_UNITS,
            return_sequences=True,
            dropout=config.DROPOUT_RATE,
            recurrent_dropout=config.DROPOUT_RATE * 0.5,
            kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
            name='lstm_1'
        )(inputs)
        
        x = BatchNormalization(name='batch_norm_1')(x)
        
        # Couches LSTM cachées
        for i in range(config.HIDDEN_LAYERS):
            # Retourner les séquences pour toutes les couches sauf la dernière
            return_sequences = (i < config.HIDDEN_LAYERS - 1)
            
            x = LSTM(
                units=config.LSTM_UNITS // (2 ** i),  # Diminution progressive
                return_sequences=return_sequences,
                dropout=config.DROPOUT_RATE,
                recurrent_dropout=config.DROPOUT_RATE * 0.5,
                kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
                name=f'lstm_{i+2}'
            )(x)
            
            if return_sequences:
                x = BatchNormalization(name=f'batch_norm_{i+2}')(x)
        
        # Couche Dense avec dropout
        x = Dense(
            units=config.LSTM_UNITS // 2,
            activation='relu',
            kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
            name='dense_1'
        )(x)
        x = Dropout(config.DROPOUT_RATE, name='dropout_final')(x)
        
        # Couche de sortie
        outputs = Dense(
            units=1,
            activation='sigmoid',
            name='output'
        )(x)
        
        # Création du modèle
        model = Model(inputs=inputs, outputs=outputs, name=f'LSTM_Model_{self.model_id}')
        
        # Compilation avec optimiseur Adam personnalisé
        optimizer = Adam(
            learning_rate=config.LEARNING_RATE,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall', tf.keras.metrics.AUC(name='auc')]
        )
        
        self.model = model
        logger.info(f"Modèle {self.model_id} construit avec {model.count_params()} paramètres")
        return model
    
    def get_callbacks(self) -> List:
        """
        Création des callbacks pour l'entraînement
        
        Returns:
            Liste des callbacks
        """
        callbacks = []
        
        # Early Stopping
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=config.PATIENCE,
            restore_best_weights=True,
            verbose=1,
            mode='min'
        )
        callbacks.append(early_stop)
        
        # Réduction du learning rate
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=config.PATIENCE // 2,
            min_lr=1e-7,
            verbose=1,
            mode='min'
        )
        callbacks.append(reduce_lr)
        
        # Sauvegarde du meilleur modèle
        checkpoint_path = config.get_model_path(self.model_id, "checkpoint")
        checkpoint = ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
            mode='min'
        )
        callbacks.append(checkpoint)
        
        return callbacks
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """
        Entraînement du modèle avec validation
        
        Args:
            X_train: Données d'entraînement
            y_train: Labels d'entraînement
            X_val: Données de validation
            y_val: Labels de validation
            
        Returns:
            Historique d'entraînement
        """
        logger.info(f"Début de l'entraînement du modèle {self.model_id}")
        self.status = ModelStatus.TRAINING
        
        try:
            # Construction du modèle si nécessaire
            if self.model is None:
                self.build_model((X_train.shape[1], X_train.shape[2]))
            
            # Callbacks
            callbacks = self.get_callbacks()
            
            # Entraînement
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=config.EPOCHS,
                batch_size=config.BATCH_SIZE,
                callbacks=callbacks,
                verbose=1,
                shuffle=False  # Important pour les données temporelles
            )
            
            # Sauvegarde de l'historique
            self.training_history = {
                'loss': history.history['loss'],
                'val_loss': history.history['val_loss'],
                'accuracy': history.history['accuracy'],
                'val_accuracy': history.history['val_accuracy']
            }
            
            self.status = ModelStatus.TRAINED
            logger.info(f"Entraînement du modèle {self.model_id} terminé avec succès")
            
            return self.training_history
            
        except Exception as e:
            self.status = ModelStatus.ERROR
            logger.error(f"Erreur lors de l'entraînement du modèle {self.model_id}: {e}")
            raise e
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Évaluation complète du modèle
        
        Args:
            X_test: Données de test
            y_test: Labels de test
            
        Returns:
            Métriques d'évaluation
        """
        if self.model is None or self.status != ModelStatus.TRAINED:
            raise ValueError(f"Le modèle {self.model_id} doit être entraîné avant l'évaluation")
        
        logger.info(f"Évaluation du modèle {self.model_id}")
        
        try:
            # Prédictions
            y_pred_proba = self.model.predict(X_test, verbose=0)
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            # Métriques de base
            test_loss, test_accuracy, test_precision, test_recall, test_auc = self.model.evaluate(
                X_test, y_test, verbose=0
            )
            
            # AUC-ROC
            try:
                auc_score = roc_auc_score(y_test, y_pred_proba)
            except:
                auc_score = test_auc
            
            # F1-Score
            from sklearn.metrics import f1_score
            f1 = f1_score(y_test, y_pred.flatten())
            
            # Matrice de confusion
            cm = confusion_matrix(y_test, y_pred.flatten())
            
            # Calcul des métriques personnalisées
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            self.metrics = {
                'test_loss': float(test_loss),
                'accuracy': float(test_accuracy),
                'precision': float(test_precision),
                'recall': float(test_recall),
                'f1_score': float(f1),
                'auc_roc': float(auc_score),
                'specificity': float(specificity),
                'sensitivity': float(sensitivity),
                'true_positives': int(tp),
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn)
            }
            
            logger.info(f"Modèle {self.model_id} - Accuracy: {test_accuracy:.4f}, AUC: {auc_score:.4f}, F1: {f1:.4f}")
            return self.metrics
            
        except Exception as e:
            logger.error(f"Erreur lors de l'évaluation du modèle {self.model_id}: {e}")
            raise e
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prédictions avec probabilités
        
        Args:
            X: Données d'entrée
            
        Returns:
            Tuple (predictions, probabilities)
        """
        if self.model is None or self.status != ModelStatus.TRAINED:
            raise ValueError(f"Le modèle {self.model_id} doit être entraîné avant la prédiction")
        
        try:
            # Prédictions probabilistes
            probabilities = self.model.predict(X, verbose=0)
            
            # Prédictions binaires (CORRECTION: seuil à 0.5, pas égalité à 0)
            predictions = (probabilities > 0.5).astype(int)
            
            return predictions.flatten(), probabilities.flatten()
            
        except Exception as e:
            logger.error(f"Erreur lors de la prédiction du modèle {self.model_id}: {e}")
            raise e
    
    def save_model(self) -> bool:
        """
        Sauvegarde complète du modèle
        
        Returns:
            True si sauvegarde réussie
        """
        if self.model is None:
            logger.warning(f"Aucun modèle à sauvegarder pour {self.model_id}")
            return False
        
        try:
            # Sauvegarde des poids
            weights_path = config.get_model_path(self.model_id, "weights")
            self.model.save_weights(weights_path)
            
            # Sauvegarde de l'architecture
            model_path = config.get_model_path(self.model_id, "model")
            self.model.save(model_path)
            
            # Sauvegarde des métadonnées
            metadata = {
                'model_id': self.model_id,
                'status': self.status,
                'metrics': self.metrics,
                'training_history': self.training_history,
                'config': {
                    'lstm_units': config.LSTM_UNITS,
                    'hidden_layers': config.HIDDEN_LAYERS,
                    'dropout_rate': config.DROPOUT_RATE,
                    'lookback_window': config.LOOKBACK_WINDOW
                }
            }
            
            metadata_path = config.get_model_path(self.model_id, "json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Modèle {self.model_id} sauvegardé: {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde du modèle {self.model_id}: {e}")
            return False
    
    def load_model(self) -> bool:
        """
        Chargement complet du modèle
        
        Returns:
            True si chargement réussi
        """
        try:
            # Chargement du modèle complet
            model_path = config.get_model_path(self.model_id, "model")
            if os.path.exists(model_path):
                self.model = tf.keras.models.load_model(model_path)
                self.status = ModelStatus.TRAINED
                logger.info(f"Modèle {self.model_id} chargé: {model_path}")
                
                # Chargement des métadonnées si disponibles
                metadata_path = config.get_model_path(self.model_id, "json")
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        self.metrics = metadata.get('metrics', {})
                        self.training_history = metadata.get('training_history', {})
                
                return True
            else:
                logger.warning(f"Modèle {self.model_id} non trouvé: {model_path}")
                return False
                
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle {self.model_id}: {e}")
            self.status = ModelStatus.ERROR
            return False
    
    def get_model_summary(self) -> str:
        """
        Résumé du modèle
        
        Returns:
            Résumé textuel du modèle
        """
        if self.model is None:
            return f"Modèle {self.model_id}: Non initialisé"
        
        summary_lines = []
        summary_lines.append(f"=== Modèle LSTM {self.model_id} ===")
        summary_lines.append(f"Status: {self.status}")
        summary_lines.append(f"Paramètres: {self.model.count_params():,}")
        
        if self.metrics:
            summary_lines.append("=== Métriques ===")
            for metric, value in self.metrics.items():
                if isinstance(value, float):
                    summary_lines.append(f"{metric}: {value:.4f}")
                else:
                    summary_lines.append(f"{metric}: {value}")
        
        return "\n".join(summary_lines)