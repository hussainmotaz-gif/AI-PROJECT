# 🚀 Système de Trading LSTM - Guide d'Installation et d'Utilisation

## 📋 Table des Matières
1. [Installation](#installation)
2. [Configuration](#configuration)
3. [Utilisation](#utilisation)
4. [Structure du Projet](#structure-du-projet)
5. [Corrections Apportées](#corrections-apportées)
6. [FAQ](#faq)

## 🛠️ Installation

### Prérequis
- Python 3.8+ 
- MetaTrader 5 installé (pour le trading réel)
- Au moins 8GB de RAM (16GB recommandé)
- GPU NVIDIA avec CUDA (optionnel mais recommandé)

### 1. Clonage et Environnement

```bash
# Créer un environnement virtuel
python -m venv lstm_trading_env

# Activer l'environnement
# Windows:
lstm_trading_env\Scripts\activate
# Linux/Mac:
source lstm_trading_env/bin/activate

# Installer les dépendances
pip install -r requirements.txt
```

### 2. Vérification de l'Installation

```bash
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
python -c "import MetaTrader5 as mt5; print('MT5 disponible')"
```

## ⚙️ Configuration

### 1. Configuration Principale

Modifiez le fichier `config.py` selon vos besoins :

```python
# Paramètres de trading
SYMBOL: str = "XAUUSD"  # Votre paire de trading
LOT_SIZE: float = 0.01  # Taille de position

# Paramètres du modèle
LOOKBACK_WINDOW: int = 20  # Fenêtre d'observation
N_MODELS: int = 5          # Nombre de modèles dans l'ensemble
EPOCHS: int = 100          # Époques d'entraînement
```

### 2. Structure des Dossiers

Le système créera automatiquement :
```
projet/
├── models/XAUUSD/     # Modèles entraînés
├── scalers/XAUUSD/    # Scalers de normalisation
└── logs/              # Fichiers de log
```

## 🎯 Utilisation

### 1. Entraînement des Modèles

```bash
# Entraînement basique
python run_trading_system.py train

# Entraînement personnalisé  
python run_trading_system.py train --symbol EURUSD --models 7 --epochs 150

# Forcer le réentraînement
python run_trading_system.py train --force
```

### 2. Backtesting

```bash
# Backtest simple
python run_trading_system.py backtest

# Backtest avec période spécifique
python run_trading_system.py backtest --start-date 2023-01-01 --end-date 2023-12-31

# Backtest avec graphiques et rapport
python run_trading_system.py backtest --plot --save-report
```

### 3. Génération de Signaux

```bash
# Signal actuel
python run_trading_system.py signal

# Signal avec sauvegarde
python run_trading_system.py signal --save --symbol GBPUSD
```

### 4. Simulation Live

```bash
# Simulation de 1 heure
python run_trading_system.py live --duration 60

# Simulation avec symbole spécifique
python run_trading_system.py live --symbol EURUSD --duration 120
```

### 5. Statut du Système

```bash
# Vérifier l'état du système
python run_trading_system.py status
```

## 📁 Structure du Projet

```
lstm_trading_system/
├── config.py              # Configuration centralisée
├── data_processor.py      # Traitement des données et features
├── lstm_model.py          # Modèle LSTM sophistiqué
├── ensemble_manager.py    # Gestionnaire d'ensemble de modèles
├── backtester.py         # Système de backtesting avancé
├── mt5_connector.py      # Connecteur MetaTrader 5
├── trading_system.py     # Système principal
├── run_trading_system.py # Interface en ligne de commande
├── requirements.txt      # Dépendances Python
└── README.md            # Ce guide
```

## ✅ Corrections Apportées

### 1. **Correction du Data Leakage**
- ❌ **Avant** : `dummy = np.round(returns + 0.5)` (prédiction du présent)
- ✅ **Après** : `target = (future_returns > 0).astype(int)` (prédiction du futur)

### 2. **Correction de la Double Standardisation**
- ❌ **Avant** : Standardisation dans feature_engineering() ET dans RNN_train()
- ✅ **Après** : Une seule standardisation après le split temporel

### 3. **Correction de la Logique de Vote**
- ❌ **Avant** : `buy = (pr1 + pr2 + pr3)[0][0] >= 1` (logique incorrecte)
- ✅ **Après** : Vote majoritaire avec seuil de confiance configurable

### 4. **Correction des Prédictions Binaires**
- ❌ **Avant** : `np.where(pred == 0, -1, 1)` (comparaison exacte)
- ✅ **Après** : `np.where(pred > 0.5, 1, 0)` (seuil de probabilité)

### 5. **Split Temporel**
- ❌ **Avant** : Split aléatoire 80/20
- ✅ **Après** : Split temporel 60/20/20 (train/val/test)

### 6. **Architecture LSTM Améliorée**
- ✅ Batch Normalization
- ✅ Régularisation L1/L2
- ✅ Dropout optimisé
- ✅ Plus d'époques d'entraînement

### 7. **Gestion d'Erreurs Complète**
- ✅ Try-catch robustes
- ✅ Logging détaillé
- ✅ Validation des données
- ✅ Vérifications de sanité

## 🔍 Métriques de Performance

Le système calcule automatiquement :

### Métriques de Trading
- **Rendement Total** : Performance globale
- **Rendement Annualisé** : Performance ajustée par le temps
- **Ratio de Sharpe** : Rendement ajusté par le risque
- **Drawdown Maximum** : Perte maximale
- **Taux de Réussite** : Pourcentage de trades gagnants

### Métriques ML
- **Précision/Recall** : Qualité des prédictions
- **F1-Score** : Harmonie précision/recall
- **AUC-ROC** : Capacité de discrimination
- **Matrice de Confusion** : Analyse détaillée des erreurs

## 🚨 Points d'Attention

### 1. Trading Réel
- **IMPORTANT** : Testez toujours sur compte DEMO avant le réel
- Vérifiez vos paramètres de risque
- Surveillez les positions ouvertes

### 2. Performance
- L'entraînement de 5+ modèles peut prendre 30-60 minutes
- Utilisez un GPU pour accélérer l'entraînement
- Surveillez l'utilisation de la RAM

### 3. Données
- Assurez-vous d'avoir une connexion stable pour yfinance
- Les données MT5 nécessitent une connexion au broker
- Vérifiez la qualité des données historiques

## 📊 Exemple de Workflow Complet

```bash
# 1. Vérifier le statut
python run_trading_system.py status

# 2. Entraîner les modèles
python run_trading_system.py train --symbol XAUUSD --models 5

# 3. Faire un backtest
python run_trading_system.py backtest --plot --save-report

# 4. Générer un signal actuel
python run_trading_system.py signal --save

# 5. (Optionnel) Simulation live
python run_trading_system.py live --duration 30
```

## 🆘 FAQ

### Q: L'entraînement est très lent, que faire ?
**R:** Installez CUDA pour GPU, réduisez le nombre de modèles ou d'époques.

### Q: Erreur "Pas assez de données"
**R:** Augmentez la période de données ou changez MIN_DATA_POINTS dans config.py

### Q: Les performances en backtest sont trop bonnes
**R:** C'est normal après correction du data leakage. Vérifiez sur données out-of-sample.

### Q: Connexion MT5 échoue
**R:** Vérifiez que MT5 est ouvert, compte connecté, et paramètres de connexion corrects.

### Q: Comment adapter à d'autres instruments ?
**R:** Changez SYMBOL dans config.py et ajustez LOT_SIZE selon l'instrument.

## 🤝 Support

Pour des questions ou problèmes :
1. Vérifiez les logs dans le dossier `logs/`
2. Consultez ce guide
3. Vérifiez la configuration dans `config.py`

## 🏁 Conclusion

Ce système corrige tous les problèmes identifiés dans le script original et offre :
- ✅ Architecture robuste et modulaire
- ✅ Gestion d'erreurs complète
- ✅ Métriques de performance avancées
- ✅ Interface utilisateur intuitive
- ✅ Documentation complète

**Bonne chance avec votre trading ! 🚀📈**