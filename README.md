# Trading Bot — Ensemble RNN (XAUUSD / MetaTrader 5)

Bot de trading algorithmique qui entraîne un **ensemble de réseaux de neurones récurrents (LSTM/GRU)** pour prédire la direction du prix sur l'or (XAUUSD), puis exécute des ordres en temps réel via **MetaTrader 5**. Le projet inclut un pipeline complet : récupération de données, feature engineering, entraînement, sélection de modèles, backtest avec coûts réalistes, et exécution live.

> Inspiré et guidé par *"Python for Finance and Algorithmic Trading"* (sections RNN / ensemble).

---

## 1. Structure du projet

```
Projet_Ai/Trading_bot/
│   backtest_models.py     # Backtest historique détaillé (rapport, métriques, CSV)
│   main.py                # Point d'entrée : lance Bot2
│   requirements.txt       # Dépendances Python
│
├───Bot1/, Bot3/, Bot4/     # Anciennes versions / variantes du bot (non actives)
│
├───Bot2/                   # Bot actif utilisé par main.py
│   │   run.py              # Cœur du système (data, features, modèle, train, live)
│   │
│   ├───data/                # (vide — réservé à un cache de données éventuel)
│   ├───models/              # Modèles .keras + scaler + métadonnées sauvegardés
│   └───__pycache__/
│
├───.venv/                  # Environnement virtuel Python
├───Weights_RNN/             # Poids RNN sauvegardés (expérimentations BTCUSD / XAUUSD)
└───.idea/, .vscode/        # Configuration des IDE
```

Le point d'entrée du projet est **`main.py`**, qui ajoute `Bot2` au `sys.path` et appelle `Bot2.run.run()`.

```python
sys.path.append(os.path.abspath("Bot2"))
from Bot2.run import run
run()
```

`backtest_models.py`, à la racine, importe directement les fonctions de `Bot2/run.py` (via `from run import ...`, donc à exécuter **depuis le dossier `Bot2`**) pour rejouer l'historique avec le modèle déjà entraîné et produire un rapport détaillé.

---

## 2. Environnement de développement

| Élément | Détail |
|---|---|
| **Langage** | Python (testé avec CPython 3.11 — fichiers `.pyc` compilés en `cpython-311`) |
| **IDE** | JetBrains PyCharm (présence de `.idea/`) + VS Code (`.vscode/`) |
| **Environnement virtuel** | `.venv/` (à la racine du projet) |
| **OS de développement** | Windows (chemins `E:\Projet_Ai\Trading_bot\...`, terminal `set VAR=valeur && py script.py`) |
| **Plateforme de trading** | MetaTrader 5 (terminal installé localement, requis pour `MetaTrader5` package) |
| **Gestion de version** | Git (`.gitignore` présent dans `.idea` et `.venv`) |

### Installation

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

> ⚠️ Le package `metatrader5` nécessite que le **terminal MetaTrader 5 soit installé et lancé** sur la machine (Windows uniquement — ce package n'est pas disponible sous Linux/macOS).

### Exécution

```bash
# Lancer le bot (entraînement si aucun modèle n'existe, puis screener live)
set TF_ENABLE_ONEDNN_OPTS=0 && py main.py

# Lancer un backtest détaillé sur un modèle déjà entraîné
cd Bot2
py ../backtest_models.py
```

### Dépendances principales (`requirements.txt`)

| Catégorie | Librairies |
|---|---|
| Trading / données marché | `MetaTrader5`, `pandas`, `numpy` |
| Machine Learning | `tensorflow`, `keras`, `scikit-learn`, `joblib` |
| Indicateurs techniques | `TA-Lib` *(et `pandas_ta` en optionnel, avec fallback manuel)* |
| Backtest | `backtesting` |
| Visualisation / reporting | `matplotlib`, `seaborn`, `plotly`, `streamlit` |
| Utilitaires | `tqdm`, `python-dateutil`, `pytz` |

---

## 3. Architecture fonctionnelle (`Bot2/run.py`)

### 3.1 Configuration (`CONFIG`)

Toute la configuration du bot est centralisée dans un dictionnaire `CONFIG` en tête de fichier :

| Paramètre | Valeur | Rôle |
|---|---|---|
| `SYMBOL` | `"XAUUSD"` | Instrument tradé |
| `TIMEFRAME` | `M30` | Unité de temps des bougies |
| `LOOKBACK` | `64` | Nombre de barres passées utilisées en entrée du RNN |
| `N_MODELS` | `5` (100 en production) | Nombre de modèles entraînés (bagging) |
| `TOP_K` | `3` | Nombre de modèles retenus dans l'ensemble final |
| `EPOCHS` / `BATCH_SIZE` / `PATIENCE` | `50` / `64` / `8` | Hyperparamètres d'entraînement |
| `LABEL_HORIZON` | `4` barres | Horizon de prédiction (2h sur M30) |
| `LABEL_MIN_MOVE` | `0.1 %` | Mouvement minimum pour labelliser un "UP" (filtre le bruit) |
| `THRESH_BUY` / `THRESH_SELL` | `0.58` / `0.42` | Seuils de probabilité asymétriques pour décider BUY/SELL |
| `CLASS_WEIGHT_SELL` / `_BUY` | `2.0` / `1.0` | Pondération de classe pour corriger le biais BUY |
| `USE_TREND_FILTER` | `True` | Active le filtre de tendance MA200 |
| `RISK_PER_TRADE_PCT` | `0.1 %` | Risque par trade pour le sizing |
| `DRY_RUN` | `False` | Si `True`, simule les ordres sans les envoyer |

### 3.2 Pipeline de données et features (`fetch_mt5`, `add_features_full`)

- **`fetch_mt5`** : récupère l'historique des bougies par tranches de 180 jours via `mt5.copy_rates_range`, pour éviter les limitations de l'API MT5 sur de longues périodes.
- **`add_features_full`** : calcule 15 features à partir de l'OHLCV brut :
  - Rendements log : `ret_1`, `ret_4`, `ret_96`
  - Indicateurs techniques : `atr14`, `rsi14`, `ma50`, `ma200`, `rv_96` (volatilité réalisée)
  - Patterns de bougies : `engulfing`, `doji`
  - Features additionnelles : `distance_ma50`, `distance_ma200` (position relative aux moyennes), `trend_slope` (pente de régression linéaire sur 20 barres), `vol_ratio` (volume relatif), `momentum` (variation du RSI)
  - Si `pandas_ta` n'est pas installé, un calcul manuel équivalent (ATR, RSI, patterns) prend le relai (`TA_AVAILABLE`).

### 3.3 Labellisation (`add_labels`)

Label binaire construit sur un horizon de 4 barres futures : `1` (UP) seulement si le rendement futur dépasse `+0.1 %`, sinon `0`. Cela réduit le bruit par rapport à une prédiction barre-à-barre classique.

### 3.4 Transformation 2D → 3D (`X_3d_RNN`)

Convertit la matrice de features `(n_barres, n_features)` en séquences glissantes `(n_échantillons, LOOKBACK, n_features)` consommables par un RNN.

### 3.5 Modèle (`build_rnn`)

Réseau séquentiel Keras configurable :
- 1 à plusieurs couches **LSTM ou GRU** (type tiré aléatoirement par modèle)
- `Dropout` après chaque couche récurrente
- Couche de sortie `Dense(1, activation='sigmoid')` → probabilité de hausse
- Optimiseur `Adam` avec `clipnorm=1.0` (gradient clipping pour la stabilité)
- Callbacks : `EarlyStopping` (patience configurable) + `ReduceLROnPlateau`

### 3.6 Entraînement par ensemble (`train_ensemble`, `pipeline_train_run`)

Stratégie de **bagging** :
1. Split chronologique train / validation / test (80 % train interne, 20 % test global)
2. Standardisation des features (`StandardScaler`, fit uniquement sur le train)
3. Pour chaque modèle (`N_MODELS`) :
   - Échantillon bootstrap du train set
   - Hyperparamètres tirés aléatoirement (`units`, `dropout`, `learning_rate`, type RNN)
   - Entraînement avec `class_weight` asymétrique
   - Évaluation sur le test set via un backtest réaliste (`backtest_with_costs`)
4. **`select_and_save`** : tri des modèles par ratio rendement/drawdown, conservation du `TOP_K`, sauvegarde de :
   - `scaler.joblib`, `feature_cols.joblib`, `ensemble_meta.joblib` (modèles sélectionnés, seuils, config label)
   - chaque modèle `.keras` individuellement

### 3.7 Vote d'ensemble (`ensemble_vote_majority`, `ensemble_predict_with_cached`)

Les modèles sélectionnés votent indépendamment (BUY/SELL) à partir d'un seuil de probabilité. Par défaut, **l'unanimité** est requise (`required_majority = len(predict_fns)`) pour émettre un signal — sinon le signal est neutre (`-1`, flat).

### 3.8 Filtre de tendance (`apply_trend_filter`)

Filtre post-prédiction : un signal BUY n'est conservé que si `close > MA200`, un signal SELL seulement si `close < MA200`. Tout signal contre-tendance est neutralisé.

### 3.9 Backtest (`backtest_with_costs`)

Simule le P&L en tenant compte de coûts réalistes : spread, slippage, commission round-trip, conversion en valeur de pip via les infos symbole MT5. Calcule : rendement total, drawdown max, Sharpe annualisé, profit factor, ratio rendement/drawdown, nombre de trades, % de temps flat.

### 3.10 Exécution live (`live_screener_loop`, `send_order_market`)

Boucle infinie (intervalle de 15 s) qui :
1. Récupère la dernière fenêtre de données valide (`get_latest_window_safe_v3`)
2. Calcule le vote d'ensemble + applique le filtre de tendance
3. Déduplique (évite de retrader la même bougie avec le même signal)
4. Calcule SL/TP basés sur l'ATR, envoie l'ordre via `mt5.order_send` (ou simule si `DRY_RUN=True`)
5. Journalise chaque itération dans `live_iteration_log.csv`

### 3.11 Point d'entrée (`run`)

```python
def run():
    if not os.path.exists(".../ensemble_meta.joblib"):
        pipeline_train_run(CONFIG)   # entraîne si aucun modèle n'existe
    live_screener_loop(CONFIG)       # puis lance le screener live
```

---

## 4. Backtest historique détaillé (`backtest_models.py`)

Script complémentaire pour rejouer un modèle déjà entraîné sur l'historique complet et produire un rapport enrichi :

- Charge `scaler`, `feature_cols`, `ensemble_meta` depuis `Bot2/models/`
- Recharge les seuils BUY/SELL depuis `ensemble_meta` (cohérence avec l'entraînement)
- Génère les signaux historiques par vote unanime, puis applique le filtre MA200
- Calcule des métriques étendues : `win_rate`, gain/perte moyens, ratio gain/perte, % flat
- Produit un **rapport mensuel** (P&L par mois, ASCII bar chart)
- Exporte un CSV enrichi (`backtest_results_detailed_optimized.csv`) avec OHLCV, indicateurs, signal, position et P&L cumulé

---

## 5. Notes techniques

- **Persistance des modèles** : chaque run d'entraînement écrase/ajoute des fichiers `.keras` dans `Bot2/models/`. Le triplet `scaler.joblib` / `feature_cols.joblib` / `ensemble_meta.joblib` doit toujours correspondre à la même session d'entraînement (les features et la normalisation doivent être cohérentes entre entraînement et inférence).
- **Mode `QUICK_TEST`** : permet de valider rapidement le pipeline avec un nombre réduit de modèles et d'epochs (`QUICK_N_MODELS=3`, `QUICK_EPOCHS=3`).
- **Mode `DRY_RUN`** : recommandé avant tout déploiement réel pour vérifier les signaux sans envoyer d'ordres.
- **Limitation plateforme** : le package `MetaTrader5` ne fonctionne que sous Windows avec le terminal MT5 installé — ce projet n'est donc pas exécutable tel quel sous Linux/macOS sans une couche d'émulation (Wine) ou un environnement Windows.

---

## 6. Auteur / Contexte

Projet académique de trading algorithmique combinant réseaux de neurones récurrents (LSTM/GRU), apprentissage en ensemble (bagging) et exécution automatisée via l'API MetaTrader 5.
