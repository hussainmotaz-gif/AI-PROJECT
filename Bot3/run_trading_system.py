# run_trading_system.py
"""
Script principal pour l'exécution du système de trading LSTM
"""

import argparse
import sys
import os
from datetime import datetime
import logging
import logging.config

# Ajout du chemin local pour les imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import config, LOGGING_CONFIG
from trading_system import TradingSystem
from mt5_connector import MT5DataProvider

# Configuration du logging
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger('trading_system.runner')

def train_models_command(args):
    """Commande d'entraînement des modèles"""
    print("=" * 60)
    print("ENTRAÎNEMENT DES MODÈLES LSTM")
    print("=" * 60)
    
    # Initialisation du système
    trading_system = TradingSystem()
    
    try:
        # Configuration personnalisée si fournie
        if args.symbol:
            config.SYMBOL = args.symbol
        if args.models:
            config.N_MODELS = args.models
        if args.epochs:
            config.EPOCHS = args.epochs
        
        print(f"Configuration:")
        print(f"- Symbole: {config.SYMBOL}")
        print(f"- Nombre de modèles: {config.N_MODELS}")
        print(f"- Époques: {config.EPOCHS}")
        print(f"- Fenêtre d'observation: {config.LOOKBACK_WINDOW}")
        print()
        
        # Entraînement
        result = trading_system.train_models(force_retrain=args.force)
        
        if result['status'] == 'success':
            print("✅ Entraînement terminé avec succès!")
            print(f"Modèles sélectionnés: {result.get('selected_models', 'N/A')}")
            
            # Affichage des métriques si disponibles
            metrics = result.get('ensemble_metrics', {})
            if metrics:
                print(f"\nMétriques de l'ensemble:")
                print(f"- Précision: {metrics.get('ensemble_accuracy', 0):.1%}")
                print(f"- F1-Score: {metrics.get('ensemble_f1', 0):.3f}")
                print(f"- AUC-ROC: {metrics.get('ensemble_auc', 0):.3f}")
        
        elif result['status'] == 'loaded':
            print("ℹ️  Modèles existants chargés")
            print("Utilisez --force pour forcer le réentraînement")
        
        else:
            print("❌ Échec de l'entraînement")
            return 1
    
    except Exception as e:
        print(f"❌ Erreur: {e}")
        logger.error(f"Erreur entraînement: {e}")
        return 1
    
    finally:
        trading_system.cleanup()
    
    return 0

def backtest_command(args):
    """Commande de backtesting"""
    print("=" * 60)
    print("BACKTESTING DU SYSTÈME")
    print("=" * 60)
    
    trading_system = TradingSystem()
    
    try:
        # Configuration
        if args.symbol:
            config.SYMBOL = args.symbol
        
        print(f"Configuration du backtest:")
        print(f"- Symbole: {config.SYMBOL}")
        print(f"- Date début: {args.start_date or 'Auto'}")
        print(f"- Date fin: {args.end_date or 'Auto'}")
        print()
        
        # Chargement des modèles
        if not trading_system.load_models():
            print("❌ Impossible de charger les modèles")
            print("Entraînez d'abord les modèles avec: python run_trading_system.py train")
            return 1
        
        print("✅ Modèles chargés")
        
        # Exécution du backtest
        result = trading_system.run_backtest(
            start_date=args.start_date,
            end_date=args.end_date
        )
        
        if result['status'] == 'success':
            metrics = result['backtest_results']['metrics']
            
            print("=" * 60)
            print("RÉSULTATS DU BACKTEST")
            print("=" * 60)
            print(f"🎯 Rendement total: {metrics['total_return']:+.2f}%")
            print(f"📈 Rendement annualisé: {metrics['annualized_return']:+.2f}%")
            print(f"📊 Volatilité: {metrics['volatility']:.2f}%")
            print(f"📉 Drawdown maximum: {metrics['max_drawdown']:.2f}%")
            print(f"⚡ Ratio de Sharpe: {metrics['sharpe_ratio']:.3f}")
            print(f"🎲 Taux de réussite: {metrics['win_rate']:.1f}%")
            print(f"🔢 Nombre de trades: {metrics['total_trades']}")
            print(f"💰 Profit factor: {metrics['profit_factor']:.2f}")
            print()
            
            # Sauvegarde du rapport si demandé
            if args.save_report:
                report_path = f"backtest_report_{config.SYMBOL}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write(result['report'])
                print(f"📄 Rapport sauvegardé: {report_path}")
            
            # Génération des graphiques si demandé
            if args.plot:
                try:
                    plot_path = f"backtest_plot_{config.SYMBOL}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    trading_system.backtester.plot_results(save_path=plot_path, show=False)
                    print(f"📊 Graphiques sauvegardés: {plot_path}")
                except Exception as e:
                    print(f"⚠️  Erreur génération graphiques: {e}")
        
        else:
            print("❌ Échec du backtest")
            return 1
    
    except Exception as e:
        print(f"❌ Erreur: {e}")
        logger.error(f"Erreur backtest: {e}")
        return 1
    
    finally:
        trading_system.cleanup()
    
    return 0

def signal_command(args):
    """Commande de génération de signal"""
    print("=" * 60)
    print("GÉNÉRATION DE SIGNAL")
    print("=" * 60)
    
    trading_system = TradingSystem()
    
    try:
        # Configuration
        if args.symbol:
            config.SYMBOL = args.symbol
        
        print(f"Symbole: {config.SYMBOL}")
        
        # Chargement des modèles
        if not trading_system.load_models():
            print("❌ Impossible de charger les modèles")
            return 1
        
        print("✅ Modèles chargés")
        
        # Génération du signal
        signal_info = trading_system.generate_current_signal()
        
        # Affichage du résultat
        signal_names = {-1: "🔴 SELL", 0: "⚪ HOLD", 1: "🟢 BUY"}
        signal_name = signal_names.get(signal_info['signal'], "❓ UNKNOWN")
        
        print("=" * 40)
        print("SIGNAL GÉNÉRÉ")
        print("=" * 40)
        print(f"📅 Timestamp: {signal_info['timestamp']}")
        print(f"💰 Prix actuel: ${signal_info['current_price']:.2f}")
        print(f"🎯 Signal: {signal_name}")
        print(f"🔍 Confiance: {signal_info['confidence']:.1%}")
        print(f"👍 Votes BUY: {signal_info['buy_votes']}")
        print(f"👎 Votes SELL: {signal_info['sell_votes']}")
        print(f"🤖 Modèles utilisés: {len(signal_info['models_used'])}")
        print(f"📊 Probabilité moyenne: {signal_info['ensemble_probability']:.3f}")
        
        # Sauvegarde si demandé
        if args.save:
            import json
            signal_file = f"signal_{config.SYMBOL}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # Conversion des types non-JSON
            save_data = signal_info.copy()
            save_data['timestamp'] = str(save_data['timestamp'])
            
            with open(signal_file, 'w') as f:
                json.dump(save_data, f, indent=2)
            print(f"💾 Signal sauvegardé: {signal_file}")
    
    except Exception as e:
        print(f"❌ Erreur: {e}")
        logger.error(f"Erreur génération signal: {e}")
        return 1
    
    finally:
        trading_system.cleanup()
    
    return 0

def live_command(args):
    """Commande de trading live (simulation)"""
    print("=" * 60)
    print("SIMULATION DE TRADING LIVE")
    print("=" * 60)
    
    trading_system = TradingSystem()
    
    try:
        # Configuration
        if args.symbol:
            config.SYMBOL = args.symbol
        
        print(f"Configuration:")
        print(f"- Symbole: {config.SYMBOL}")
        print(f"- Durée: {args.duration} minutes")
        print()
        
        # Chargement des modèles
        if not trading_system.load_models():
            print("❌ Impossible de charger les modèles")
            return 1
        
        print("✅ Modèles chargés")
        print("🚀 Démarrage de la simulation...")
        print("(Appuyez sur Ctrl+C pour arrêter)")
        print()
        
        # Simulation live
        trading_system.run_live_trading_simulation(duration_minutes=args.duration)
    
    except KeyboardInterrupt:
        print("\n⏹️  Simulation arrêtée par l'utilisateur")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        logger.error(f"Erreur simulation live: {e}")
        return 1
    
    finally:
        trading_system.cleanup()
    
    return 0

def status_command(args):
    """Commande d'affichage du statut"""
    print("=" * 60)
    print("STATUT DU SYSTÈME")
    print("=" * 60)
    
    trading_system = TradingSystem()
    
    try:
        # Tentative de chargement des modèles
        models_loaded = trading_system.load_models()
        
        # Récupération du statut
        status = trading_system.get_system_status()
        
        print(f"🤖 Modèles entraînés: {'✅ Oui' if status['is_trained'] else '❌ Non'}")
        print(f"📊 Points de données: {status['current_data_points']}")
        print(f"🕐 Dernière mise à jour: {status['last_data_update'] or 'Jamais'}")
        print()
        
        print("Configuration:")
        config_info = status['config']
        print(f"- Symbole: {config_info['symbol']}")
        print(f"- Fenêtre d'observation: {config_info['lookback_window']}")
        print(f"- Nombre de modèles: {config_info['n_models']}")
        print(f"- Seuil d'ensemble: {config_info['ensemble_threshold']:.1%}")
        
        if models_loaded and 'ensemble_info' in status:
            ensemble_info = status['ensemble_info']
            print()
            print("Informations de l'ensemble:")
            print(f"- Modèles sélectionnés: {ensemble_info['selected_models']}")
            
            metrics = ensemble_info.get('ensemble_metrics', {})
            if metrics:
                print(f"- Précision: {metrics.get('ensemble_accuracy', 0):.1%}")
                print(f"- F1-Score: {metrics.get('ensemble_f1', 0):.3f}")
        
        # Vérification des fichiers
        print()
        print("Fichiers du système:")
        models_dir = os.path.join(config.BASE_PATH, config.MODELS_DIR, config.SYMBOL)
        scalers_dir = os.path.join(config.BASE_PATH, config.SCALERS_DIR, config.SYMBOL)
        
        model_files = []
        if os.path.exists(models_dir):
            model_files = [f for f in os.listdir(models_dir) if f.endswith('.h5')]
        
        scaler_files = []
        if os.path.exists(scalers_dir):
            scaler_files = [f for f in os.listdir(scalers_dir) if f.endswith('.joblib')]
        
        print(f"- Modèles: {len(model_files)} fichiers")
        print(f"- Scalers: {len(scaler_files)} fichiers")
    
    except Exception as e:
        print(f"❌ Erreur: {e}")
        logger.error(f"Erreur status: {e}")
        return 1
    
    finally:
        trading_system.cleanup()
    
    return 0

def main():
    """Fonction principale avec interface en ligne de commande"""
    parser = argparse.ArgumentParser(
        description="Système de Trading LSTM Avancé",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Exemples d'utilisation:
            python run_trading_system.py train --symbol XAUUSD --models 5
            python run_trading_system.py backtest --plot --save-report
            python run_trading_system.py signal --symbol EURUSD --save
            python run_trading_system.py live --duration 30
            python run_trading_system.py status
                    """
    )
    
    # Arguments globaux
    parser.add_argument('--symbol', type=str, help='Symbole de trading (ex: XAUUSD)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Mode verbeux')
    
    # Sous-commandes
    subparsers = parser.add_subparsers(dest='command', help='Commandes disponibles')
    
    # Commande train
    train_parser = subparsers.add_parser('train', help='Entraîner les modèles')
    train_parser.add_argument('--models', type=int, help='Nombre de modèles')
    train_parser.add_argument('--epochs', type=int, help="Nombre d'époques")
    train_parser.add_argument('--force', action='store_true', help='Forcer le réentraînement')
    
    # Commande backtest
    backtest_parser = subparsers.add_parser('backtest', help='Exécuter un backtest')
    backtest_parser.add_argument('--start-date', type=str, help='Date de début (YYYY-MM-DD)')
    backtest_parser.add_argument('--end-date', type=str, help='Date de fin (YYYY-MM-DD)')
    backtest_parser.add_argument('--plot', action='store_true', help='Générer les graphiques')
    backtest_parser.add_argument('--save-report', action='store_true', help='Sauvegarder le rapport')
    
    # Commande signal
    signal_parser = subparsers.add_parser('signal', help='Générer un signal actuel')
    signal_parser.add_argument('--save', action='store_true', help='Sauvegarder le signal')
    
    # Commande live
    live_parser = subparsers.add_parser('live', help='Simulation de trading live')
    live_parser.add_argument('--duration', type=int, default=60, help='Durée en minutes')
    
    # Commande status
    status_parser = subparsers.add_parser('status', help='Afficher le statut du système')
    
    # Parse des arguments
    args = parser.parse_args()
    
    # Configuration du logging si verbose
    if args.verbose:
        logging.getLogger('trading_system').setLevel(logging.DEBUG)
    
    # Exécution de la commande
    if args.command == 'train':
        return train_models_command(args)
    elif args.command == 'backtest':
        return backtest_command(args)
    elif args.command == 'signal':
        return signal_command(args)
    elif args.command == 'live':
        return live_command(args)
    elif args.command == 'status':
        return status_command(args)
    else:
        parser.print_help()
        return 1

# if __name__ == "__main__":
#     exit_code = main()
#     sys.exit(exit_code)