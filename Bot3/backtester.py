# backtester.py
"""
Système de backtesting avancé avec gestion des risques et métriques complètes
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import logging
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass

from config import config, backtest_config, TradingSignals

logger = logging.getLogger('trading_system.backtester')

@dataclass
class Trade:
    """Représentation d'un trade"""
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    signal: int  # 1 pour BUY, -1 pour SELL
    size: float
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    commission: float = 0.0
    is_open: bool = True
    exit_reason: str = "open"

class AdvancedBacktester:
    """
    Backtester sophistiqué avec gestion des risques et métriques avancées
    """
    
    def __init__(self):
        self.trades: List[Trade] = []
        self.portfolio_value: List[float] = []
        self.dates: List[datetime] = []
        self.signals_history: List[int] = []
        self.positions_history: List[float] = []
        self.cash_history: List[float] = []
        
        # Métriques de performance
        self.metrics: Dict[str, float] = {}
        
        # État du portefeuille
        self.cash = backtest_config.INITIAL_CAPITAL
        self.position = 0.0
        self.current_trade: Optional[Trade] = None
        
    def reset(self):
        """Réinitialisation du backtester"""
        self.trades = []
        self.portfolio_value = []
        self.dates = []
        self.signals_history = []
        self.positions_history = []
        self.cash_history = []
        self.metrics = {}
        self.cash = backtest_config.INITIAL_CAPITAL
        self.position = 0.0
        self.current_trade = None
        
    def calculate_position_size(self, price: float, signal: int, 
                              volatility: float = None) -> float:
        """
        Calcul de la taille de position avec gestion des risques
        
        Args:
            price: Prix actuel
            signal: Signal de trading
            volatility: Volatilité pour ajustement de taille
            
        Returns:
            Taille de position
        """
        # Capital disponible pour le trade
        available_capital = self.cash * backtest_config.RISK_PER_TRADE
        
        # Taille de base
        base_size = available_capital / price
        
        # Ajustement selon la volatilité si fournie
        if volatility is not None:
            # Réduction de la taille si volatilité élevée
            volatility_multiplier = min(1.0, 0.02 / max(volatility, 0.001))
            base_size *= volatility_multiplier
        
        # Limites de position
        max_position_value = self.cash * 0.95  # Maximum 95% du capital
        max_size = max_position_value / price
        
        return min(base_size, max_size)
    
    def execute_trade(self, current_time: datetime, price: float, 
                     signal: int, volatility: float = None) -> bool:
        """
        Exécution d'un trade avec gestion des risques
        
        Args:
            current_time: Timestamp actuel
            price: Prix actuel
            signal: Signal de trading
            volatility: Volatilité actuelle
            
        Returns:
            True si trade exécuté
        """
        # Vérification de la liquidité
        if self.cash < price * config.LOT_SIZE:
            return False
        
        # Fermeture de la position existante si signal opposé
        if self.current_trade is not None:
            if (self.current_trade.signal == TradingSignals.BUY and signal == TradingSignals.SELL) or \
               (self.current_trade.signal == TradingSignals.SELL and signal == TradingSignals.BUY):
                self._close_trade(current_time, price, "signal_reversal")
        
        # Ouverture nouvelle position si pas de position ou signal HOLD
        if signal != TradingSignals.HOLD and self.current_trade is None:
            position_size = self.calculate_position_size(price, signal, volatility)
            
            if position_size > 0:
                # Commission
                commission = position_size * price * backtest_config.COMMISSION
                
                # Vérification de la liquidité après commission
                total_cost = position_size * price + commission
                if total_cost <= self.cash:
                    # Création du trade
                    trade = Trade(
                        entry_time=current_time,
                        exit_time=None,
                        entry_price=price,
                        exit_price=None,
                        signal=signal,
                        size=position_size,
                        commission=commission
                    )
                    
                    # Mise à jour du cash et position
                    self.cash -= total_cost
                    self.position = position_size if signal == TradingSignals.BUY else -position_size
                    self.current_trade = trade
                    
                    return True
        
        return False
    
    def _close_trade(self, exit_time: datetime, exit_price: float, 
                    exit_reason: str) -> None:
        """
        Fermeture d'un trade
        
        Args:
            exit_time: Timestamp de sortie
            exit_price: Prix de sortie
            exit_reason: Raison de la fermeture
        """
        if self.current_trade is None:
            return
        
        trade = self.current_trade
        
        # Commission de sortie
        exit_commission = abs(self.position) * exit_price * backtest_config.COMMISSION
        
        # Calcul du P&L
        if trade.signal == TradingSignals.BUY:
            # Position longue
            pnl = self.position * (exit_price - trade.entry_price) - exit_commission
        else:
            # Position courte
            pnl = abs(self.position) * (trade.entry_price - exit_price) - exit_commission
        
        # Slippage
        slippage_cost = abs(self.position) * exit_price * backtest_config.SLIPPAGE
        pnl -= slippage_cost
        
        # Mise à jour du trade
        trade.exit_time = exit_time
        trade.exit_price = exit_price
        trade.pnl = pnl
        trade.pnl_pct = pnl / (abs(self.position) * trade.entry_price) * 100
        trade.is_open = False
        trade.exit_reason = exit_reason
        trade.commission += exit_commission
        
        # Mise à jour du cash
        if trade.signal == TradingSignals.BUY:
            self.cash += self.position * exit_price
        else:
            self.cash += abs(self.position) * trade.entry_price + pnl
        
        self.cash += pnl
        
        # Réinitialisation de la position
        self.position = 0.0
        self.trades.append(trade)
        self.current_trade = None
    
    def check_stop_loss_take_profit(self, current_time: datetime, 
                                   current_price: float) -> bool:
        """
        Vérification des stop-loss et take-profit
        
        Args:
            current_time: Timestamp actuel
            current_price: Prix actuel
            
        Returns:
            True si position fermée
        """
        if self.current_trade is None:
            return False
        
        trade = self.current_trade
        
        # Calcul des niveaux
        if trade.signal == TradingSignals.BUY:
            # Position longue
            stop_loss_level = trade.entry_price * (1 - backtest_config.STOP_LOSS_PCT)
            take_profit_level = trade.entry_price * (1 + backtest_config.TAKE_PROFIT_PCT)
            
            if current_price <= stop_loss_level:
                self._close_trade(current_time, current_price, "stop_loss")
                return True
            elif current_price >= take_profit_level:
                self._close_trade(current_time, current_price, "take_profit")
                return True
                
        else:
            # Position courte
            stop_loss_level = trade.entry_price * (1 + backtest_config.STOP_LOSS_PCT)
            take_profit_level = trade.entry_price * (1 - backtest_config.TAKE_PROFIT_PCT)
            
            if current_price >= stop_loss_level:
                self._close_trade(current_time, current_price, "stop_loss")
                return True
            elif current_price <= take_profit_level:
                self._close_trade(current_time, current_price, "take_profit")
                return True
        
        return False
    
    def update_portfolio_value(self, current_time: datetime, current_price: float):
        """
        Mise à jour de la valeur du portefeuille
        
        Args:
            current_time: Timestamp actuel
            current_price: Prix actuel
        """
        portfolio_value = self.cash
        
        # Ajout de la valeur de la position ouverte
        if self.current_trade is not None:
            if self.current_trade.signal == TradingSignals.BUY:
                position_value = self.position * current_price
            else:
                # Position courte: valeur = capital initial + P&L non réalisé
                unrealized_pnl = abs(self.position) * (self.current_trade.entry_price - current_price)
                position_value = abs(self.position) * self.current_trade.entry_price + unrealized_pnl
            
            portfolio_value += position_value - abs(self.position) * self.current_trade.entry_price
        
        # Enregistrement
        self.portfolio_value.append(portfolio_value)
        self.dates.append(current_time)
        self.cash_history.append(self.cash)
        self.positions_history.append(self.position)
    
    def run_backtest(self, data: pd.DataFrame, signals: List[int], 
                    start_date: datetime = None, end_date: datetime = None) -> Dict[str, Any]:
        """
        Exécution complète du backtest
        
        Args:
            data: DataFrame avec OHLCV et indicateurs
            signals: Liste des signaux de trading
            start_date: Date de début (optionnel)
            end_date: Date de fin (optionnel)
            
        Returns:
            Résultats du backtest
        """
        logger.info("Début du backtest")
        self.reset()
        
        # Filtrage des données par date si spécifié
        if start_date or end_date:
            mask = pd.Series(True, index=data.index)
            if start_date:
                mask &= (data.index >= start_date)
            if end_date:
                mask &= (data.index <= end_date)
            data = data[mask]
            signals = [signals[i] for i in range(len(signals)) if mask.iloc[i]]
        
        # Vérification de la cohérence des données
        if len(signals) != len(data):
            min_len = min(len(signals), len(data))
            signals = signals[:min_len]
            data = data[:min_len]
            logger.warning(f"Tailles ajustées à {min_len} points")
        
        volatility_column = 'volatility_20' if 'volatility_20' in data.columns else None
        
        # Boucle principale du backtest
        for i, (timestamp, row) in enumerate(data.iterrows()):
            current_price = row['close']
            signal = signals[i]
            volatility = row[volatility_column] if volatility_column else None
            
            # Vérification stop-loss/take-profit
            self.check_stop_loss_take_profit(timestamp, current_price)
            
            # Exécution des trades
            self.execute_trade(timestamp, current_price, signal, volatility)
            
            # Mise à jour du portefeuille
            self.update_portfolio_value(timestamp, current_price)
            self.signals_history.append(signal)
        
        # Fermeture de la position finale si ouverte
        if self.current_trade is not None:
            final_price = data.iloc[-1]['close']
            final_time = data.index[-1]
            self._close_trade(final_time, final_price, "end_of_data")
        
        # Calcul des métriques
        metrics = self.calculate_metrics()
        
        logger.info(f"Backtest terminé: {len(self.trades)} trades, "
                   f"rendement total: {metrics.get('total_return', 0):.2f}%")
        
        return {
            'trades': self.trades,
            'portfolio_value': self.portfolio_value,
            'dates': self.dates,
            'metrics': metrics,
            'signals': self.signals_history,
            'final_value': self.portfolio_value[-1] if self.portfolio_value else backtest_config.INITIAL_CAPITAL
        }
    
    def calculate_metrics(self) -> Dict[str, float]:
        """
        Calcul des métriques de performance complètes
        
        Returns:
            Dictionnaire des métriques
        """
        if not self.portfolio_value or not self.trades:
            return {}
        
        # Conversion en arrays pour calculs
        portfolio_values = np.array(self.portfolio_value)
        dates = pd.to_datetime(self.dates)
        
        # Rendements
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        # Métriques de base
        initial_value = backtest_config.INITIAL_CAPITAL
        final_value = portfolio_values[-1]
        total_return = (final_value - initial_value) / initial_value * 100
        
        # Durée du backtest
        duration_days = (dates.iloc[-1] - dates.iloc[0]).days
        duration_years = duration_days / 365.25
        
        # Rendement annualisé
        if duration_years > 0:
            annualized_return = (final_value / initial_value) ** (1 / duration_years) - 1
        else:
            annualized_return = 0
        
        # Volatilité
        volatility = np.std(returns) * np.sqrt(252)  # Annualisée
        
        # Drawdown
        cumulative_returns = portfolio_values / initial_value
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdowns) * 100
        
        # Trades statistics
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl < 0]
        
        win_rate = len(winning_trades) / len(self.trades) * 100 if self.trades else 0
        
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        # Sharpe Ratio
        if backtest_config.CALCULATE_SHARPE and volatility > 0:
            sharpe_ratio = (annualized_return - backtest_config.RISK_FREE_RATE) / volatility
        else:
            sharpe_ratio = 0
        
        # Sortino Ratio
        if backtest_config.CALCULATE_SORTINO:
            negative_returns = returns[returns < 0]
            downside_volatility = np.std(negative_returns) * np.sqrt(252) if len(negative_returns) > 0 else 0
            sortino_ratio = (annualized_return - backtest_config.RISK_FREE_RATE) / downside_volatility if downside_volatility > 0 else 0
        else:
            sortino_ratio = 0
        
        # Calmar Ratio
        if backtest_config.CALCULATE_CALMAR and max_drawdown < 0:
            calmar_ratio = annualized_return / abs(max_drawdown / 100)
        else:
            calmar_ratio = 0
        
        # Métriques compilées
        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return * 100,
            'volatility': volatility * 100,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': len(self.trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'initial_capital': initial_value,
            'final_capital': final_value,
            'duration_days': duration_days,
            'duration_years': duration_years
        }
        
        self.metrics = metrics
        return metrics
    
    def plot_results(self, save_path: str = None, show: bool = True) -> None:
        """
        Visualisation des résultats du backtest
        
        Args:
            save_path: Chemin de sauvegarde (optionnel)
            show: Afficher les graphiques
        """
        if not self.portfolio_value:
            logger.warning("Aucune donnée à visualiser")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Résultats du Backtest - {config.SYMBOL}', fontsize=16)
        
        # 1. Évolution du portefeuille
        ax1 = axes[0, 0]
        dates_series = pd.to_datetime(self.dates)
        ax1.plot(dates_series, self.portfolio_value, linewidth=1.5, color='blue')
        ax1.axhline(y=backtest_config.INITIAL_CAPITAL, color='red', linestyle='--', alpha=0.7)
        ax1.set_title('Évolution de la Valeur du Portefeuille')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Valeur ($)')
        ax1.grid(True, alpha=0.3)
        
        # 2. Drawdown
        ax2 = axes[0, 1]
        portfolio_array = np.array(self.portfolio_value)
        cumulative_returns = portfolio_array / backtest_config.INITIAL_CAPITAL
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max * 100
        
        ax2.fill_between(dates_series, drawdowns, 0, alpha=0.3, color='red')
        ax2.plot(dates_series, drawdowns, color='red', linewidth=1)
        ax2.set_title('Drawdown (%)')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True, alpha=0.3)
        
        # 3. Distribution des P&L
        ax3 = axes[1, 0]
        if self.trades:
            pnl_values = [t.pnl for t in self.trades if t.pnl is not None]
            if pnl_values:
                ax3.hist(pnl_values, bins=20, alpha=0.7, color='green', edgecolor='black')
                ax3.axvline(x=0, color='red', linestyle='--')
                ax3.set_title('Distribution des P&L par Trade')
                ax3.set_xlabel('P&L ($)')
                ax3.set_ylabel('Fréquence')
                ax3.grid(True, alpha=0.3)
        
        # 4. Métriques textuelles
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        if self.metrics:
            metrics_text = f"""
Métriques de Performance:

Rendement Total: {self.metrics.get('total_return', 0):.2f}%
Rendement Annualisé: {self.metrics.get('annualized_return', 0):.2f}%
Volatilité: {self.metrics.get('volatility', 0):.2f}%
Max Drawdown: {self.metrics.get('max_drawdown', 0):.2f}%

Sharpe Ratio: {self.metrics.get('sharpe_ratio', 0):.3f}
Sortino Ratio: {self.metrics.get('sortino_ratio', 0):.3f}
Calmar Ratio: {self.metrics.get('calmar_ratio', 0):.3f}

Trades Total: {self.metrics.get('total_trades', 0)}
Taux de Réussite: {self.metrics.get('win_rate', 0):.1f}%
Profit Factor: {self.metrics.get('profit_factor', 0):.2f}

Gain Moyen: ${self.metrics.get('avg_win', 0):.2f}
Perte Moyenne: ${self.metrics.get('avg_loss', 0):.2f}
            """
            ax4.text(0, 1, metrics_text, transform=ax4.transAxes, fontsize=10, 
                    verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Graphiques sauvegardés: {save_path}")
        
        if show:
            plt.show()
    
    def generate_report(self) -> str:
        """
        Génération d'un rapport textuel complet
        
        Returns:
            Rapport détaillé
        """
        if not self.metrics:
            return "Aucune métrique disponible"
        
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("RAPPORT DE BACKTEST DÉTAILLÉ")
        report_lines.append("=" * 60)
        report_lines.append(f"Symbole: {config.SYMBOL}")
        report_lines.append(f"Période: {self.dates[0]} à {self.dates[-1]}")
        report_lines.append(f"Durée: {self.metrics['duration_days']} jours ({self.metrics['duration_years']:.2f} années)")
        report_lines.append("")
        
        report_lines.append("PERFORMANCE GLOBALE")
        report_lines.append("-" * 30)
        report_lines.append(f"Capital Initial: ${self.metrics['initial_capital']:,.2f}")
        report_lines.append(f"Capital Final: ${self.metrics['final_capital']:,.2f}")
        report_lines.append(f"Rendement Total: {self.metrics['total_return']:+.2f}%")
        report_lines.append(f"Rendement Annualisé: {self.metrics['annualized_return']:+.2f}%")
        report_lines.append("")
        
        report_lines.append("MÉTRIQUES DE RISQUE")
        report_lines.append("-" * 30)
        report_lines.append(f"Volatilité Annualisée: {self.metrics['volatility']:.2f}%")
        report_lines.append(f"Drawdown Maximum: {self.metrics['max_drawdown']:.2f}%")
        report_lines.append(f"Sharpe Ratio: {self.metrics['sharpe_ratio']:.3f}")
        report_lines.append(f"Sortino Ratio: {self.metrics['sortino_ratio']:.3f}")
        report_lines.append(f"Calmar Ratio: {self.metrics['calmar_ratio']:.3f}")
        report_lines.append("")
        
        report_lines.append("STATISTIQUES DE TRADING")
        report_lines.append("-" * 30)
        report_lines.append(f"Nombre Total de Trades: {self.metrics['total_trades']}")
        report_lines.append(f"Trades Gagnants: {self.metrics['winning_trades']}")
        report_lines.append(f"Trades Perdants: {self.metrics['losing_trades']}")
        report_lines.append(f"Taux de Réussite: {self.metrics['win_rate']:.1f}%")
        report_lines.append(f"Profit Factor: {self.metrics['profit_factor']:.2f}")
        report_lines.append(f"Gain Moyen: ${self.metrics['avg_win']:.2f}")
        report_lines.append(f"Perte Moyenne: ${self.metrics['avg_loss']:.2f}")
        report_lines.append("")
        
        # Analyse des trades récents
        if len(self.trades) > 0:
            report_lines.append("DERNIERS TRADES")
            report_lines.append("-" * 30)
            recent_trades = self.trades[-5:]  # 5 derniers trades
            for i, trade in enumerate(recent_trades, 1):
                signal_str = "BUY" if trade.signal == TradingSignals.BUY else "SELL"
                report_lines.append(f"{i}. {trade.entry_time.strftime('%Y-%m-%d')} - "
                                  f"{signal_str} @ ${trade.entry_price:.2f} → "
                                  f"${trade.exit_price:.2f} = "
                                  f"${trade.pnl:+.2f} ({trade.pnl_pct:+.1f}%)")
        
        return "\n".join(report_lines)