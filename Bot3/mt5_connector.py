# mt5_connector.py
"""
Connecteur MetaTrader 5 amélioré avec gestion d'erreurs et sécurité
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
import time

from config import config, TradingSignals

logger = logging.getLogger('trading_system.mt5_connector')

@dataclass
class Position:
    """Représentation d'une position MT5"""
    ticket: int
    symbol: str
    type: int  # 0=BUY, 1=SELL
    volume: float
    price_open: float
    price_current: float
    profit: float
    swap: float
    commission: float
    time_open: datetime
    comment: str = ""

@dataclass
class OrderRequest:
    """Requête d'ordre MT5"""
    action: int
    symbol: str
    volume: float
    type: int
    price: float = 0.0
    sl: float = 0.0
    tp: float = 0.0
    deviation: int = 20
    magic: int = 12345
    comment: str = "LSTM_Trading"
    type_time: int = mt5.ORDER_TIME_GTC
    type_filling: int = mt5.ORDER_FILLING_IOC

class MT5Connector:
    """
    Connecteur MetaTrader 5 avec gestion d'erreurs robuste
    """
    
    def __init__(self):
        self.is_connected = False
        self.account_info = None
        self.symbol_info = {}
        self.last_error = None
        
    def connect(self, login: int = None, password: str = None, 
                server: str = None) -> bool:
        """
        Connexion à MT5 avec gestion d'erreurs
        
        Args:
            login: Numéro de compte (optionnel)
            password: Mot de passe (optionnel)
            server: Serveur (optionnel)
            
        Returns:
            True si connexion réussie
        """
        try:
            # Initialisation de MT5
            if not mt5.initialize():
                error = mt5.last_error()
                self.last_error = f"Échec d'initialisation MT5: {error}"
                logger.error(self.last_error)
                return False
            
            # Connexion avec identifiants si fournis
            if login and password and server:
                if not mt5.login(login, password, server):
                    error = mt5.last_error()
                    self.last_error = f"Échec de connexion: {error}"
                    logger.error(self.last_error)
                    mt5.shutdown()
                    return False
            
            # Vérification de la connexion
            self.account_info = mt5.account_info()
            if self.account_info is None:
                self.last_error = "Impossible de récupérer les informations du compte"
                logger.error(self.last_error)
                mt5.shutdown()
                return False
            
            self.is_connected = True
            logger.info(f"Connexion MT5 réussie - Compte: {self.account_info.login}, "
                       f"Solde: {self.account_info.balance}")
            
            return True
            
        except Exception as e:
            self.last_error = f"Erreur lors de la connexion: {e}"
            logger.error(self.last_error)
            return False
    
    def disconnect(self) -> None:
        """Déconnexion propre de MT5"""
        try:
            if self.is_connected:
                mt5.shutdown()
                self.is_connected = False
                logger.info("Déconnexion MT5 réussie")
        except Exception as e:
            logger.error(f"Erreur lors de la déconnexion: {e}")
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Récupération des informations du symbole
        
        Args:
            symbol: Symbole de trading
            
        Returns:
            Informations du symbole ou None
        """
        if not self.is_connected:
            logger.error("Pas de connexion MT5")
            return None
        
        try:
            # Cache des informations de symbole
            if symbol in self.symbol_info:
                return self.symbol_info[symbol]
            
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                logger.error(f"Symbole {symbol} non trouvé")
                return None
            
            # Conversion en dictionnaire
            info_dict = {
                'name': symbol_info.name,
                'digits': symbol_info.digits,
                'point': symbol_info.point,
                'min_lot': symbol_info.volume_min,
                'max_lot': symbol_info.volume_max,
                'lot_step': symbol_info.volume_step,
                'contract_size': symbol_info.trade_contract_size,
                'margin_required': symbol_info.margin_initial,
                'tick_size': symbol_info.trade_tick_size,
                'tick_value': symbol_info.trade_tick_value,
                'spread': symbol_info.spread
            }
            
            self.symbol_info[symbol] = info_dict
            return info_dict
            
        except Exception as e:
            logger.error(f"Erreur récupération info symbole {symbol}: {e}")
            return None
    
    def get_current_price(self, symbol: str) -> Optional[Tuple[float, float]]:
        """
        Récupération du prix actuel (bid, ask)
        
        Args:
            symbol: Symbole de trading
            
        Returns:
            Tuple (bid, ask) ou None
        """
        if not self.is_connected:
            return None
        
        try:
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return None
            
            return tick.bid, tick.ask
            
        except Exception as e:
            logger.error(f"Erreur récupération prix {symbol}: {e}")
            return None
    
    def get_historical_data(self, symbol: str, timeframe: int, 
                           count: int) -> Optional[pd.DataFrame]:
        """
        Récupération des données historiques
        
        Args:
            symbol: Symbole de trading
            timeframe: Timeframe MT5 (ex: mt5.TIMEFRAME_D1)
            count: Nombre de barres
            
        Returns:
            DataFrame avec données OHLCV ou None
        """
        if not self.is_connected:
            return None
        
        try:
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
            if rates is None or len(rates) == 0:
                logger.error(f"Aucune donnée historique pour {symbol}")
                return None
            
            # Conversion en DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            # Renommage des colonnes
            df.columns = ['open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']
            df.rename(columns={'tick_volume': 'volume'}, inplace=True)
            
            logger.info(f"Données historiques récupérées: {len(df)} barres pour {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Erreur récupération données historiques {symbol}: {e}")
            return None
    
    def calculate_lot_size(self, symbol: str, risk_amount: float, 
                          stop_loss_pips: float) -> float:
        """
        Calcul de la taille de lot basée sur le risque
        
        Args:
            symbol: Symbole de trading
            risk_amount: Montant à risquer en devise du compte
            stop_loss_pips: Distance du stop loss en pips
            
        Returns:
            Taille de lot calculée
        """
        symbol_info = self.get_symbol_info(symbol)
        if not symbol_info:
            return config.LOT_SIZE
        
        try:
            # Valeur d'un pip
            pip_value = symbol_info['tick_value'] * (symbol_info['tick_size'] / symbol_info['point'])
            
            # Calcul de la taille de lot
            lot_size = risk_amount / (stop_loss_pips * pip_value)
            
            # Arrondi selon les règles du symbole
            lot_step = symbol_info['lot_step']
            lot_size = round(lot_size / lot_step) * lot_step
            
            # Vérification des limites
            lot_size = max(symbol_info['min_lot'], 
                          min(lot_size, symbol_info['max_lot']))
            
            return lot_size
            
        except Exception as e:
            logger.error(f"Erreur calcul taille lot: {e}")
            return config.LOT_SIZE
    
    def send_order(self, request: OrderRequest) -> Optional[Dict[str, Any]]:
        """
        Envoi d'un ordre avec gestion d'erreurs
        
        Args:
            request: Requête d'ordre
            
        Returns:
            Résultat de l'ordre ou None
        """
        if not self.is_connected:
            logger.error("Pas de connexion MT5 pour envoyer l'ordre")
            return None
        
        try:
            # Préparation de la requête
            order_request = {
                "action": request.action,
                "symbol": request.symbol,
                "volume": request.volume,
                "type": request.type,
                "price": request.price,
                "sl": request.sl,
                "tp": request.tp,
                "deviation": request.deviation,
                "magic": request.magic,
                "comment": request.comment,
                "type_time": request.type_time,
                "type_filling": request.type_filling,
            }
            
            # Envoi de l'ordre
            result = mt5.order_send(order_request)
            
            if result is None:
                logger.error("Erreur envoi ordre: résultat None")
                return None
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Ordre rejeté: {result.retcode} - {result.comment}")
                return {
                    "success": False,
                    "retcode": result.retcode,
                    "comment": result.comment
                }
            
            logger.info(f"Ordre exécuté: ticket={result.order}, "
                       f"volume={result.volume}, price={result.price}")
            
            return {
                "success": True,
                "ticket": result.order,
                "volume": result.volume,
                "price": result.price,
                "retcode": result.retcode,
                "comment": result.comment
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de l'envoi d'ordre: {e}")
            return None
    
    def open_position(self, symbol: str, signal: int, volume: float = None,
                     sl_pips: float = None, tp_pips: float = None) -> Optional[Dict[str, Any]]:
        """
        Ouverture d'une position
        
        Args:
            symbol: Symbole de trading
            signal: Signal (TradingSignals.BUY ou TradingSignals.SELL)
            volume: Volume (optionnel)
            sl_pips: Stop loss en pips (optionnel)
            tp_pips: Take profit en pips (optionnel)
            
        Returns:
            Résultat de l'ouverture
        """
        if signal == TradingSignals.HOLD:
            return None
        
        # Volume par défaut
        if volume is None:
            volume = config.LOT_SIZE
        
        # Prix actuel
        prices = self.get_current_price(symbol)
        if not prices:
            logger.error(f"Impossible de récupérer le prix pour {symbol}")
            return None
        
        bid, ask = prices
        symbol_info = self.get_symbol_info(symbol)
        
        if not symbol_info:
            logger.error(f"Informations symbole non disponibles pour {symbol}")
            return None
        
        try:
            # Détermination du type d'ordre et du prix
            if signal == TradingSignals.BUY:
                order_type = mt5.ORDER_TYPE_BUY
                price = ask
                sl_price = 0.0
                tp_price = 0.0
                
                if sl_pips:
                    sl_price = price - (sl_pips * symbol_info['point'] * 10)
                if tp_pips:
                    tp_price = price + (tp_pips * symbol_info['point'] * 10)
                    
            else:  # SELL
                order_type = mt5.ORDER_TYPE_SELL
                price = bid
                sl_price = 0.0
                tp_price = 0.0
                
                if sl_pips:
                    sl_price = price + (sl_pips * symbol_info['point'] * 10)
                if tp_pips:
                    tp_price = price - (tp_pips * symbol_info['point'] * 10)
            
            # Création de la requête d'ordre
            request = OrderRequest(
                action=mt5.TRADE_ACTION_DEAL,
                symbol=symbol,
                volume=volume,
                type=order_type,
                price=price,
                sl=sl_price,
                tp=tp_price,
                comment=f"LSTM_{['SELL', 'HOLD', 'BUY'][signal + 1]}"
            )
            
            # Envoi de l'ordre
            result = self.send_order(request)
            
            if result and result.get("success"):
                logger.info(f"Position ouverte: {symbol} "
                          f"{'BUY' if signal == TradingSignals.BUY else 'SELL'} "
                          f"{volume} lots @ {price}")
            
            return result
            
        except Exception as e:
            logger.error(f"Erreur ouverture position {symbol}: {e}")
            return None
    
    def close_position(self, position: Position) -> Optional[Dict[str, Any]]:
        """
        Fermeture d'une position
        
        Args:
            position: Position à fermer
            
        Returns:
            Résultat de la fermeture
        """
        try:
            # Prix actuel
            prices = self.get_current_price(position.symbol)
            if not prices:
                return None
            
            bid, ask = prices
            
            # Détermination du type d'ordre de fermeture
            if position.type == 0:  # Position BUY
                close_type = mt5.ORDER_TYPE_SELL
                close_price = bid
            else:  # Position SELL
                close_type = mt5.ORDER_TYPE_BUY
                close_price = ask
            
            # Requête de fermeture
            request = OrderRequest(
                action=mt5.TRADE_ACTION_DEAL,
                symbol=position.symbol,
                volume=position.volume,
                type=close_type,
                price=close_price,
                comment=f"Close_{position.ticket}"
            )
            
            result = self.send_order(request)
            
            if result and result.get("success"):
                logger.info(f"Position fermée: ticket={position.ticket}, "
                          f"profit={position.profit}")
            
            return result
            
        except Exception as e:
            logger.error(f"Erreur fermeture position {position.ticket}: {e}")
            return None
    
    def get_open_positions(self, symbol: str = None) -> List[Position]:
        """
        Récupération des positions ouvertes
        
        Args:
            symbol: Filtrer par symbole (optionnel)
            
        Returns:
            Liste des positions ouvertes
        """
        if not self.is_connected:
            return []
        
        try:
            if symbol:
                positions = mt5.positions_get(symbol=symbol)
            else:
                positions = mt5.positions_get()
            
            if positions is None:
                return []
            
            position_list = []
            for pos in positions:
                position = Position(
                    ticket=pos.ticket,
                    symbol=pos.symbol,
                    type=pos.type,
                    volume=pos.volume,
                    price_open=pos.price_open,
                    price_current=pos.price_current,
                    profit=pos.profit,
                    swap=pos.swap,
                    commission=pos.commission,
                    time_open=datetime.fromtimestamp(pos.time),
                    comment=pos.comment
                )
                position_list.append(position)
            
            return position_list
            
        except Exception as e:
            logger.error(f"Erreur récupération positions: {e}")
            return []
    
    def close_all_positions(self, symbol: str = None) -> Dict[str, Any]:
        """
        Fermeture de toutes les positions
        
        Args:
            symbol: Symbole spécifique (optionnel)
            
        Returns:
            Résumé des fermetures
        """
        positions = self.get_open_positions(symbol)
        
        if not positions:
            return {"closed": 0, "errors": 0, "total_profit": 0.0}
        
        closed_count = 0
        error_count = 0
        total_profit = 0.0
        
        for position in positions:
            result = self.close_position(position)
            if result and result.get("success"):
                closed_count += 1
                total_profit += position.profit
            else:
                error_count += 1
        
        logger.info(f"Fermeture positions: {closed_count} fermées, "
                   f"{error_count} erreurs, profit total: {total_profit}")
        
        return {
            "closed": closed_count,
            "errors": error_count,
            "total_profit": total_profit
        }
    
    def get_account_info(self) -> Optional[Dict[str, Any]]:
        """
        Récupération des informations de compte
        
        Returns:
            Informations du compte
        """
        if not self.is_connected:
            return None
        
        try:
            account = mt5.account_info()
            if account is None:
                return None
            
            return {
                "login": account.login,
                "balance": account.balance,
                "equity": account.equity,
                "profit": account.profit,
                "margin": account.margin,
                "margin_free": account.margin_free,
                "margin_level": account.margin_level,
                "currency": account.currency,
                "leverage": account.leverage,
                "server": account.server,
                "company": account.company
            }
            
        except Exception as e:
            logger.error(f"Erreur récupération info compte: {e}")
            return None
    
    def check_margin_requirements(self, symbol: str, volume: float, 
                                 order_type: int) -> bool:
        """
        Vérification des exigences de marge
        
        Args:
            symbol: Symbole de trading
            volume: Volume de l'ordre
            order_type: Type d'ordre
            
        Returns:
            True si marge suffisante
        """
        try:
            # Calcul de la marge requise
            margin_required = mt5.order_calc_margin(order_type, symbol, volume, 0.0)
            
            if margin_required is None:
                logger.warning("Impossible de calculer la marge requise")
                return False
            
            # Informations du compte
            account = self.get_account_info()
            if not account:
                return False
            
            # Vérification
            margin_available = account["margin_free"]
            has_sufficient_margin = margin_required <= margin_available
            
            if not has_sufficient_margin:
                logger.warning(f"Marge insuffisante: requis={margin_required}, "
                             f"disponible={margin_available}")
            
            return has_sufficient_margin
            
        except Exception as e:
            logger.error(f"Erreur vérification marge: {e}")
            return False
    
    def is_market_open(self, symbol: str) -> bool:
        """
        Vérification si le marché est ouvert
        
        Args:
            symbol: Symbole à vérifier
            
        Returns:
            True si marché ouvert
        """
        try:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return False
            
            # Vérification du statut de trading
            return symbol_info.trade_mode == mt5.SYMBOL_TRADE_MODE_FULL
            
        except Exception as e:
            logger.error(f"Erreur vérification marché {symbol}: {e}")
            return False
    
    def execute_trading_signal(self, symbol: str, signal: int, 
                              volume: float = None, 
                              risk_management: bool = True) -> Dict[str, Any]:
        """
        Exécution complète d'un signal de trading avec gestion des risques
        
        Args:
            symbol: Symbole de trading
            signal: Signal de trading
            volume: Volume (optionnel)
            risk_management: Appliquer la gestion des risques
            
        Returns:
            Résultat de l'exécution
        """
        result = {
            "success": False,
            "action_taken": "none",
            "message": "",
            "position_info": None
        }
        
        try:
            # Vérifications préliminaires
            if not self.is_connected:
                result["message"] = "Pas de connexion MT5"
                return result
            
            if not self.is_market_open(symbol):
                result["message"] = f"Marché fermé pour {symbol}"
                return result
            
            # Gestion des positions existantes
            existing_positions = self.get_open_positions(symbol)
            
            if signal == TradingSignals.HOLD:
                result["action_taken"] = "hold"
                result["message"] = "Signal HOLD - aucune action"
                result["success"] = True
                return result
            
            # Fermeture des positions opposées si nécessaire
            for pos in existing_positions:
                should_close = False
                
                if pos.type == 0 and signal == TradingSignals.SELL:  # BUY position + SELL signal
                    should_close = True
                elif pos.type == 1 and signal == TradingSignals.BUY:  # SELL position + BUY signal
                    should_close = True
                
                if should_close:
                    close_result = self.close_position(pos)
                    if close_result and close_result.get("success"):
                        logger.info(f"Position opposée fermée: {pos.ticket}")
            
            # Volume par défaut
            if volume is None:
                volume = config.LOT_SIZE
            
            # Gestion des risques
            if risk_management:
                # Vérification de la marge
                order_type = mt5.ORDER_TYPE_BUY if signal == TradingSignals.BUY else mt5.ORDER_TYPE_SELL
                if not self.check_margin_requirements(symbol, volume, order_type):
                    result["message"] = "Marge insuffisante"
                    return result
                
                # Limitation du nombre de positions
                if len(existing_positions) >= config.MAX_POSITIONS:
                    result["message"] = f"Limite de positions atteinte ({config.MAX_POSITIONS})"
                    return result
            
            # Ouverture de la nouvelle position
            position_result = self.open_position(
                symbol=symbol,
                signal=signal,
                volume=volume,
                sl_pips=50,  # Stop loss à 50 pips
                tp_pips=100  # Take profit à 100 pips
            )
            
            if position_result and position_result.get("success"):
                result["success"] = True
                result["action_taken"] = "buy" if signal == TradingSignals.BUY else "sell"
                result["message"] = f"Position ouverte: {result['action_taken']} {volume} lots"
                result["position_info"] = position_result
            else:
                result["message"] = "Échec de l'ouverture de position"
            
            return result
            
        except Exception as e:
            result["message"] = f"Erreur lors de l'exécution: {e}"
            logger.error(result["message"])
            return result


class MT5DataProvider:
    """
    Fournisseur de données MT5 pour l'entraînement et les prédictions
    """
    
    def __init__(self):
        self.connector = MT5Connector()
    
    def connect(self, login: int = None, password: str = None, 
                server: str = None) -> bool:
        """Connexion à MT5"""
        return self.connector.connect(login, password, server)
    
    def get_training_data(self, symbol: str, days: int = None) -> Optional[pd.DataFrame]:
        """
        Récupération des données pour l'entraînement
        
        Args:
            symbol: Symbole de trading
            days: Nombre de jours (défaut: config.TRAINING_DAYS)
            
        Returns:
            DataFrame avec données historiques
        """
        if days is None:
            days = config.TRAINING_DAYS
        
        # Ajout de marge pour les indicateurs techniques
        total_bars = days + 100
        
        return self.connector.get_historical_data(
            symbol=symbol,
            timeframe=mt5.TIMEFRAME_D1,
            count=total_bars
        )
    
    def get_current_data(self, symbol: str, bars: int = None) -> Optional[pd.DataFrame]:
        """
        Récupération des données récentes pour les prédictions
        
        Args:
            symbol: Symbole de trading
            bars: Nombre de barres (défaut: config.EXECUTION_DAYS)
            
        Returns:
            DataFrame avec données récentes
        """
        if bars is None:
            bars = config.EXECUTION_DAYS
        
        return self.connector.get_historical_data(
            symbol=symbol,
            timeframe=mt5.TIMEFRAME_D1,
            count=bars
        )
    
    def disconnect(self) -> None:
        """Déconnexion"""
        self.connector.disconnect()


# Exemple d'utilisation
def example_mt5_usage():
    """
    Exemple d'utilisation du connecteur MT5
    """
    # Initialisation
    mt5_connector = MT5Connector()
    
    try:
        # Connexion (remplacer par vos vrais identifiants)
        if mt5_connector.connect():
            print("Connexion MT5 réussie")
            
            # Informations du compte
            account_info = mt5_connector.get_account_info()
            if account_info:
                print(f"Solde: {account_info['balance']} {account_info['currency']}")
            
            # Test d'ouverture de position (DEMO uniquement!)
            # result = mt5_connector.execute_trading_signal(
            #     symbol="XAUUSD",
            #     signal=TradingSignals.BUY,
            #     volume=0.01
            # )
            # print(f"Résultat trading: {result}")
            
            # Positions ouvertes
            positions = mt5_connector.get_open_positions("XAUUSD")
            print(f"Positions ouvertes: {len(positions)}")
            
        else:
            print(f"Échec de connexion: {mt5_connector.last_error}")
    
    finally:
        mt5_connector.disconnect()


if __name__ == "__main__":
    example_mt5_usage()