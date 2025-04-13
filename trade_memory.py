# trade_memory.py

import os
import sqlite3
import json
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta, timezone

logger = logging.getLogger("TradingBot")

class TradeMemory:
    """Advanced memory system to store, analyze, and learn from past trades."""
    
    def __init__(self, config):
        self.config = config
        self.db_path = config.get('DATABASE_PATH', 'data/trading_memory.db')
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Connect to SQLite database
        self.conn = self._init_database()
        
        # Internal cache for performance metrics
        self._metrics_cache = {}
        self._last_cache_update = datetime.now(timezone.utc) - timedelta(hours=1)  # Force initial update
        self._cache_ttl = timedelta(minutes=15)
        
    def _init_database(self):
        """Initialize the database with necessary tables."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create trades table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                epic TEXT NOT NULL,
                direction TEXT NOT NULL,
                size REAL NOT NULL,
                entry_price REAL,
                exit_price REAL,
                stop_level REAL,
                limit_level REAL,
                pnl REAL,
                outcome TEXT,
                market_regime TEXT,
                trade_duration INTEGER,
                confidence TEXT,
                deal_id TEXT,
                reason TEXT,
                technical_data TEXT,
                trade_context TEXT
            )
            ''')
            
            # Create market_data table to store snapshots with technical indicators
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                epic TEXT NOT NULL,
                price_data TEXT NOT NULL,
                technical_data TEXT,
                market_regime TEXT
            )
            ''')
            
            # Create performance_metrics table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                timeframe TEXT NOT NULL,
                epic TEXT
            )
            ''')
            
            # Create a table for trade patterns
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS trade_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_name TEXT NOT NULL,
                epic TEXT,
                direction TEXT,
                success_rate REAL,
                avg_pnl REAL,
                count INTEGER,
                pattern_data TEXT,
                last_updated TEXT
            )
            ''')
            
            conn.commit()
            logger.info(f"Trade memory database initialized at {self.db_path}")
            return conn
            
        except sqlite3.Error as e:
            logger.error(f"Database initialization error: {e}")
            raise

    def store_trade(self, trade_data, technical_data=None, market_regime=None):
        """Store a trade in the database with additional context."""
        try:
            cursor = self.conn.cursor()
            
            # Prepare trade data
            trade_record = {
                'timestamp': trade_data.get('timestamp', datetime.now(timezone.utc).isoformat()),
                'epic': trade_data.get('epic', 'unknown'),
                'direction': trade_data.get('direction', 'unknown'),
                'size': float(trade_data.get('size', 0)),
                'entry_price': float(trade_data.get('entry_price', 0)) if trade_data.get('entry_price') else None,
                'exit_price': float(trade_data.get('exit_price', 0)) if trade_data.get('exit_price') else None,
                'stop_level': float(trade_data.get('stop_level', 0)) if trade_data.get('stop_level') else None,
                'limit_level': float(trade_data.get('limit_level', 0)) if trade_data.get('limit_level') else None,
                'pnl': float(trade_data.get('pnl', 0)) if trade_data.get('pnl') else None,
                'outcome': trade_data.get('outcome', 'UNKNOWN'),
                'market_regime': market_regime,
                'trade_duration': trade_data.get('trade_duration'),
                'confidence': trade_data.get('confidence', 'medium'),
                'deal_id': trade_data.get('deal_id'),
                'reason': trade_data.get('reason', ''),
                'technical_data': json.dumps(technical_data) if technical_data else None,
                'trade_context': json.dumps(trade_data.get('context', {}))
            }
            
            # Insert trade
            cols = ', '.join(trade_record.keys())
            placeholders = ', '.join(['?'] * len(trade_record))
            
            query = f"INSERT INTO trades ({cols}) VALUES ({placeholders})"
            cursor.execute(query, list(trade_record.values()))
            
            self.conn.commit()
            logger.info(f"Trade stored: {trade_record['epic']} {trade_record['direction']} -> {trade_record['outcome']}")
            
            # Reset cache to trigger refresh
            self._last_cache_update = datetime.now(timezone.utc) - timedelta(hours=1)
            
            # Update performance metrics after storing a new trade
            self.update_performance_metrics()
            
            return cursor.lastrowid
            
        except Exception as e:
            logger.error(f"Error storing trade: {e}")
            return None

    def store_market_data(self, epic, price_data, technical_data=None, market_regime=None):
        """Store market data snapshot with technical indicators."""
        try:
            cursor = self.conn.cursor()
            
            record = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'epic': epic,
                'price_data': json.dumps(price_data),
                'technical_data': json.dumps(technical_data) if technical_data else None,
                'market_regime': market_regime
            }
            
            cols = ', '.join(record.keys())
            placeholders = ', '.join(['?'] * len(record))
            
            query = f"INSERT INTO market_data ({cols}) VALUES ({placeholders})"
            cursor.execute(query, list(record.values()))
            
            self.conn.commit()
            logger.debug(f"Market data stored for {epic}")
            
            return cursor.lastrowid
            
        except Exception as e:
            logger.error(f"Error storing market data: {e}")
            return None

    def get_recent_trades(self, limit=10, epic=None, outcome=None):
        """Get recent trades with optional filtering."""
        try:
            cursor = self.conn.cursor()
            
            query = "SELECT * FROM trades"
            params = []
            
            # Add filters if specified
            conditions = []
            if epic:
                conditions.append("epic = ?")
                params.append(epic)
            if outcome:
                conditions.append("outcome = ?")
                params.append(outcome)
                
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
                
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            
            # Get column names
            columns = [description[0] for description in cursor.description]
            
            # Fetch data and create a list of dictionaries
            trades = []
            for row in cursor.fetchall():
                trade_dict = dict(zip(columns, row))
                # Parse JSON fields
                if trade_dict.get('technical_data'):
                    try:
                        trade_dict['technical_data'] = json.loads(trade_dict['technical_data'])
                    except:
                        trade_dict['technical_data'] = {}
                if trade_dict.get('trade_context'):
                    try:
                        trade_dict['trade_context'] = json.loads(trade_dict['trade_context'])
                    except:
                        trade_dict['trade_context'] = {}
                trades.append(trade_dict)
                
            return trades
            
        except Exception as e:
            logger.error(f"Error getting recent trades: {e}")
            return []

    def get_trade_history_summary(self, days=30, epic=None):
        """Get a summary of trade history for a specified timeframe."""
        try:
            cursor = self.conn.cursor()
            
            # Calculate the date range
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=days)
            
            query = """
            SELECT 
                epic,
                direction,
                COUNT(*) as total_trades,
                SUM(CASE WHEN outcome = 'PROFIT' THEN 1 ELSE 0 END) as winning_trades,
                SUM(CASE WHEN outcome = 'LOSS' THEN 1 ELSE 0 END) as losing_trades,
                SUM(pnl) as net_pnl,
                AVG(CASE WHEN outcome = 'PROFIT' THEN pnl ELSE NULL END) as avg_win,
                AVG(CASE WHEN outcome = 'LOSS' THEN pnl ELSE NULL END) as avg_loss,
                AVG(trade_duration) as avg_duration
            FROM trades
            WHERE timestamp BETWEEN ? AND ?
            """
            
            params = [start_date.isoformat(), end_date.isoformat()]
            
            if epic:
                query += " AND epic = ?"
                params.append(epic)
                
            query += " GROUP BY epic, direction"
            
            cursor.execute(query, params)
            
            # Get column names
            columns = [description[0] for description in cursor.description]
            
            # Fetch data and create a list of dictionaries
            results = []
            for row in cursor.fetchall():
                result_dict = dict(zip(columns, row))
                results.append(result_dict)
                
            return results
            
        except Exception as e:
            logger.error(f"Error getting trade history summary: {e}")
            return []

    def calculate_win_rate(self, days=30, epic=None):
        """Calculate the win rate for a specific period and instrument."""
        try:
            cursor = self.conn.cursor()
            
            # Calculate the date range
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=days)
            
            query = """
            SELECT 
                COUNT(*) as total_trades,
                SUM(CASE WHEN outcome = 'PROFIT' THEN 1 ELSE 0 END) as winning_trades
            FROM trades
            WHERE timestamp BETWEEN ? AND ?
            AND outcome IN ('PROFIT', 'LOSS')
            """
            
            params = [start_date.isoformat(), end_date.isoformat()]
            
            if epic:
                query += " AND epic = ?"
                params.append(epic)
            
            cursor.execute(query, params)
            result = cursor.fetchone()
            
            total_trades, winning_trades = result
            
            # Handle None values
            if total_trades is None:
                total_trades = 0
            if winning_trades is None:
                winning_trades = 0
            
            if total_trades > 0:
                win_rate = winning_trades / total_trades
                return win_rate
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error calculating win rate: {e}")
            return 0.0

    def calculate_risk_reward_ratio(self, days=30, epic=None):
        """Calculate the average risk/reward ratio for completed trades."""
        try:
            cursor = self.conn.cursor()
            
            # Calculate the date range
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=days)
            
            query = """
            SELECT 
                direction,
                entry_price,
                exit_price,
                stop_level
            FROM trades
            WHERE timestamp BETWEEN ? AND ?
            AND outcome IN ('PROFIT', 'LOSS')
            AND entry_price IS NOT NULL
            AND exit_price IS NOT NULL
            AND stop_level IS NOT NULL
            """
            
            params = [start_date.isoformat(), end_date.isoformat()]
            
            if epic:
                query += " AND epic = ?"
                params.append(epic)
            
            cursor.execute(query, params)
            
            ratios = []
            for row in cursor.fetchall():
                direction, entry_price, exit_price, stop_level = row
                
                # Calculate risk and reward
                if direction == 'BUY':
                    risk = entry_price - stop_level
                    reward = exit_price - entry_price
                else:  # SELL
                    risk = stop_level - entry_price
                    reward = entry_price - exit_price
                
                # Calculate R:R ratio
                if risk > 0:
                    ratio = abs(reward / risk)
                    ratios.append(ratio)
            
            if ratios:
                avg_ratio = sum(ratios) / len(ratios)
                return avg_ratio
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error calculating R:R ratio: {e}")
            return 0.0

    def calculate_profit_factor(self, days=30, epic=None):
        """Calculate the profit factor (gross profit / gross loss)."""
        try:
            cursor = self.conn.cursor()
            
            # Calculate the date range
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=days)
            
            query = """
            SELECT 
                SUM(CASE WHEN pnl > 0 THEN pnl ELSE 0 END) as gross_profit,
                SUM(CASE WHEN pnl < 0 THEN ABS(pnl) ELSE 0 END) as gross_loss
            FROM trades
            WHERE timestamp BETWEEN ? AND ?
            AND pnl IS NOT NULL
            """
            
            params = [start_date.isoformat(), end_date.isoformat()]
            
            if epic:
                query += " AND epic = ?"
                params.append(epic)
            
            cursor.execute(query, params)
            result = cursor.fetchone()
            
            gross_profit, gross_loss = result
            
            # Handle None values from the query
            if gross_profit is None:
                gross_profit = 0.0
            if gross_loss is None:
                gross_loss = 0.0
            
            if gross_loss > 0:
                profit_factor = gross_profit / gross_loss
                return profit_factor
            elif gross_profit > 0:
                return float('inf')  # No losses but some profits
            else:
                return 0.0  # No profits or losses
                
        except Exception as e:
            logger.error(f"Error calculating profit factor: {e}")
            return 0.0

    def calculate_expectancy(self, days=30, epic=None):
        """Calculate the mathematical expectancy of the trading system."""
        try:
            win_rate = self.calculate_win_rate(days, epic)
            
            cursor = self.conn.cursor()
            
            # Calculate the date range
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=days)
            
            query = """
            SELECT 
                AVG(CASE WHEN pnl > 0 THEN pnl ELSE NULL END) as avg_win,
                AVG(CASE WHEN pnl < 0 THEN pnl ELSE NULL END) as avg_loss
            FROM trades
            WHERE timestamp BETWEEN ? AND ?
            AND pnl IS NOT NULL
            """
            
            params = [start_date.isoformat(), end_date.isoformat()]
            
            if epic:
                query += " AND epic = ?"
                params.append(epic)
            
            cursor.execute(query, params)
            result = cursor.fetchone()
            
            avg_win, avg_loss = result
            
            # Handle None values
            if avg_win is None:
                avg_win = 0.0
            if avg_loss is None:
                avg_loss = 0.0
            
            # Calculate expectancy: (Win% * Avg Win) + (Loss% * Avg Loss)
            expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
            return expectancy
                
        except Exception as e:
            logger.error(f"Error calculating expectancy: {e}")
            return 0.0

    def calculate_sharpe_ratio(self, days=30, epic=None, risk_free_rate=0.02):
        """Calculate the Sharpe ratio of returns."""
        try:
            cursor = self.conn.cursor()
            
            # Calculate the date range
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=days)
            
            query = """
            SELECT 
                timestamp,
                pnl
            FROM trades
            WHERE timestamp BETWEEN ? AND ?
            AND pnl IS NOT NULL
            """
            
            params = [start_date.isoformat(), end_date.isoformat()]
            
            if epic:
                query += " AND epic = ?"
                params.append(epic)
                
            query += " ORDER BY timestamp"
            
            cursor.execute(query, params)
            
            # Calculate returns for Sharpe ratio
            returns = []
            current_balance = 100000  # Arbitrary starting balance
            
            for _, pnl in cursor.fetchall():
                # Handle None values
                if pnl is None:
                    pnl = 0.0
                    
                return_pct = pnl / current_balance
                returns.append(return_pct)
                current_balance += pnl
            
            if returns and len(returns) > 1:  # Need at least 2 data points for std
                daily_returns = pd.Series(returns)
                daily_rfr = risk_free_rate / 252  # Convert annual to daily
                
                excess_returns = daily_returns - daily_rfr
                
                # Avoid division by zero
                std_dev = excess_returns.std()
                if std_dev > 0:
                    sharpe = np.sqrt(252) * (excess_returns.mean() / std_dev)
                    return sharpe
                else:
                    return 0.0
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0

    def analyze_trade_by_condition(self, condition_field, days=30):
        """Analyze trade performance by a specific condition (e.g., market_regime)."""
        try:
            cursor = self.conn.cursor()
            
            # Calculate the date range
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=days)
            
            query = f"""
            SELECT 
                {condition_field},
                COUNT(*) as total_trades,
                SUM(CASE WHEN outcome = 'PROFIT' THEN 1 ELSE 0 END) as winning_trades,
                SUM(CASE WHEN outcome = 'LOSS' THEN 1 ELSE 0 END) as losing_trades,
                SUM(pnl) as net_pnl
            FROM trades
            WHERE timestamp BETWEEN ? AND ?
            AND {condition_field} IS NOT NULL
            GROUP BY {condition_field}
            """
            
            params = [start_date.isoformat(), end_date.isoformat()]
            
            cursor.execute(query, params)
            
            # Get column names
            columns = [description[0] for description in cursor.description]
            
            # Fetch data and create a list of dictionaries
            results = []
            for row in cursor.fetchall():
                result_dict = dict(zip(columns, row))
                
                # Handle None values
                if result_dict['total_trades'] is None:
                    result_dict['total_trades'] = 0
                if result_dict['winning_trades'] is None:
                    result_dict['winning_trades'] = 0
                
                # Calculate win rate
                if result_dict['total_trades'] > 0:
                    result_dict['win_rate'] = result_dict['winning_trades'] / result_dict['total_trades']
                else:
                    result_dict['win_rate'] = 0
                    
                results.append(result_dict)
                
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing trade by {condition_field}: {e}")
            return []

    def analyze_trade_patterns(self):
        """Identify and analyze trade patterns."""
        try:
            # This would be a complex implementation depending on pattern criteria
            # For now, we'll implement a simple example: analyzing patterns by time of day
            cursor = self.conn.cursor()
            
            query = """
            SELECT 
                substr(timestamp, 12, 2) as hour_of_day,
                COUNT(*) as total_trades,
                SUM(CASE WHEN outcome = 'PROFIT' THEN 1 ELSE 0 END) as winning_trades,
                SUM(pnl) as net_pnl
            FROM trades
            WHERE outcome IN ('PROFIT', 'LOSS')
            GROUP BY hour_of_day
            ORDER BY hour_of_day
            """
            
            cursor.execute(query)
            
            # Get column names
            columns = [description[0] for description in cursor.description]
            
            # Fetch data and create a list of dictionaries
            time_patterns = []
            for row in cursor.fetchall():
                result_dict = dict(zip(columns, row))
                
                # Handle None values
                if result_dict['total_trades'] is None:
                    result_dict['total_trades'] = 0
                if result_dict['winning_trades'] is None:
                    result_dict['winning_trades'] = 0
                if result_dict['net_pnl'] is None:
                    result_dict['net_pnl'] = 0
                
                # Calculate win rate
                if result_dict['total_trades'] > 0:
                    result_dict['win_rate'] = result_dict['winning_trades'] / result_dict['total_trades']
                else:
                    result_dict['win_rate'] = 0
                    
                time_patterns.append(result_dict)
            
            # Store the pattern analysis
            for pattern in time_patterns:
                self.store_trade_pattern(
                    pattern_name=f"Hour_{pattern['hour_of_day']}",
                    success_rate=pattern['win_rate'],
                    avg_pnl=pattern['net_pnl'] / pattern['total_trades'] if pattern['total_trades'] > 0 else 0,
                    count=pattern['total_trades'],
                    pattern_data=pattern
                )
                
            return time_patterns
            
        except Exception as e:
            logger.error(f"Error analyzing trade patterns: {e}")
            return []

    def store_trade_pattern(self, pattern_name, success_rate, avg_pnl, count, pattern_data=None, epic=None, direction=None):
        """Store or update a trade pattern."""
        try:
            cursor = self.conn.cursor()
            
            # Check if pattern already exists
            query = "SELECT id FROM trade_patterns WHERE pattern_name = ?"
            params = [pattern_name]
            
            if epic:
                query += " AND epic = ?"
                params.append(epic)
                
            if direction:
                query += " AND direction = ?"
                params.append(direction)
                
            cursor.execute(query, params)
            result = cursor.fetchone()
            
            now = datetime.now(timezone.utc).isoformat()
            
            if result:
                # Update existing pattern
                pattern_id = result[0]
                query = """
                UPDATE trade_patterns 
                SET success_rate = ?, avg_pnl = ?, count = ?, pattern_data = ?, last_updated = ?
                WHERE id = ?
                """
                cursor.execute(query, [
                    success_rate, 
                    avg_pnl, 
                    count, 
                    json.dumps(pattern_data) if pattern_data else None,
                    now,
                    pattern_id
                ])
            else:
                # Insert new pattern
                query = """
                INSERT INTO trade_patterns 
                (pattern_name, epic, direction, success_rate, avg_pnl, count, pattern_data, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """
                cursor.execute(query, [
                    pattern_name,
                    epic,
                    direction,
                    success_rate,
                    avg_pnl,
                    count,
                    json.dumps(pattern_data) if pattern_data else None,
                    now
                ])
            
            self.conn.commit()
            logger.debug(f"Trade pattern stored/updated: {pattern_name}")
            
        except Exception as e:
            logger.error(f"Error storing trade pattern: {e}")
            return None

    def get_trade_patterns(self, min_trades=5):
        """Get stored trade patterns with minimum number of occurrences."""
        try:
            cursor = self.conn.cursor()
            
            query = """
            SELECT * FROM trade_patterns
            WHERE count >= ?
            ORDER BY success_rate DESC
            """
            
            cursor.execute(query, [min_trades])
            
            # Get column names
            columns = [description[0] for description in cursor.description]
            
            # Fetch data and create a list of dictionaries
            patterns = []
            for row in cursor.fetchall():
                pattern_dict = dict(zip(columns, row))
                
                # Parse JSON fields
                if pattern_dict.get('pattern_data'):
                    try:
                        pattern_dict['pattern_data'] = json.loads(pattern_dict['pattern_data'])
                    except:
                        pattern_dict['pattern_data'] = {}
                        
                patterns.append(pattern_dict)
                
            return patterns
            
        except Exception as e:
            logger.error(f"Error getting trade patterns: {e}")
            return []

    def update_performance_metrics(self):
        """Calculate and store current performance metrics."""
        try:
            now = datetime.now(timezone.utc).isoformat()
            cursor = self.conn.cursor()
            
            # Time frames to calculate metrics for
            timeframes = [
                {'name': 'daily', 'days': 1},
                {'name': 'weekly', 'days': 7},
                {'name': 'monthly', 'days': 30},
                {'name': 'quarterly', 'days': 90}
            ]
            
            # Get all unique epics from trades
            cursor.execute("SELECT DISTINCT epic FROM trades")
            epics = [row[0] for row in cursor.fetchall()]
            epics.append(None)  # For overall metrics
            
            for timeframe in timeframes:
                for epic in epics:
                    # Calculate metrics for this timeframe and epic
                    metrics = {
                        'win_rate': self.calculate_win_rate(timeframe['days'], epic),
                        'profit_factor': self.calculate_profit_factor(timeframe['days'], epic),
                        'expectancy': self.calculate_expectancy(timeframe['days'], epic),
                        'sharpe_ratio': self.calculate_sharpe_ratio(timeframe['days'], epic)
                    }
                    
                    # Store each metric
                    for metric_name, metric_value in metrics.items():
                        query = """
                        INSERT INTO performance_metrics 
                        (timestamp, metric_name, metric_value, timeframe, epic)
                        VALUES (?, ?, ?, ?, ?)
                        """
                        cursor.execute(query, [
                            now,
                            metric_name,
                            metric_value,
                            timeframe['name'],
                            epic
                        ])
            
            self.conn.commit()
            logger.info(f"Performance metrics updated.")
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")

    def get_performance_dashboard(self, timeframe='monthly'):
        """Get comprehensive performance dashboard data."""
        # Check if we have a recent cache
        now = datetime.now(timezone.utc)
        if (now - self._last_cache_update < self._cache_ttl and 
            timeframe in self._metrics_cache):
            return self._metrics_cache[timeframe]
        
        try:
            # Convert timeframe to days
            days = {
                'daily': 1,
                'weekly': 7,
                'monthly': 30,
                'quarterly': 90
            }.get(timeframe, 30)
            
            # Get overall metrics
            overall = {
                'win_rate': self.calculate_win_rate(days),
                'profit_factor': self.calculate_profit_factor(days),
                'expectancy': self.calculate_expectancy(days),
                'sharpe_ratio': self.calculate_sharpe_ratio(days),
                'risk_reward_ratio': self.calculate_risk_reward_ratio(days)
            }
            
            # Get per-instrument metrics
            cursor = self.conn.cursor()
            cursor.execute("SELECT DISTINCT epic FROM trades")
            epics = [row[0] for row in cursor.fetchall()]
            
            instruments = {}
            for epic in epics:
                instruments[epic] = {
                    'win_rate': self.calculate_win_rate(days, epic),
                    'profit_factor': self.calculate_profit_factor(days, epic),
                    'expectancy': self.calculate_expectancy(days, epic),
                    'trade_count': 0,
                    'net_pnl': 0
                }
            
            # Get trade history summary for counts and PnL
            trade_summary = self.get_trade_history_summary(days)
            for record in trade_summary:
                epic = record['epic']
                if epic in instruments:
                    instruments[epic]['trade_count'] = record['total_trades']
                    instruments[epic]['net_pnl'] = record['net_pnl'] if record['net_pnl'] is not None else 0
            
            # Get condition analysis
            market_regimes = self.analyze_trade_by_condition('market_regime', days)
            confidence_levels = self.analyze_trade_by_condition('confidence', days)
            
            # Get patterns
            patterns = self.get_trade_patterns()
            
            # Compile dashboard
            dashboard = {
                'overall': overall,
                'instruments': instruments,
                'market_regimes': market_regimes,
                'confidence_levels': confidence_levels,
                'patterns': patterns,
                'timeframe': timeframe,
                'days': days,
                'timestamp': now.isoformat()
            }
            
            # Cache the results
            self._metrics_cache[timeframe] = dashboard
            self._last_cache_update = now
            
            return dashboard
            
        except Exception as e:
            logger.error(f"Error getting performance dashboard: {e}")
            return {'error': str(e)}

    def get_trading_recommendations(self):
        """Generate recommendations based on historical performance."""
        try:
            recommendations = []
            
            # Analyze patterns to find best performing conditions
            market_regimes = self.analyze_trade_by_condition('market_regime', 90)
            best_regimes = [r for r in market_regimes if r['win_rate'] > 0.55 and r['total_trades'] >= 5]
            
            for regime in best_regimes:
                regime_name = regime.get('market_regime')
                if regime_name:
                    recommendations.append({
                        'type': 'market_regime',
                        'condition': regime_name,
                        'win_rate': regime['win_rate'],
                        'net_pnl': regime['net_pnl'],
                        'trade_count': regime['total_trades'],
                        'message': f"Prioritize trades in {regime_name} markets (Win rate: {regime['win_rate']:.2f})"
                    })
            
            # Get best performing instruments
            cursor = self.conn.cursor()
            query = """
            SELECT 
                epic,
                direction,
                COUNT(*) as total_trades,
                SUM(CASE WHEN outcome = 'PROFIT' THEN 1 ELSE 0 END) as winning_trades,
                SUM(pnl) as net_pnl
            FROM trades
            WHERE outcome IN ('PROFIT', 'LOSS')
            GROUP BY epic, direction
            HAVING total_trades >= 5
            """
            
            cursor.execute(query)
            
            columns = [description[0] for description in cursor.description]
            instruments = []
            for row in cursor.fetchall():
                inst_dict = dict(zip(columns, row))
                # Handle None values
                if inst_dict['winning_trades'] is None:
                    inst_dict['winning_trades'] = 0
                if inst_dict['total_trades'] is None or inst_dict['total_trades'] == 0:
                    inst_dict['win_rate'] = 0
                else:
                    inst_dict['win_rate'] = inst_dict['winning_trades'] / inst_dict['total_trades']
                instruments.append(inst_dict)
            
            # Sort by win rate
            instruments.sort(key=lambda x: x['win_rate'], reverse=True)
            
            for inst in instruments[:3]:  # Top 3
                if inst['win_rate'] > 0.5:
                    recommendations.append({
                        'type': 'instrument',
                        'epic': inst['epic'],
                        'direction': inst['direction'],
                        'win_rate': inst['win_rate'],
                        'net_pnl': inst['net_pnl'],
                        'trade_count': inst['total_trades'],
                        'message': f"Favor {inst['direction']} trades on {inst['epic']} (Win rate: {inst['win_rate']:.2f})"
                    })
            
            # Get time-based patterns
            time_patterns = self.analyze_trade_patterns()
            best_hours = [p for p in time_patterns if p['win_rate'] > 0.6 and p['total_trades'] >= 3]
            
            for hour in best_hours:
                recommendations.append({
                    'type': 'time_pattern',
                    'hour': hour['hour_of_day'],
                    'win_rate': hour['win_rate'],
                    'net_pnl': hour['net_pnl'],
                    'trade_count': hour['total_trades'],
                    'message': f"Trading at hour {hour['hour_of_day']} UTC shows higher success (Win rate: {hour['win_rate']:.2f})"
                })
            
            # Add position sizing recommendations
            profit_factor = self.calculate_profit_factor(30)
            if profit_factor > 1.5:
                recommendations.append({
                    'type': 'position_sizing',
                    'metric': 'profit_factor',
                    'value': profit_factor,
                    'message': f"System showing strong profit factor ({profit_factor:.2f}). Consider increasing position sizes."
                })
            elif profit_factor < 1.0:
                recommendations.append({
                    'type': 'position_sizing',
                    'metric': 'profit_factor',
                    'value': profit_factor,
                    'message': f"System showing weak profit factor ({profit_factor:.2f}). Reduce position sizes until performance improves."
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating trading recommendations: {e}")
            return []

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Trade memory database connection closed.")