import ccxt
import time
import pandas as pd
import numpy as np
import logging
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from groq import Groq
from typing import Optional, Dict
from flask import Flask, jsonify
import threading
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

app = Flask(__name__)

class TradingBot:
    def __init__(self, api_key: str, secret: str, groq_api_key: str):
        self.api_key = api_key
        self.secret = secret
        self.groq_api_key = groq_api_key
        # Define feature names first, before any method calls
        self.feature_names = ['rsi', 'ma5', 'ma10', 'ma20', 'momentum', 'ma_crossover', 'volume_change', 
                              'bb_upper', 'bb_lower', 'macd', 'signal', 'lag1', 'lag2', 'atr']
        self.exchange = self._initialize_exchange()
        self.symbol = self._get_symbol()
        self.leverage = 20
        self._set_leverage()
        self._set_position_mode()
        self.tick_size = self._get_tick_size()
        self.position_size = 0.002
        self.profit_target = 0.30
        self.stop_loss = -0.10
        self.model = self._initialize_ml_model()
        self.groq_client = Groq(api_key=self.groq_api_key)
        self.last_close_time = None
        self.running = False
        self.status = {"message": "Bot initialized", "position": None, "last_action": None}
        self.trailing_activation = 0.25
        self.trailing_distance = 0.10
        self.trailing_stop_price = None
        self.highest_profit = 0.0
        self.trade_history = []
        logging.info("Trading bot initialized with ML model (LightGBM), Groq API, and trailing stop loss")

    def _initialize_exchange(self) -> ccxt.phemex:
        try:
            exchange = ccxt.phemex({
                'apiKey': self.api_key,
                'secret': self.secret,
                'enableRateLimit': True,
                'options': {'defaultType': 'swap'}
            })
            balance = exchange.fetch_balance()
            usdt_balance = balance['total'].get('USDT', 0)
            logging.info(f"Connected to Phemex Live Futures | USDT Balance: {usdt_balance}")
            exchange.load_markets()
            if 'BTC/USDT:USDT' not in exchange.markets:
                raise ValueError(f"BTC/USDT:USDT not found! Available: {list(exchange.markets.keys())}")
            return exchange
        except Exception as e:
            logging.error(f"Error initializing exchange: {e}")
            raise

    def _set_leverage(self) -> None:
        try:
            self.exchange.set_leverage(self.leverage, self.symbol)
            logging.info(f"Leverage set to {self.leverage}x for {self.symbol}")
        except Exception as e:
            logging.error(f"Error setting leverage: {e}")

    def _set_position_mode(self) -> None:
        try:
            self.exchange.set_position_mode(False, self.symbol)
            logging.info("Position mode set to One-Way")
        except Exception as e:
            logging.error(f"Error setting position mode: {e}")

    def _get_symbol(self) -> str:
        try:
            self.exchange.load_markets()
            symbol = 'BTC/USDT:USDT'
            if symbol not in self.exchange.markets:
                raise ValueError(f"BTC/USDT:USDT not found! Available: {list(exchange.markets.keys())}")
            market = self.exchange.markets[symbol]
            logging.info(f"Trading Pair: {symbol} | Type: {market['type']} | Precision: {market['precision']['price']}")
            return symbol
        except Exception as e:
            logging.error(f"Error loading markets: {e}")
            raise

    def _get_tick_size(self) -> float:
        try:
            market = self.exchange.market(self.symbol)
            tick_size = market['precision']['price']
            logging.info(f"Tick size: {tick_size}")
            return tick_size
        except Exception as e:
            logging.error(f"Error getting tick size: {e}")
            return 0.1

    @staticmethod
    def calculate_rsi(data: pd.Series, periods: int = 14) -> pd.Series:
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def get_current_price(self) -> Optional[float]:
        try:
            ticker = self.exchange.fetch_ticker(self.symbol)
            return ticker['last']
        except Exception as e:
            logging.error(f"Error fetching price: {e}")
            return None

    def fetch_historical_data(self, timeframe: str = '5m', limit: int = 500, pages: int = 1, use_binance: bool = False) -> pd.DataFrame:
        try:
            exchange = ccxt.binance() if use_binance else self.exchange
            symbol = 'BTCUSDT' if use_binance else self.symbol
            all_data = []
            since = int(time.time() * 1000) - (60 * 24 * 60 * 60 * 1000) if pages > 1 else None
            for page in range(pages):
                logging.info(f"Fetching page {page + 1}/{pages} | Symbol: {symbol} | Timeframe: {timeframe} | Limit: {limit} | Since: {since}")
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
                if not ohlcv:
                    logging.warning("No data returned for this page")
                    break
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                all_data.append(df)
                since = ohlcv[-1][0] + 1
                time.sleep(1)
            if all_data:
                df = pd.concat(all_data).drop_duplicates(subset='timestamp').reset_index(drop=True)
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                logging.info(f"Fetched {len(df)} candles of historical data")
                return df
            return pd.DataFrame()
        except Exception as e:
            logging.error(f"Error fetching historical data: {e}")
            return pd.DataFrame()

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df['ma5'] = df['close'].rolling(window=5).mean()
        df['ma10'] = df['close'].rolling(window=10).mean()
        df['ma20'] = df['close'].rolling(window=20).mean()
        df['rsi'] = self.calculate_rsi(df['close'])
        df['momentum'] = df['close'].pct_change(periods=5)
        df['ma_crossover'] = np.where(df['ma5'] > df['ma10'], 1, 0)
        df['volume_change'] = df['volume'].pct_change()
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
        df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema12'] - df['ema26']
        df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['lag1'] = df['close'].shift(1)
        df['lag2'] = df['close'].shift(2)
        tr = np.maximum(df['high'] - df['low'], 
                        np.maximum(abs(df['high'] - df['close'].shift(1)), 
                                  abs(df['low'] - df['close'].shift(1))))
        df['atr'] = tr.rolling(window=14).mean()
        return df[['close', 'rsi', 'ma5', 'ma10', 'ma20', 'momentum', 'ma_crossover', 'volume_change', 
                   'bb_upper', 'bb_lower', 'macd', 'signal', 'lag1', 'lag2', 'atr']].dropna()

    def _initialize_ml_model(self) -> LGBMClassifier:
        df = self.fetch_historical_data(timeframe='5m', limit=1000, pages=19, use_binance=True)
        logging.info(f"Raw candles fetched: {len(df)}")
        df = self.prepare_data(df)
        logging.info(f"Processed candles after feature prep: {len(df)}")
        if df.empty or len(df) < 50:
            logging.warning("Insufficient real data for ML training, using default model")
            return LGBMClassifier(n_estimators=100, random_state=42)
        df['pct_change'] = df['close'].pct_change().shift(-1)
        df['target'] = np.where(df['pct_change'] > 0.005, 1, np.where(df['pct_change'] < -0.005, 0, np.nan))
        df = df.dropna()
        logging.info(f"Samples after target filtering: {len(df)}")
        X = df[self.feature_names]
        y = df['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LGBMClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            verbose=-1
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logging.info(f"ML Model (LightGBM) trained with {len(X_train)} samples | Test Accuracy: {accuracy:.2f}")
        scores = cross_val_score(model, X, y, cv=5)
        logging.info(f"CV scores: {scores.mean():.2f} (+/- {scores.std() * 2:.2f})")
        return model

    def _retrain_ml_model(self, entry_price: float, exit_price: float, direction: str) -> None:
        try:
            trade = {
                'entry_price': entry_price,
                'exit_price': exit_price,
                'direction': direction,
                'timestamp': time.time() * 1000
            }
            self.trade_history.append(trade)
            logging.info(f"Trade added to history: {trade}")

            df = self.fetch_historical_data(timeframe='5m', limit=1000, pages=19, use_binance=True)
            logging.info(f"Raw candles fetched for retraining: {len(df)}")
            df = self.prepare_data(df)
            logging.info(f"Processed candles after feature prep: {len(df)}")

            if df.empty or len(df) < 50:
                logging.warning("Insufficient data for retraining, skipping...")
                return

            latest_candle = df.iloc[-1].copy()
            pct_change = (exit_price - entry_price) / entry_price if direction == 'long' else (entry_price - exit_price) / entry_price
            target = 1 if pct_change > 0.005 else 0 if pct_change < -0.005 else np.nan
            if not np.isnan(target):
                latest_candle['pct_change'] = pct_change
                latest_candle['target'] = target
                df = pd.concat([df, pd.DataFrame([latest_candle])], ignore_index=True)

            df['pct_change'] = df['close'].pct_change().shift(-1)
            df['target'] = np.where(df['pct_change'] > 0.005, 1, np.where(df['pct_change'] < -0.005, 0, np.nan))
            df = df.dropna()
            logging.info(f"Samples after target filtering for retraining: {len(df)}")

            if len(df) < 50:
                logging.warning("Too few samples after filtering, skipping retraining...")
                return

            X = df[self.feature_names]
            y = df['target']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            self.model = LGBMClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                verbose=-1
            )
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            logging.info(f"ML Model retrained with {len(X_train)} samples | Test Accuracy: {accuracy:.2f}")
            scores = cross_val_score(self.model, X, y, cv=5)
            logging.info(f"CV scores after retraining: {scores.mean():.2f} (+/- {scores.std() * 2:.2f})")
        except Exception as e:
            logging.error(f"Error retraining ML model: {e}")

    def consult_groq(self, data: pd.DataFrame) -> str:
        try:
            latest_data = data.iloc[-1]
            trend = "up" if data['close'].iloc[-1] > data['close'].iloc[-5] else "down"
            X_latest = pd.DataFrame([latest_data[self.feature_names]], columns=self.feature_names)
            ml_pred = self.model.predict(X_latest)[0]
            ml_direction = 'long' if ml_pred == 1 else 'short'
            prompt = (
                f"Analyze this 5-minute BTC/USDT:USDT data from an ML model:\n"
                f"Close: {latest_data['close']:.2f}, RSI: {latest_data['rsi']:.2f}, "
                f"MA5: {latest_data['ma5']:.2f}, MA10: {latest_data['ma10']:.2f}, MA20: {latest_data['ma20']:.2f}, "
                f"Momentum: {latest_data['momentum']:.4f}, MA Crossover: {latest_data['ma_crossover']}, "
                f"Volume Change: {latest_data['volume_change']:.4f}, "
                f"BB Upper: {latest_data['bb_upper']:.2f}, BB Lower: {latest_data['bb_lower']:.2f}, "
                f"MACD: {latest_data['macd']:.4f}, Signal: {latest_data['signal']:.4f}, "
                f"ATR: {latest_data['atr']:.2f}, "
                f"Recent Trend (25m): {trend}, ML Prediction (LightGBM): {ml_direction}.\n"
                f"Should I go 'long' or 'short' in a leveraged futures trade? Respond with only 'long' or 'short'."
            )
            response = self.groq_client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0.3
            )
            decision = response.choices[0].message.content.strip().lower()
            logging.info(f"Groq recommends: {decision}")
            return decision if decision in ['long', 'short'] else 'long'
        except Exception as e:
            logging.error(f"Groq consultation failed: {e}")
            return 'long'

    def place_market_order(self, direction: str) -> bool:
        try:
            current_price = self.get_current_price()
            logging.info(f"Current price before order: {current_price}")
            if direction == 'long':
                order = self.exchange.create_market_buy_order(self.symbol, self.position_size)
            else:
                order = self.exchange.create_market_sell_order(self.symbol, self.position_size)
            logging.info(f"Placed market {direction} order | Order ID: {order['id']} | "
                         f"Price: {order['price'] or 'Market'} | Amount: {order['amount']}")
            self.status["last_action"] = f"Placed {direction} order"
            if order['price'] and abs((order['price'] - current_price) / current_price) > 0.05:
                logging.warning(f"Slippage exceeded 5%: Entry {order['price']} vs Current {current_price}")
            return True
        except Exception as e:
            logging.error(f"Error placing market order: {e}")
            return False

    def _close_position(self, direction: str, entry_price: float, exit_price: float) -> None:
        try:
            position_qty = self.position_size
            if direction == 'long':
                self.exchange.create_market_sell_order(self.symbol, position_qty, {'reduceOnly': True})
            else:
                self.exchange.create_market_buy_order(self.symbol, position_qty, {'reduceOnly': True})
            logging.info(f"Closed {direction} leveraged position | Entry: {entry_price:.2f} | Exit: {exit_price:.2f}")
            self.last_close_time = time.time()
            self.status["last_action"] = f"Closed {direction} position"
            self.trailing_stop_price = None
            self.highest_profit = 0.0
            self._retrain_ml_model(entry_price, exit_price, direction)
        except Exception as e:
            logging.error(f"Error closing position: {e}")

    def get_open_positions(self) -> Dict:
        try:
            positions = self.exchange.fetch_positions([self.symbol])
            for pos in positions:
                if pos['symbol'] == self.symbol and float(pos['contracts']) > 0:
                    position = {
                        'side': pos['side'],
                        'size': float(pos['contracts']),
                        'entry_price': float(pos['entryPrice']),
                        'current_price': float(pos['markPrice']),
                        'unrealized_pnl': float(pos['unrealizedPnl']),
                        'leverage': float(pos['leverage'])
                    }
                    self.status["position"] = position
                    return position
            self.status["position"] = None
            return {}
        except Exception as e:
            logging.error(f"Error fetching positions: {e}")
            return {}

    def run(self) -> None:
        logging.info("Starting trading bot with ML data (LightGBM), Groq decisions, 5-min delay, and trailing stop loss...")
        self.running = True
        decision_time = None
        post_close_delay = 900
        
        while self.running:
            try:
                current_time = time.time()
                position = self.get_open_positions()
                current_price = self.get_current_price()
                
                if position:
                    direction = position['side']
                    entry_price = position['entry_price']
                    size = position['size']
                    pl = (current_price - entry_price) / entry_price * position['leverage'] \
                        if direction == 'long' else (entry_price - current_price) / entry_price * position['leverage']
                    logging.info(f"Monitoring {direction} | Size: {size} | Entry: {entry_price:.2f} | "
                                 f"Current: {current_price:.2f} | P/L: {pl*100:.2f}%")
                    self.status["message"] = f"Monitoring {direction} position, P/L: {pl*100:.2f}%"
                    
                    self.highest_profit = max(self.highest_profit, pl)
                    
                    if pl >= self.profit_target:
                        logging.info(f"Profit target {self.profit_target*100}% reached")
                        self._close_position(direction, entry_price, current_price)
                    elif pl <= self.stop_loss:
                        logging.info(f"Initial stop loss {self.stop_loss*100}% triggered")
                        self._close_position(direction, entry_price, current_price)
                    elif self.highest_profit >= self.trailing_activation:
                        if direction == 'long':
                            highest_price = entry_price * (1 + self.highest_profit / position['leverage'])
                            new_trailing_stop = highest_price * (1 - self.trailing_distance)
                            self.trailing_stop_price = max(self.trailing_stop_price or 0, new_trailing_stop)
                            logging.info(f"Trailing stop active | Highest P/L: {self.highest_profit*100:.2f}% | "
                                         f"Stop Price: {self.trailing_stop_price:.2f}")
                            if current_price <= self.trailing_stop_price:
                                logging.info(f"Trailing stop triggered at {self.trailing_stop_price:.2f} | P/L: {pl*100:.2f}%")
                                self._close_position(direction, entry_price, current_price)
                        else:
                            highest_price = entry_price * (1 - self.highest_profit / position['leverage'])
                            new_trailing_stop = highest_price * (1 + self.trailing_distance)
                            self.trailing_stop_price = min(self.trailing_stop_price or float('inf'), new_trailing_stop)
                            logging.info(f"Trailing stop active | Highest P/L: {self.highest_profit*100:.2f}% | "
                                         f"Stop Price: {self.trailing_stop_price:.2f}")
                            if current_price >= self.trailing_stop_price:
                                logging.info(f"Trailing stop triggered at {self.trailing_stop_price:.2f} | P/L: {pl*100:.2f}%")
                                self._close_position(direction, entry_price, current_price)
                    else:
                        logging.info(f"P/L {pl*100:.2f}% below trailing activation {self.trailing_activation*100}%")
                else:
                    logging.info(f"No open positions | Current Price: {current_price or 'N/A'}")
                    self.status["message"] = "No open positions"
                    
                    if self.last_close_time is not None:
                        time_since_close = current_time - self.last_close_time
                        if time_since_close < post_close_delay:
                            remaining_time = post_close_delay - time_since_close
                            logging.info(f"Delaying ML/AI analysis after trade closure... {remaining_time:.1f} seconds remaining")
                            self.status["message"] = f"Delaying ML/AI analysis, {remaining_time:.1f}s remaining"
                            time.sleep(min(5, remaining_time))
                            continue
                    
                    df = self.fetch_historical_data(timeframe='5m', limit=100, pages=1)
                    df_processed = self.prepare_data(df)
                    if not df_processed.empty:
                        if decision_time is None:
                            decision_time = current_time
                            logging.info("Groq analysis started—60 seconds until trade decision")
                            self.status["message"] = "Groq analysis started"
                        
                        time_elapsed = current_time - decision_time
                        if time_elapsed >= 60:
                            direction = self.consult_groq(df_processed)
                            logging.info(f"Groq decision after 60s analysis: {direction}")
                            self.status["message"] = f"Groq decided: {direction}"
                            success = self.place_market_order(direction)
                            if success:
                                decision_time = None
                                self.trailing_stop_price = None
                                self.highest_profit = 0.0
                            else:
                                logging.error("Failed to place order, retrying next cycle")
                                self.status["message"] = "Failed to place order, retrying"
                        else:
                            logging.info(f"Groq analyzing... {60 - time_elapsed:.1f} seconds remaining")
                            self.status["message"] = f"Groq analyzing, {60 - time_elapsed:.1f}s remaining"
                    else:
                        logging.info("No sufficient data for prediction, waiting...")
                        self.status["message"] = "No sufficient data"
                        decision_time = None
            
            except Exception as e:
                logging.error(f"Loop error: {e}")
                self.status["message"] = f"Error: {str(e)}"
            
            time.sleep(5)

    def stop(self):
        self.running = False
        self.status["message"] = "Bot stopped"

# Global bot instance
bot = None

@app.route('/status', methods=['GET'])
def get_status():
    if bot is None:
        return jsonify({"error": "Bot not initialized"}), 500
    return jsonify(bot.status)

@app.route('/start', methods=['POST'])
def start_bot():
    global bot
    if bot is None or not bot.running:
        API_KEY = os.environ.get('API_KEY')
        SECRET = os.environ.get('SECRET')
        GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
        if not all([API_KEY, SECRET, GROQ_API_KEY]):
            return jsonify({"error": "Missing required environment variables"}), 400
        bot = TradingBot(API_KEY, SECRET, GROQ_API_KEY)
        threading.Thread(target=bot.run, daemon=True).start()
        return jsonify({"message": "Bot started"}), 200
    return jsonify({"message": "Bot already running"}), 200

@app.route('/stop', methods=['POST'])
def stop_bot():
    if bot is not None and bot.running:
        bot.stop()
        return jsonify({"message": "Bot stopped"}), 200
    return jsonify({"message": "Bot not running"}), 200

if __name__ == "__main__":
    API_KEY = os.environ.get('API_KEY')
    SECRET = os.environ.get('SECRET')
    GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
    if not all([API_KEY, SECRET, GROQ_API_KEY]):
        logging.error("Missing required environment variables: API_KEY, SECRET, or GROQ_API_KEY")
        exit(1)
    bot = TradingBot(API_KEY, SECRET, GROQ_API_KEY)
    threading.Thread(target=bot.run, daemon=True).start()
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 10000)))
