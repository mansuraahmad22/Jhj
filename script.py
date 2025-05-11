# COMPLETE TRADING BOT WITH ACCURATE TREND FLIP DETECTION

import ccxt
import pandas as pd
import numpy as np
from pybit.unified_trading import HTTP
from datetime import datetime, timedelta
import pytz
from twilio.rest import Client

# ======== GLOBAL SETTINGS ========
# Twilio settings
TWILIO_ACCOUNT_SID = 'AC5ab4c9778fff846bd300ac9fd0f16656'
TWILIO_AUTH_TOKEN = 'e1742ac942a79545b3c026066fba2425'
TWILIO_WHATSAPP_NUMBER = 'whatsapp:+14155238886'
YOUR_WHATSAPP_NUMBER = 'whatsapp:+2349060558418'

# Strategy parameters
TIMEFRAME = '15m'
LIMIT = 500
ORDER_AMOUNT_USDT = 50
EMA_LENGTH = 200

# Timezone settings
LAGOS_TZ = pytz.timezone('Africa/Lagos')
UTC_TZ = pytz.UTC

# Initialize exchanges
exchange = ccxt.bitget()
session = HTTP(
    api_key="lJu52hbBTbPkg2VXZ2",
    api_secret="e43RV6YDZsn24Q9mucr0i4xbU7YytdL2HtuV",
    demo=True
)

# Initialize Twilio client
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# ======== UTILITY FUNCTIONS ========
def send_whatsapp_message(body):
    """Send WhatsApp alerts using Twilio"""
    try:
        message = twilio_client.messages.create(
            from_=TWILIO_WHATSAPP_NUMBER,
            body=body,
            to=YOUR_WHATSAPP_NUMBER
        )
        print(f"WhatsApp message sent: {message.sid}")
        return True
    except Exception as e:
        print(f"WhatsApp message failed: {e}")
        return False

def get_pybit_symbol(ccxt_symbol):
    """Convert CCXT symbol to Bybit format"""
    return ccxt_symbol.split('/')[0] + ccxt_symbol.split('/')[1].split(':')[0]

def format_pnl(pnl):
    """Format PnL with proper sign and profit/loss indication"""
    if pnl > 0:
        return f"+{pnl:.2f}% (Profit)"
    elif pnl < 0:
        return f"{pnl:.2f}% (Loss)"
    return f"{pnl:.2f}% (Break-even)"

def fetch_market_data(symbol, timeframe, limit=500):
    """Fetch OHLCV data with proper timezone handling"""
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert(LAGOS_TZ)
        df.set_index('timestamp', inplace=True)
        df['EMA'] = df['close'].ewm(span=EMA_LENGTH, adjust=False).mean()
        return df
    except Exception as e:
        print(f"Error fetching market data: {e}")
        return None

def detect_trend(df, candle_index):
    """Determine trend for a specific candle using your exact method"""
    if candle_index < 1 or candle_index >= len(df):
        return "Sideways"
    
    ema_current = df['EMA'].iloc[candle_index]
    ema_prev = df['EMA'].iloc[candle_index-1]
    price = df['close'].iloc[candle_index]
    
    if ema_current > ema_prev and price > ema_current:
        return "Uptrend"
    elif ema_current < ema_prev and price < ema_current:
        return "Downtrend"
    return "Sideways"

def get_last_closed_trade():
    """Get details of the most recent closed trade with accurate trend detection"""
    try:
        trades = session.get_executions(category="linear", limit=50)
        if trades["retCode"] != 0:
            print(f"Error fetching trades: {trades['retMsg']}")
            return None
            
        trades = trades["result"]["list"]
        if not trades:
            return None

        for trade in sorted(trades, key=lambda x: int(x["execTime"]), reverse=True):
            symbol = trade["symbol"]
            
            # Convert symbol to CCXT format
            base = symbol[:-4] if symbol.endswith('USDT') else symbol[:-3]
            ccxt_symbol = f"{base}/USDT:USDT"
            
            positions = session.get_positions(category="linear", symbol=symbol)
            if positions["retCode"] != 0:
                continue
                
            position_open = any(float(p["size"]) > 0 for p in positions["result"]["list"])
            
            if not position_open and trade["closedSize"]:
                utc_time = datetime.fromtimestamp(int(trade["execTime"])/1000, UTC_TZ)
                lagos_time = utc_time.astimezone(LAGOS_TZ)
                
                # Get trend at close time
                df = fetch_market_data(ccxt_symbol, TIMEFRAME, LIMIT)
                if df is None or len(df) < 2:
                    trend_at_close = "Unknown"
                else:
                    close_candle_idx = df.index.get_indexer([lagos_time], method='nearest')[0]
                    if close_candle_idx < 1:
                        close_candle_idx = 1
                    trend_at_close = detect_trend(df, close_candle_idx)
                
                return {
                    "symbol": ccxt_symbol,
                    "bybit_symbol": symbol,
                    "close_time": lagos_time,
                    "close_price": float(trade["execPrice"]),
                    "side": "LONG" if trade["side"] == "Sell" else "SHORT",
                    "utc_close_time": utc_time,
                    "trend_at_close": trend_at_close,
                    "execution_side": trade["side"]  # Original execution side (Buy/Sell)
                }
        return None
    except Exception as e:
        print(f"Error fetching trade history: {e}")
        return None

def detect_trend_flip(trade_info):
    """Detect if trend has flipped since trade was closed"""
    if not trade_info:
        return None, "No trade info provided"
    
    try:
        df = fetch_market_data(trade_info['symbol'], TIMEFRAME, LIMIT)
        if df is None or len(df) < 2:
            return None, "Not enough market data"
        
        # Find the candle where trade was closed
        close_candle_idx = df.index.get_indexer([trade_info['close_time']], method='nearest')[0]
        if close_candle_idx < 1:
            close_candle_idx = 1
        
        # Check for counter-trend closing
        if (trade_info['side'] == "SHORT" and trade_info['trend_at_close'] == "Uptrend") or \
           (trade_info['side'] == "LONG" and trade_info['trend_at_close'] == "Downtrend"):
            return None, "Counter-trend closing detected - skipping flip check"
        
        current_trend = trade_info['trend_at_close']
        first_flip = None
        
        # Start checking from the next candle after close
        for i in range(close_candle_idx + 1, len(df)):
            new_trend = detect_trend(df, i)
            
            if new_trend != current_trend and new_trend in ["Uptrend", "Downtrend"]:
                first_flip = {
                    'time': df.index[i],
                    'new_trend': new_trend,
                    'price': df['close'].iloc[i],
                    'candle_time': df.index[i].strftime('%Y-%m-%d %H:%M:%S')
                }
                break
        
        if first_flip:
            duration = first_flip['time'] - trade_info['close_time']
            hours = int(duration.total_seconds() // 3600)
            minutes = int((duration.total_seconds() % 3600) // 60)
            return True, f"Trend flipped to {first_flip['new_trend']} at {first_flip['candle_time']} ({hours}h {minutes}m after close)"
        else:
            return False, "No trend flip detected since closing"
    except Exception as e:
        print(f"Error detecting trend flip: {e}")
        return None, f"Error detecting trend flip: {e}"

# ======== CORE TRADING FUNCTIONS ========
def calculate_atr(df, period=14):
    """Calculate ATR"""
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def has_open_orders(symbol):
    """Check for existing limit orders"""
    try:
        bybit_symbol = get_pybit_symbol(symbol)
        orders = session.get_open_orders(category="linear", symbol=bybit_symbol)
        return orders["retCode"] == 0 and len(orders["result"]["list"]) > 0
    except Exception as e:
        print(f"Order check error: {e}")
        return True

def has_open_position(symbol):
    """Check for open positions"""
    try:
        bybit_symbol = get_pybit_symbol(symbol)
        positions = session.get_positions(category="linear", symbol=bybit_symbol)
        if positions["retCode"] == 0:
            for pos in positions["result"]["list"]:
                if float(pos["size"]) > 0:
                    return True
        return False
    except Exception as e:
        print(f"Position check error: {e}")
        return True

def place_limit_order(symbol, side, price, order_amount_usdt=ORDER_AMOUNT_USDT):
    """Place limit order with proper risk management"""
    try:
        bybit_symbol = get_pybit_symbol(symbol)
        ticker = session.get_tickers(category="linear", symbol=bybit_symbol)
        current_price = float(ticker["result"]["list"][0]["lastPrice"])
        
        if (side == "Buy" and price >= current_price) or (side == "Sell" and price <= current_price):
            print(f"Invalid {side} limit price {price} vs current {current_price}")
            return False
            
        instrument = session.get_instruments_info(category="linear", symbol=bybit_symbol)
        qty_step = float(instrument["result"]["list"][0]["lotSizeFilter"]["qtyStep"])
        quantity = round((order_amount_usdt / price) / qty_step) * qty_step
        
        response = session.place_order(
            category="linear",
            symbol=bybit_symbol,
            side=side,
            orderType="Limit",
            qty=str(quantity),
            price=str(price),
            timeInForce="GTC"
        )
        
        if response["retCode"] == 0:
            msg = f"Placed {side} limit order at {price} for {quantity} {symbol.split('/')[0]}"
            print(msg)
            send_whatsapp_message(msg)
            return True
        else:
            print(f"Order failed: {response['retMsg']}")
            return False
    except Exception as e:
        print(f"Order placement error: {e}")
        return False

def place_market_order(symbol, side, order_amount_usdt=ORDER_AMOUNT_USDT):
    """Place market order with proper risk management"""
    try:
        bybit_symbol = get_pybit_symbol(symbol)
        ticker = session.get_tickers(category="linear", symbol=bybit_symbol)
        current_price = float(ticker["result"]["list"][0]["lastPrice"])
        
        instrument = session.get_instruments_info(category="linear", symbol=bybit_symbol)
        qty_step = float(instrument["result"]["list"][0]["lotSizeFilter"]["qtyStep"])
        quantity = round((order_amount_usdt / current_price) / qty_step) * qty_step
        
        response = session.place_order(
            category="linear",
            symbol=bybit_symbol,
            side=side,
            orderType="Market",
            qty=str(quantity)
        )
        
        if response["retCode"] == 0:
            msg = f"Placed {side} market order for {quantity} {symbol.split('/')[0]} at {current_price}"
            print(msg)
            send_whatsapp_message(msg)
            return True
        else:
            print(f"Order failed: {response['retMsg']}")
            return False
    except Exception as e:
        print(f"Order placement error: {e}")
        return False

def cancel_all_pending_orders(symbol):
    """Cancel all pending orders for our symbol"""
    try:
        bybit_symbol = get_pybit_symbol(symbol)
        response = session.cancel_all_orders(
            category="linear",
            symbol=bybit_symbol
        )
        if response["retCode"] == 0:
            print("All pending orders canceled")
            return True
        else:
            print(f"Failed to cancel orders: {response['retMsg']}")
            return False
    except Exception as e:
        print(f"Error canceling orders: {e}")
        return False

def close_position(symbol):
    """Close any open position for the symbol"""
    try:
        bybit_symbol = get_pybit_symbol(symbol)
        positions = session.get_positions(category="linear", symbol=bybit_symbol)
        
        if positions["retCode"] == 0 and positions["result"]["list"]:
            for pos in positions["result"]["list"]:
                if float(pos["size"]) > 0:
                    side = "Buy" if pos["side"] == "Sell" else "Sell"
                    response = session.place_order(
                        category="linear",
                        symbol=bybit_symbol,
                        side=side,
                        orderType="Market",
                        qty=pos["size"]
                    )
                    
                    if response["retCode"] == 0:
                        pnl = float(pos["unrealisedPnl"])
                        msg = f"Closed {pos['side']} position | PnL: {format_pnl(pnl)}"
                        print(msg)
                        send_whatsapp_message(msg)
                        return True
                    else:
                        print(f"Failed to close position: {response['retMsg']}")
                        return False
        return False
    except Exception as e:
        print(f"Error closing position: {e}")
        return False

# ======== TRADING STRATEGIES ========
def atr_strategy(trade_info):
    """ATR Mean Reversion Strategy with accurate trend detection"""
    if not trade_info:
        print("No trade info available for ATR strategy")
        return
    
    symbol = trade_info['symbol']
    print(f"\n[ATR STRATEGY] Checking {symbol} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Strategy settings
    atr_period = 14
    atr_multiplier = 2.0
    
    # 1. Check for existing positions/orders
    if has_open_position(symbol) or has_open_orders(symbol):
        print("Open position/order exists - doing nothing")
        return
    
    # 2. Get last trade direction and price
    last_side = trade_info['side']
    last_price = trade_info['close_price']
    last_time = trade_info['close_time']
    
    print(f"Last trade: {last_side} at {last_price} ({last_time.strftime('%Y-%m-%d %H:%M:%S')})")
    print(f"Trend at close: {trade_info['trend_at_close']}")
    
    # 3. Check for trend flip since last trade
    flip_detected, flip_message = detect_trend_flip(trade_info)
    if flip_detected is None:
        print(f"❌ Trend flip check error: {flip_message}")
        return
    elif flip_detected:
        print(f"❌ Rejected: {flip_message}")
        return
    
    # 4. Get current trend
    df = fetch_market_data(symbol, TIMEFRAME, atr_period+1)
    if df is None:
        print("❌ Failed to fetch market data")
        return
    
    current_trend = detect_trend(df, len(df)-1)
    print(f"Current trend: {current_trend}")
    
    # 5. Calculate 2xATR levels
    try:
        atr = calculate_atr(df, atr_period).iloc[-1]
        
        ticker = session.get_tickers(category="linear", symbol=get_pybit_symbol(symbol))
        current_price = float(ticker["result"]["list"][0]["lastPrice"])
        
        # 6. LONG Trade Logic
        if last_side == 'LONG':
            upper_projection = last_price + (atr * atr_multiplier)
            print(f"ATR: {atr:.2f} | Upper Projection: {upper_projection:.2f} | Current: {current_price:.2f}")
            
            if current_trend == "Uptrend" and current_price <= upper_projection:
                print("✅ Conditions met - placing LONG limit order")
                place_limit_order(symbol, "Buy", upper_projection)
            else:
                reject_reasons = []
                if current_trend != "Uptrend": reject_reasons.append(f"trend is {current_trend}")
                if current_price > upper_projection: reject_reasons.append("price beyond projection")
                print(f"❌ Rejected LONG: {', '.join(reject_reasons)}")
        
        # 7. SHORT Trade Logic
        elif last_side == 'SHORT':
            lower_projection = last_price - (atr * atr_multiplier)
            print(f"ATR: {atr:.2f} | Lower Projection: {lower_projection:.2f} | Current: {current_price:.2f}")
            
            if current_trend == "Downtrend" and current_price >= lower_projection:
                print("✅ Conditions met - placing SHORT limit order")
                place_limit_order(symbol, "Sell", lower_projection)
            else:
                reject_reasons = []
                if current_trend != "Downtrend": reject_reasons.append(f"trend is {current_trend}")
                if current_price < lower_projection: reject_reasons.append("price beyond projection")
                print(f"❌ Rejected SHORT: {', '.join(reject_reasons)}")
                
    except Exception as e:
        print(f"ATR calculation error: {e}")

def crossover_strategy(trade_info):
    """Crossover Strategy with accurate trend detection"""
    if not trade_info:
        print("No trade info available for Crossover strategy")
        return
    
    symbol = trade_info['symbol']
    print(f"\n[CROSSOVER STRATEGY] Checking {symbol} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Strategy settings
    h = 8.0
    mult = 3.0
    repaint = True
    
    def calculate_nwe(src, h, mult, repaint):
        """Calculate Nadaraya-Watson Envelope"""
        n = len(src)
        out = np.zeros(n)
        upper = np.zeros(n)
        lower = np.zeros(n)
        
        if repaint:
            nwe = []
            sae = 0.0
            for i in range(n):
                sum_val = sumw = 0.0
                for j in range(n):
                    w = np.exp(-((i-j)**2)/(h**2*2))
                    sum_val += src[j] * w
                    sumw += w
                y2 = sum_val / sumw
                nwe.append(y2)
                sae += abs(src[i] - y2)
            sae = (sae / n) * mult
            upper = [nwe[i] + sae for i in range(n)]
            lower = [nwe[i] - sae for i in range(n)]
        else:
            coefs = np.array([np.exp(-(i**2)/(h**2*2)) for i in range(n)])
            den = np.sum(coefs)
            out = np.sum(src * coefs) / den
            mae = pd.Series(np.abs(src - out)).rolling(499).mean().values * mult
            upper = out + mae
            lower = out - mae
        
        return out, upper, lower

    def detect_crossover(close, upper):
        """Detect if price has crossed above the upper envelope"""
        return (close.shift(1) < upper.shift(1)) & (close > upper)

    def check_crossover():
        """Check if crossover occurred on last closed candle"""
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=LIMIT)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            src = df['close'].values
            out, upper, lower = calculate_nwe(src, h, mult, repaint)
            
            close_series = pd.Series(src)
            upper_series = pd.Series(upper)
            crossover = detect_crossover(close_series, upper_series)
            
            return crossover.iloc[-2]  # Check second-to-last candle (last closed)
        except Exception as e:
            print(f"Error checking crossover: {e}")
            return False

    # 1. Check for open trade
    if has_open_position(symbol):
        position = session.get_positions(category="linear", symbol=get_pybit_symbol(symbol))
        if position["retCode"] == 0 and position["result"]["list"]:
            pos = position["result"]["list"][0]
            if float(pos["size"]) > 0:
                print(f"\nOpen {pos['side']} trade found:")
                print(f"Entry Price: {pos['avgPrice']}")
                print(f"Size: {pos['size']}")
                
                # 2. Handle open LONG trade
                if pos['side'] == 'Buy':
                    if check_crossover():
                        print("\nCrossover detected - closing long position immediately")
                        if close_position(symbol):
                            print("Long position closed successfully")
                        else:
                            print("Failed to close long position")
                    else:
                        print("\nNo crossover detected - keeping long position open")
                # 3. Handle open SHORT trade
                else:
                    print("\nOpen SHORT trade - doing nothing as per strategy")
        return
    
    print("\nNo open trades found")
    
    # 4. Check if last closed was SHORT trade
    if trade_info['side'] == "SHORT":
        # 5. Check if trend flipped TO UPTREND since short closed
        flip_detected, flip_message = detect_trend_flip(trade_info)
        
        if flip_detected is None:
            print(f"\n❌ Trend flip check error: {flip_message}")
            return
        elif flip_detected:
            print(f"\n❌ {flip_message} - Not entering SHORT (trend flipped)")
            return
        
        # 6. Check crossover condition
        if check_crossover():
            print("\nCrossover detected - preparing to enter SHORT")
            if cancel_all_pending_orders(symbol):
                print("Pending orders canceled - entering SHORT")
                place_market_order(symbol, "Sell")
            else:
                print("Failed to cancel pending orders")
        else:
            print("\nNo crossover detected - doing nothing")
    else:
        print("\nLast closed trade was not SHORT - doing nothing")
    
    print("\nCrossover strategy execution completed")

def band_touch_strategy(trade_info):
    """Band Touch Strategy with accurate position detection"""
    if not trade_info or not has_open_position(trade_info['symbol']):
        print("No open position - Band Touch strategy requires an open position")
        return
    
    symbol = trade_info['symbol']
    print(f"\n[BAND TOUCH STRATEGY] Checking {symbol} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Strategy settings
    TAKE_PROFIT_PCT = 5.0    # Close if profit exceeds this %
    TRAIL_SL_PCT = 0.1       # Trail stop-loss to this % profit
    MIN_PROFIT_PCT = 0.1     # Close if profit <= this % or loss
    h = 8.0
    mult = 3.0
    repaint = True
    
    def calculate_nwe_bands(src, h, mult, repaint):
        """Calculate Nadaraya-Watson Envelope bands"""
        n = len(src)
        upper = np.zeros(n)
        lower = np.zeros(n)
        
        if repaint:
            nwe = []
            sae = 0.0
            for i in range(n):
                sum_val = sumw = 0.0
                for j in range(n):
                    w = np.exp(-((i-j)**2)/(h**2*2))
                    sum_val += src[j] * w
                    sumw += w
                y2 = sum_val / sumw
                nwe.append(y2)
                sae += abs(src[i] - y2)
            sae = (sae / n) * mult
            upper = [nwe[i] + sae for i in range(n)]
            lower = [nwe[i] - sae for i in range(n)]
        else:
            coefs = np.array([np.exp(-(i**2)/(h**2*2)) for i in range(n)])
            den = np.sum(coefs)
            out = np.sum(src * coefs) / den
            mae = pd.Series(np.abs(src - out)).rolling(499).mean().values * mult
            upper = out + mae
            lower = out - mae
        
        return upper, lower

    def check_band_touch():
        """Detect if price touched bands"""
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=LIMIT)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            src = df['close'].values
            upper, lower = calculate_nwe_bands(src, h, mult, repaint)
            
            last_candle = df.iloc[-2]  # Last closed candle
            last_upper = upper[-2]
            last_lower = lower[-2]
            
            touched_upper = (last_candle['high'] >= last_upper) or (last_candle['close'] >= last_upper)
            touched_lower = (last_candle['low'] <= last_lower) or (last_candle['close'] <= last_lower)
            
            return touched_upper, touched_lower, last_upper, last_lower
            
        except Exception as e:
            print(f"Band check error: {e}")
            return False, False, None, None

    # Get current position details
    position = session.get_positions(category="linear", symbol=get_pybit_symbol(symbol))
    if position["retCode"] != 0 or not position["result"]["list"]:
        print("❌ Failed to get position details")
        return
    
    pos = position["result"]["list"][0]
    current_price = float(session.get_tickers(category="linear", symbol=get_pybit_symbol(symbol))["result"]["list"][0]["lastPrice"])
    entry_price = float(pos["avgPrice"])
    
    # Calculate current PnL
    if pos['side'] == 'Buy':
        pnl = (current_price - entry_price) / entry_price * 100
        position_side = "LONG"
    else:
        pnl = (entry_price - current_price) / entry_price * 100
        position_side = "SHORT"
    
    print(f"{position_side} position | PnL: {format_pnl(pnl)}")
    
    # Check band touches
    touched_upper, touched_lower, last_upper, last_lower = check_band_touch()
    print(f"Touched upper: {touched_upper} | Touched lower: {touched_lower}")
    
    # LONG POSITION RULES
    if position_side == "LONG" and touched_upper:
        if pnl > TAKE_PROFIT_PCT:
            print(f"Profit >{TAKE_PROFIT_PCT}% + Upper touch → Closing")
            close_position(symbol)
        elif pnl > MIN_PROFIT_PCT:
            print(f"Profit {MIN_PROFIT_PCT}-{TAKE_PROFIT_PCT}% + Upper touch → Trailing SL")
            # Implement trailing stop logic here
        else:
            print(f"Profit ≤{MIN_PROFIT_PCT}% + Upper touch → Closing")
            close_position(symbol)
    
    # SHORT POSITION RULES
    elif position_side == "SHORT" and touched_lower:
        if pnl > TAKE_PROFIT_PCT:
            print(f"Profit >{TAKE_PROFIT_PCT}% + Lower touch → Closing")
            close_position(symbol)
        elif pnl > MIN_PROFIT_PCT:
            print(f"Profit {MIN_PROFIT_PCT}-{TAKE_PROFIT_PCT}% + Lower touch → Trailing SL")
            # Implement trailing stop logic here
        else:
            print(f"Profit ≤{MIN_PROFIT_PCT}% + Lower touch → Closing")
            close_position(symbol)
    
    else:
        print("No band touch → Holding")

def crossunder_strategy(trade_info):
    """Crossunder Strategy with accurate trend detection"""
    if not trade_info:
        print("No trade info available for Crossunder strategy")
        return
    
    symbol = trade_info['symbol']
    print(f"\n[CROSSUNDER STRATEGY] Checking {symbol} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Strategy settings
    h = 8.0
    mult = 3.0
    repaint = True
    
    def calculate_nwe(src, h, mult, repaint):
        """Calculate Nadaraya-Watson Envelope"""
        n = len(src)
        out = np.zeros(n)
        upper = np.zeros(n)
        lower = np.zeros(n)
        
        if repaint:
            nwe = []
            sae = 0.0
            for i in range(n):
                sum_val = sumw = 0.0
                for j in range(n):
                    w = np.exp(-((i-j)**2)/(h**2*2))
                    sum_val += src[j] * w
                    sumw += w
                y2 = sum_val / sumw
                nwe.append(y2)
                sae += abs(src[i] - y2)
            sae = (sae / n) * mult
            upper = [nwe[i] + sae for i in range(n)]
            lower = [nwe[i] - sae for i in range(n)]
        else:
            coefs = np.array([np.exp(-(i**2)/(h**2*2)) for i in range(n)])
            den = np.sum(coefs)
            out = np.sum(src * coefs) / den
            mae = pd.Series(np.abs(src - out)).rolling(499).mean().values * mult
            upper = out + mae
            lower = out - mae
        
        return out, upper, lower

    def detect_crossunder(close, lower):
        """Detect if price has crossed below the lower envelope"""
        return (close.shift(1) > lower.shift(1)) & (close < lower)

    def check_crossunder():
        """Check if crossunder occurred on last closed candle"""
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=LIMIT)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            src = df['close'].values
            out, upper, lower = calculate_nwe(src, h, mult, repaint)
            
            close_series = pd.Series(src)
            lower_series = pd.Series(lower)
            crossunder = detect_crossunder(close_series, lower_series)
            
            return crossunder.iloc[-2]
        except Exception as e:
            print(f"Error checking crossunder: {e}")
            return False

    # 1. Check for open trade
    if has_open_position(symbol):
        position = session.get_positions(category="linear", symbol=get_pybit_symbol(symbol))
        if position["retCode"] == 0 and position["result"]["list"]:
            pos = position["result"]["list"][0]
            if float(pos["size"]) > 0:
                print(f"\nOpen {pos['side']} trade found:")
                print(f"Entry Price: {pos['avgPrice']}")
                print(f"Size: {pos['size']}")
                
                # 2. Handle open SHORT trade
                if pos['side'] == 'Sell':
                    if check_crossunder():
                        print("\nCrossunder detected - closing short position immediately")
                        if close_position(symbol):
                            print("Short position closed successfully")
                        else:
                            print("Failed to close short position")
                    else:
                        print("\nNo crossunder detected - keeping short position open")
                # 3. Handle open LONG trade
                else:
                    print("\nOpen LONG trade - doing nothing as per strategy")
        return
    
    print("\nNo open trades found")
    
    # 4. Check if last closed was LONG trade
    if trade_info['side'] == "LONG":
        # 5. Check if trend has flipped since trade was closed
        flip_detected, flip_message = detect_trend_flip(trade_info)
        
        if flip_detected is None:
            print(f"\n❌ Trend flip check error: {flip_message}")
            return
        elif flip_detected:
            print(f"\n❌ {flip_message} - Not entering LONG (trend flipped)")
            return
        
        # 6. Check crossunder condition
        if check_crossunder():
            print("\nCrossunder detected - preparing to enter LONG")
            if cancel_all_pending_orders(symbol):
                print("Pending orders canceled - entering LONG")
                place_market_order(symbol, "Buy")
            else:
                print("Failed to cancel pending orders")
        else:
            print("\nNo crossunder detected - doing nothing")
    else:
        print("\nLast closed trade was not LONG - doing nothing")
    
    print("\nCrossunder strategy execution completed")

# ======== MAIN EXECUTION ========
if __name__ == "__main__":
    print("\n=== TRADING BOT STARTED ===")
    print(f"Current Time: {datetime.now(LAGOS_TZ).strftime('%Y-%m-%d %H:%M:%S')} Lagos Time")
    
    # Analyze last trade to get symbol and position info
    trade_info = get_last_closed_trade()
    
    if trade_info:
        print("\n📊 Detected Last Trade:")
        print(f"Symbol: {trade_info['symbol']}")
        print(f"Direction: {trade_info['side']}")
        print(f"Closed at: {trade_info['close_time'].strftime('%Y-%m-%d %H:%M:%S')} Lagos Time")
        print(f"Close Price: {trade_info['close_price']}")
        print(f"Trend at Close: {trade_info['trend_at_close']}")
        
        # Run all strategies based on detected symbol
        atr_strategy(trade_info)
        crossover_strategy(trade_info)
        band_touch_strategy(trade_info)
        crossunder_strategy(trade_info)
        
    else:
        print("❌ No recent trade found to analyze")
    
    print("\n=== TRADING BOT COMPLETED ===")

































import ccxt
import pandas as pd
import pytz
from pybit.unified_trading import HTTP
import time
from datetime import datetime, timedelta
import logging
from twilio.rest import Client

# ===== Configuration =====
SYMBOLS = ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'XRP/USDT:USDT', 'EOS/USDT:USDT', 'BCH/USDT:USDT', 'LTC/USDT:USDT', 'ADA/USDT:USDT', 'ETC/USDT:USDT', 'LINK/USDT:USDT', 'TRX/USDT:USDT', 'DOT/USDT:USDT', 'DOGE/USDT:USDT', 'SOL/USDT:USDT', 'BNB/USDT:USDT', 'UNI/USDT:USDT', 'ICP/USDT:USDT', 'AAVE/USDT:USDT', 'FIL/USDT:USDT', 'XLM/USDT:USDT', 'ATOM/USDT:USDT', 'XTZ/USDT:USDT', 'SUSHI/USDT:USDT', 'AXS/USDT:USDT', 'THETA/USDT:USDT', 'AVAX/USDT:USDT', 'SHIB/USDT:USDT', 'MANA/USDT:USDT', 'GALA/USDT:USDT', 'SAND/USDT:USDT', 'DYDX/USDT:USDT', 'CRV/USDT:USDT', 'NEAR/USDT:USDT', 'EGLD/USDT:USDT', 'KSM/USDT:USDT', 'AR/USDT:USDT', 'PEOPLE/USDT:USDT', 'LRC/USDT:USDT', 'NEO/USDT:USDT', 'ALICE/USDT:USDT', 'WAVES/USDT:USDT', 'ALGO/USDT:USDT', 'IOTA/USDT:USDT', 'ENJ/USDT:USDT', 'GMT/USDT:USDT', 'ZIL/USDT:USDT', 'IOST/USDT:USDT', 'APE/USDT:USDT', 'RUNE/USDT:USDT', 'KNC/USDT:USDT', 'APT/USDT:USDT', 'CHZ/USDT:USDT', 'ROSE/USDT:USDT', 'ZRX/USDT:USDT', 'KAVA/USDT:USDT', 'ENS/USDT:USDT', 'MTL/USDT:USDT', 'AUDIO/USDT:USDT', 'SXP/USDT:USDT', 'C98/USDT:USDT', 'OP/USDT:USDT', 'RSR/USDT:USDT', 'SNX/USDT:USDT', 'STORJ/USDT:USDT', '1INCH/USDT:USDT', 'COMP/USDT:USDT', 'IMX/USDT:USDT', 'LUNA/USDT:USDT', 'FLOW/USDT:USDT', 'TRB/USDT:USDT', 'QTUM/USDT:USDT', 'API3/USDT:USDT', 'MASK/USDT:USDT', 'WOO/USDT:USDT', 'GRT/USDT:USDT', 'BAND/USDT:USDT', 'STG/USDT:USDT', 'LUNC/USDT:USDT', 'ONE/USDT:USDT', 'JASMY/USDT:USDT', 'MKR/USDT:USDT', 'BAT/USDT:USDT', 'MAGIC/USDT:USDT', 'ALPHA/USDT:USDT', 'LDO/USDT:USDT', 'CELO/USDT:USDT', 'BLUR/USDT:USDT', 'MINA/USDT:USDT', 'CORE/USDT:USDT', 'CFX/USDT:USDT', 'ASTR/USDT:USDT', 'GMX/USDT:USDT', 'ANKR/USDT:USDT', 'ACH/USDT:USDT', 'FET/USDT:USDT', 'FXS/USDT:USDT', 'HOOK/USDT:USDT', 'SSV/USDT:USDT', 'USDC/USDT:USDT', 'LQTY/USDT:USDT', 'STX/USDT:USDT', 'TRU/USDT:USDT', 'HBAR/USDT:USDT', 'INJ/USDT:USDT', 'BEL/USDT:USDT', 'COTI/USDT:USDT', 'VET/USDT:USDT', 'ARB/USDT:USDT', 'LOOKS/USDT:USDT', 'KAIA/USDT:USDT', 'FLM/USDT:USDT', 'CKB/USDT:USDT', 'ID/USDT:USDT', 'JOE/USDT:USDT', 'TLM/USDT:USDT', 'HOT/USDT:USDT', 'CHR/USDT:USDT', 'RDNT/USDT:USDT', 'ICX/USDT:USDT', 'ONT/USDT:USDT', 'NKN/USDT:USDT', 'ARPA/USDT:USDT', 'SFP/USDT:USDT', 'CTSI/USDT:USDT', 'SKL/USDT:USDT', 'RVN/USDT:USDT', 'CELR/USDT:USDT', 'FLOKI/USDT:USDT', 'SPELL/USDT:USDT', 'SUI/USDT:USDT', 'PEPE/USDT:USDT', 'IOTX/USDT:USDT', 'CTK/USDT:USDT', 'UMA/USDT:USDT', 'TURBO/USDT:USDT', 'BSV/USDT:USDT', 'TON/USDT:USDT', 'GTC/USDT:USDT', 'DENT/USDT:USDT', 'ZEN/USDT:USDT', 'PHB/USDT:USDT', 'ORDI/USDT:USDT', '1000BONK/USDT:USDT', 'LEVER/USDT:USDT', 'USTC/USDT:USDT', 'RAD/USDT:USDT', 'QNT/USDT:USDT', 'MAV/USDT:USDT', 'XVG/USDT:USDT', '1000XEC/USDT:USDT', 'AGLD/USDT:USDT', 'WLD/USDT:USDT', 'PENDLE/USDT:USDT', 'ARKM/USDT:USDT', 'CVX/USDT:USDT', 'YGG/USDT:USDT', 'OGN/USDT:USDT', 'LPT/USDT:USDT', 'BNT/USDT:USDT', 'SEI/USDT:USDT', 'CYBER/USDT:USDT', 'BAKE/USDT:USDT', 'BIGTIME/USDT:USDT', 'WAXP/USDT:USDT', 'POLYX/USDT:USDT', 'TIA/USDT:USDT', 'MEME/USDT:USDT', 'PYTH/USDT:USDT', 'JTO/USDT:USDT', '1000SATS/USDT:USDT', '1000RATS/USDT:USDT', 'ACE/USDT:USDT', 'XAI/USDT:USDT', 'MANTA/USDT:USDT', 'ALT/USDT:USDT', 'JUP/USDT:USDT', 'ZETA/USDT:USDT', 'STRK/USDT:USDT', 'PIXEL/USDT:USDT', 'DYM/USDT:USDT', 'WIF/USDT:USDT', 'AXL/USDT:USDT', 'BEAM/USDT:USDT', 'BOME/USDT:USDT', 'METIS/USDT:USDT', 'NFP/USDT:USDT', 'VANRY/USDT:USDT', 'AEVO/USDT:USDT', 'ETHFI/USDT:USDT', 'OM/USDT:USDT', 'ONDO/USDT:USDT', 'CAKE/USDT:USDT', 'PORTAL/USDT:USDT', 'NTRN/USDT:USDT', 'KAS/USDT:USDT', 'AI/USDT:USDT', 'ENA/USDT:USDT', 'W/USDT:USDT', 'CVC/USDT:USDT', 'TNSR/USDT:USDT', 'SAGA/USDT:USDT', 'TAO/USDT:USDT', 'RAY/USDT:USDT', 'ATA/USDT:USDT', 'SUPER/USDT:USDT', 'ONG/USDT:USDT', 'OMNI1/USDT:USDT', 'LSK/USDT:USDT', 'GLM/USDT:USDT', 'REZ/USDT:USDT', 'XVS/USDT:USDT', 'MOVR/USDT:USDT', 'BB/USDT:USDT', 'NOT/USDT:USDT', 'BICO/USDT:USDT', 'HIFI/USDT:USDT', 'IO/USDT:USDT', 'TAIKO/USDT:USDT', 'BRETT/USDT:USDT', 'ATH/USDT:USDT', 'ZK/USDT:USDT', 'MEW/USDT:USDT', 'LISTA/USDT:USDT', 'ZRO/USDT:USDT', 'BLAST/USDT:USDT', 'DOG/USDT:USDT', 'PAXG/USDT:USDT', 'ZKJ/USDT:USDT', 'BGB/USDT:USDT', 'MOCA/USDT:USDT', 'GAS/USDT:USDT', 'UXLINK/USDT:USDT', 'BANANA/USDT:USDT', 'MYRO/USDT:USDT', 'POPCAT/USDT:USDT', 'PRCL/USDT:USDT', 'CLOUD/USDT:USDT', 'AVAIL/USDT:USDT', 'RENDER/USDT:USDT', 'RARE/USDT:USDT', 'PONKE/USDT:USDT', 'T/USDT:USDT', '1000000MOG/USDT:USDT', 'G/USDT:USDT', 'SYN/USDT:USDT', 'SYS/USDT:USDT', 'VOXEL/USDT:USDT', 'SUN/USDT:USDT', 'DOGS/USDT:USDT', 'ORDER/USDT:USDT', 'SUNDOG/USDT:USDT', 'AKT/USDT:USDT', 'MBOX/USDT:USDT', 'HNT/USDT:USDT', 'CHESS/USDT:USDT', 'FLUX/USDT:USDT', 'POL/USDT:USDT', 'BSW/USDT:USDT', 'NEIROETH/USDT:USDT', 'RPL/USDT:USDT', 'QUICK/USDT:USDT', 'AERGO/USDT:USDT', '1MBABYDOGE/USDT:USDT', '1000CAT/USDT:USDT', 'KDA/USDT:USDT', 'FIDA/USDT:USDT', 'CATI/USDT:USDT', 'FIO/USDT:USDT', 'ARK/USDT:USDT', 'GHST/USDT:USDT', 'LOKA/USDT:USDT', 'VELO/USDT:USDT', 'HMSTR/USDT:USDT', 'AGI/USDT:USDT', 'REI/USDT:USDT', 'COS/USDT:USDT', 'EIGEN/USDT:USDT', 'MOODENG/USDT:USDT', 'DIA/USDT:USDT', 'FTN/USDT:USDT', 'OG/USDT:USDT', 'NEIROCTO/USDT:USDT', 'ETHW/USDT:USDT', 'DegenReborn/USDT:USDT', 'KMNO/USDT:USDT', 'POWR/USDT:USDT', 'PYR/USDT:USDT', 'CARV/USDT:USDT', 'SLERF/USDT:USDT', 'PUFFER/USDT:USDT', '10000WHY/USDT:USDT', 'DEEP/USDT:USDT', 'DBR/USDT:USDT', 'LUMIA/USDT:USDT', 'SCR/USDT:USDT', 'GOAT/USDT:USDT', 'X/USDT:USDT', 'SAFE/USDT:USDT', 'GRASS/USDT:USDT', 'SWEAT/USDT:USDT', 'SANTOS/USDT:USDT', 'SPX/USDT:USDT', 'VIRTUAL/USDT:USDT', 'AERO/USDT:USDT', 'CETUS/USDT:USDT', 'COW/USDT:USDT', 'SWELL/USDT:USDT', 'DRIFT/USDT:USDT', 'PNUT/USDT:USDT', 'ACT/USDT:USDT', 'CRO/USDT:USDT', 'PEAQ/USDT:USDT', 'FOXY/USDT:USDT', 'FWOG/USDT:USDT', 'HIPPO/USDT:USDT', 'SNT/USDT:USDT', 'MERL/USDT:USDT', 'STEEM/USDT:USDT', 'BAN/USDT:USDT', 'OL/USDT:USDT', 'MORPHO/USDT:USDT', 'SCRT/USDT:USDT', 'CHILLGUY/USDT:USDT', 'MEMEFI/USDT:USDT', '1MCHEEMS/USDT:USDT', 'OXT/USDT:USDT', 'ZRC/USDT:USDT', 'THE/USDT:USDT', 'MAJOR/USDT:USDT', 'CTC/USDT:USDT', 'XDC/USDT:USDT', 'XION/USDT:USDT', 'ORCA/USDT:USDT', 'ACX/USDT:USDT', 'NS/USDT:USDT', 'MOVE/USDT:USDT', 'KOMA/USDT:USDT', 'ME/USDT:USDT', 'VELODROME/USDT:USDT', 'AVA/USDT:USDT', 'VANA/USDT:USDT', 'HYPE/USDT:USDT', 'PENGU/USDT:USDT', 'USUAL/USDT:USDT', 'FUEL/USDT:USDT', 'CGPT/USDT:USDT', 'AIXBT/USDT:USDT', 'FARTCOIN/USDT:USDT', 'HIVE/USDT:USDT', 'DEXE/USDT:USDT', 'GIGA/USDT:USDT', 'PHA/USDT:USDT', 'DF/USDT:USDT', 'AI16Z/USDT:USDT', 'GRIFFAIN/USDT:USDT', 'ZEREBRO/USDT:USDT', 'BIO/USDT:USDT', 'SWARMS/USDT:USDT', 'ALCH/USDT:USDT', 'COOKIE/USDT:USDT', 'SONIC/USDT:USDT', 'AVAAI/USDT:USDT', 'S/USDT:USDT', 'PROM/USDT:USDT', 'DUCK/USDT:USDT', 'BGSC/USDT:USDT', 'SOLV/USDT:USDT', 'ARC/USDT:USDT', 'NC/USDT:USDT', 'PIPPIN/USDT:USDT', 'TRUMP/USDT:USDT', 'MELANIA/USDT:USDT', 'PLUME/USDT:USDT', 'VTHO/USDT:USDT', 'J/USDT:USDT', 'VINE/USDT:USDT', 'ANIME/USDT:USDT', 'STPT/USDT:USDT', 'XCN/USDT:USDT', 'TOSHI/USDT:USDT', 'VVV/USDT:USDT', 'FORTH/USDT:USDT', 'BERA/USDT:USDT', 'TSTBSC/USDT:USDT', '10000ELON/USDT:USDT', 'LAYER/USDT:USDT', 'B3/USDT:USDT', 'IP/USDT:USDT', 'RON/USDT:USDT', 'HEI/USDT:USDT', 'SHELL/USDT:USDT', 'BROCCOLI/USDT:USDT', 'AUCTION/USDT:USDT', 'GPS/USDT:USDT', 'GNO/USDT:USDT', 'AIOZ/USDT:USDT', 'PI/USDT:USDT', 'AVL/USDT:USDT', 'KAITO/USDT:USDT', 'GODS/USDT:USDT', 'ROAM/USDT:USDT', 'RED/USDT:USDT', 'ELX/USDT:USDT', 'SERAPH/USDT:USDT', 'BMT/USDT:USDT', 'VIC/USDT:USDT', 'EPIC/USDT:USDT', 'OBT/USDT:USDT', 'MUBARAK/USDT:USDT', 'NMR/USDT:USDT', 'TUT/USDT:USDT', 'FORM/USDT:USDT', 'RSS3/USDT:USDT', 'BID/USDT:USDT', 'SIREN/USDT:USDT', 'BROCCOLIF3B/USDT:USDT', 'BANANAS31/USDT:USDT', 'BR/USDT:USDT', 'NIL/USDT:USDT', 'PARTI/USDT:USDT', 'NAVX/USDT:USDT', 'WAL/USDT:USDT', 'KILO/USDT:USDT', 'FUN/USDT:USDT', 'MLN/USDT:USDT', 'GUN/USDT:USDT', 'PUMP/USDT:USDT', 'STO/USDT:USDT', 'XAUT/USDT:USDT', 'AMP/USDT:USDT', 'BABY/USDT:USDT', 'FHE/USDT:USDT', 'PROMPT/USDT:USDT', 'RFC/USDT:USDT', 'KERNEL/USDT:USDT', 'WCT/USDT:USDT', 'PAWS/USDT:USDT', '10000000AIDOGE/USDT:USDT', 'BANK/USDT:USDT', 'EPT/USDT:USDT', 'HYPER/USDT:USDT', 'ZORA/USDT:USDT', 'INIT/USDT:USDT', 'DOLO/USDT:USDT', 'FIS/USDT:USDT', 'DARK/USDT:USDT', 'JST/USDT:USDT', 'TAI/USDT:USDT', 'SIGN/USDT:USDT', 'MILK/USDT:USDT', 'HAEDAL/USDT:USDT', 'PUNDIX/USDT:USDT', 'B2/USDT:USDT', 'AIOT/USDT:USDT', 'GORK/USDT:USDT', 'HOUSE/USDT:USDT', 'ASR/USDT:USDT', 'ALPINE/USDT:USDT', 'SYRUP/USDT:USDT', 'OBOL/USDT:USDT', 'SXT/USDT:USDT', 'DOOD/USDT:USDT']
TRADE_AMOUNT_USDT = 50          # Position size in USDT
STOPLOSS_PERCENT = 2            # 2% stop-loss
TAKEPROFIT_PERCENT = 7.5        # 7.5% take-profit
MAX_CONCURRENT_TRADES = 1       # Only 1 trade at a time by default

# Twilio WhatsApp Configuration
TWILIO_ACCOUNT_SID = 'AC5ab4c9778fff846bd300ac9fd0f16656'  # Replace with your SID
TWILIO_AUTH_TOKEN = 'e1742ac942a79545b3c026066fba2425'     # Replace with your token
TWILIO_WHATSAPP_NUMBER = 'whatsapp:+14155238886'  # Twilio's WhatsApp sandbox number
YOUR_WHATSAPP_NUMBER = 'whatsapp:+2349060558418'  # Your WhatsApp number (with country code)

# Strategy Parameters
EMA_FAST = 38
EMA_SLOW = 62
EMA_TREND = 200
TIMEFRAME = '15m'

# Timezone Configuration
LAGOS_TZ = pytz.timezone('Africa/Lagos')
UTC_TZ = pytz.UTC

# ===== Initialize Connections =====
bybit = HTTP(
    api_key="lJu52hbBTbPkg2VXZ2",
    api_secret="e43RV6YDZsn24Q9mucr0i4xbU7YytdL2HtuV",
    demo=True
)

bitget = ccxt.bitget({
    'enableRateLimit': True
})

# Initialize Twilio client
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# ===== Utility Functions =====
def send_whatsapp_notification(message):
    """Send WhatsApp notifications via Twilio"""
    try:
        msg = twilio_client.messages.create(
            from_=TWILIO_WHATSAPP_NUMBER,
            body=message,
            to=YOUR_WHATSAPP_NUMBER
        )
        logging.info(f"WhatsApp notification sent: {msg.sid}")
    except Exception as e:
        logging.error(f"WhatsApp notification failed: {str(e)}")

def has_open_positions():
    """Check for any open positions with proper API parameters"""
    try:
        # Get positions for all symbols
        positions = bybit.get_positions(
            category="linear",
            settleCoin="USDT"
        ).get("result", {}).get("list", [])
        
        return any(float(pos["size"]) > 0 for pos in positions)
    except Exception as e:
        logging.error(f"Position check error: {str(e)}")
        return True  # Default to blocking if error occurs

def has_pending_orders():
    """Check for any pending orders with proper API parameters"""
    try:
        # Get open orders for all symbols
        orders = bybit.get_open_orders(
            category="linear",
            settleCoin="USDT"
        ).get("result", {}).get("list", [])
        
        return len(orders) > 0
    except Exception as e:
        logging.error(f"Order check error: {str(e)}")
        return True  # Default to blocking

# ===== Enhanced Trend Detection Functions =====
def fetch_market_data(symbol, timeframe, limit=500):
    """Fetch OHLCV data with proper timezone handling"""
    try:
        ohlcv = bitget.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert(LAGOS_TZ)
        df.set_index('timestamp', inplace=True)
        df['EMA_Trend'] = df['close'].ewm(span=EMA_TREND, adjust=False).mean()
        df['EMA_Fast'] = df['close'].ewm(span=EMA_FAST, adjust=False).mean()
        df['EMA_Slow'] = df['close'].ewm(span=EMA_SLOW, adjust=False).mean()
        return df
    except Exception as e:
        logging.error(f"Error fetching market data for {symbol}: {str(e)}")
        return None

def detect_trend(df, candle_index):
    """Determine trend for a specific candle using EMA analysis"""
    if candle_index < 1 or candle_index >= len(df):
        return "Sideways"
    
    ema_current = df['EMA_Trend'].iloc[candle_index]
    ema_prev = df['EMA_Trend'].iloc[candle_index-1]
    price = df['close'].iloc[candle_index]
    
    if ema_current > ema_prev and price > ema_current:
        return "Uptrend"
    elif ema_current < ema_prev and price < ema_current:
        return "Downtrend"
    return "Sideways"

def get_last_closed_trade():
    """Get details of the most recent closed trade with improved accuracy"""
    try:
        # Get trades with proper parameters
        trades = bybit.get_closed_pnl(
            category="linear",
            settleCoin="USDT",
            limit=50
        ).get("result", {}).get("list", [])
        
        # Get current positions with proper parameters
        positions = bybit.get_positions(
            category="linear",
            settleCoin="USDT"
        ).get("result", {}).get("list", [])
        
        open_symbols = [pos['symbol'] for pos in positions if float(pos['size']) > 0]
        
        for trade in sorted(trades, key=lambda x: int(x["updatedTime"]), reverse=True):
            if trade['symbol'] not in open_symbols and float(trade['closedSize']) > 0:
                utc_time = datetime.fromtimestamp(int(trade["updatedTime"])/1000, UTC_TZ)
                lagos_time = utc_time.astimezone(LAGOS_TZ)
                
                return {
                    "symbol": f"{trade['symbol'].replace('USDT', '')}/USDT:USDT",
                    "close_time": lagos_time,
                    "close_price": float(trade["avgEntryPrice"]),
                    "side": "LONG" if trade["side"] == "Sell" else "SHORT",
                    "utc_close_time": utc_time
                }
        return None
    except Exception as e:
        logging.error(f"Error fetching trade history: {str(e)}")
        return None

def check_trend_flip(symbol, since_time, last_side):
    """Enhanced trend flip detection with counter-trend check"""
    try:
        df = fetch_market_data(symbol, TIMEFRAME, 500)
        if df is None or len(df) < 2:
            return False, "No data"
            
        # Find the candle where trade was closed
        close_candle_idx = df.index.get_indexer([since_time], method='nearest')[0]
        if close_candle_idx < 1:  # Need at least 2 candles for trend detection
            close_candle_idx = 1
        
        # Get trend at close time
        trend_at_close = detect_trend(df, close_candle_idx)
        
        # Check for counter-trend closing
        if (last_side == "SHORT" and trend_at_close == "Uptrend") or \
           (last_side == "LONG" and trend_at_close == "Downtrend"):
            return False, "Counter-trend closing detected"
        
        # Analyze trend changes since closing
        current_trend = trend_at_close
        first_flip = None
        
        for i in range(close_candle_idx + 1, len(df)):
            new_trend = detect_trend(df, i)
            
            if new_trend != current_trend and new_trend in ["Uptrend", "Downtrend"]:
                flip_time = df.index[i]
                duration = flip_time - since_time
                return True, (f"Trend flipped to {new_trend} at {flip_time.strftime('%Y-%m-%d %H:%M:%S')} "
                             f"({duration.total_seconds()/3600:.1f} hours after close)")
        
        return False, "No trend flip detected"
    except Exception as e:
        logging.error(f"Trend flip check error for {symbol}: {str(e)}")
        return False, "Error"

def check_for_pullback_signal(symbol):
    """Improved signal detection with trend confirmation"""
    try:
        df = fetch_market_data(symbol, TIMEFRAME)
        if df is None or len(df) < 3:
            return None
            
        last_candle = df.iloc[-2]
        prev_candle = df.iloc[-3]
        
        # Current market trend
        current_trend = detect_trend(df, -1)
        
        # Check long signal (only in uptrend)
        long_condition = (
            current_trend == "Uptrend" and
            last_candle['EMA_Fast'] > last_candle['EMA_Slow'] and
            prev_candle['EMA_Fast'] <= prev_candle['EMA_Slow'] and
            last_candle['close'] > last_candle['EMA_Trend']
        )
        
        # Check short signal (only in downtrend)
        short_condition = (
            current_trend == "Downtrend" and
            last_candle['EMA_Fast'] < last_candle['EMA_Slow'] and
            prev_candle['EMA_Fast'] >= prev_candle['EMA_Slow'] and
            last_candle['close'] < last_candle['EMA_Trend']
        )
        
        return "buy" if long_condition else "sell" if short_condition else None
    except Exception as e:
        logging.error(f"Signal check error for {symbol}: {str(e)}")
        return None

def place_trade_order(symbol, signal, price):
    """Execute trade with risk management"""
    try:
        bybit_symbol = symbol.replace('/USDT:USDT', 'USDT')
        qty = round((TRADE_AMOUNT_USDT / price) / 0.1) * 0.1  # Adjust for lot size
        
        sl_price = round(price * (0.98 if signal == "buy" else 1.02), 4)
        tp_price = round(price * (1.075 if signal == "buy" else 0.925), 4)
        
        order = bybit.place_order(
            category="linear",
            symbol=bybit_symbol,
            side="Buy" if signal == "buy" else "Sell",
            orderType="Market",
            qty=str(qty),
            takeProfit=str(tp_price),
            stopLoss=str(sl_price),
            timeInForce="GTC"
        )
        
        if order['retCode'] == 0:
            msg = f"⚡ Trade Executed ⚡\n\n{symbol}\nAction: {signal.upper()}\nPrice: {price}\nSL: {sl_price}\nTP: {tp_price}"
            send_whatsapp_notification(msg)
            return True
        else:
            raise Exception(order['retMsg'])
    except Exception as e:
        error_msg = f"❌ Trade Failed ❌\n\n{symbol}\nAction: {signal.upper()}\nError: {str(e)}"
        logging.error(error_msg)
        send_whatsapp_notification(error_msg)
        return False

# ===== Main Execution =====
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    startup_msg = f"\n{'='*50}\nEnhanced Trading Bot\n{datetime.now(LAGOS_TZ).strftime('%Y-%m-%d %H:%M:%S')}\n{'='*50}"
    print(startup_msg)
    send_whatsapp_notification(startup_msg)

    try:
        # 1. Check for blocking conditions
        if has_open_positions():
            msg = "🛑 Blocked: Open positions exist"
            print(msg)
            send_whatsapp_notification(msg)
            exit()
            
        if has_pending_orders():
            msg = "🛑 Blocked: Pending orders exist"
            print(msg)
            send_whatsapp_notification(msg)
            exit()
            
        # 2. Get last trade and check trend flip
        last_trade = get_last_closed_trade()
        check_all_symbols = False
        
        if last_trade:
            trade_msg = f"🔍 Last Trade: {last_trade['symbol']} ({last_trade['side']})\nClosed at: {last_trade['close_time'].strftime('%Y-%m-%d %H:%M:%S')}"
            print(trade_msg)
            
            flipped, msg = check_trend_flip(
                last_trade['symbol'],
                last_trade['close_time'],
                last_trade['side']
            )
            
            if flipped:
                flip_msg = f"🔀 Trend Flip Detected: {msg}"
                print(flip_msg)
                send_whatsapp_notification(flip_msg)
                check_all_symbols = True
            else:
                no_flip_msg = f"⏳ {msg} - Only monitoring last traded symbol"
                print(no_flip_msg)
                signal = check_for_pullback_signal(last_trade['symbol'])
                if signal:
                    price = bitget.fetch_ticker(last_trade['symbol'])['last']
                    signal_msg = f"⚡ Signal Found: {signal.upper()} {last_trade['symbol']} at {price}"
                    print(signal_msg)
                    place_trade_order(last_trade['symbol'], signal, price)
                else:
                    print("❌ No valid signal on last traded symbol")
                exit()
        else:
            no_history_msg = "📜 No trade history found - Checking all symbols"
            print(no_history_msg)
            check_all_symbols = True
            
        # 3. Scan all symbols if conditions met
        if check_all_symbols:
            print("\n🔎 Scanning all symbols for signals...")
            traded_count = 0
            for symbol in SYMBOLS:
                if traded_count >= MAX_CONCURRENT_TRADES:
                    break
                    
                signal = check_for_pullback_signal(symbol)
                if signal:
                    price = bitget.fetch_ticker(symbol)['last']
                    signal_msg = f"⚡ Signal Found: {signal.upper()} {symbol} at {price}"
                    print(signal_msg)
                    if place_trade_order(symbol, signal, price):
                        traded_count += 1
                else:
                    print(f"➖ No signal on {symbol}")
                
                time.sleep(0)  # Rate limit
                
        completion_msg = f"\n✅ Run completed. Trades executed: {traded_count}"
        print(completion_msg)
        send_whatsapp_notification(completion_msg)
        
    except Exception as e:
        error_msg = f"❌ CRITICAL ERROR: {str(e)}"
        logging.error(error_msg)
        print(error_msg)
        send_whatsapp_notification(error_msg)