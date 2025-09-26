import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
from datetime import datetime
from ml_model import compute_features, train_lightgbm
from signals import calculate_barrier_metrics, portfolio_signals
from volatility import VolatilityFeatures

# -------------------- CONFIG --------------------
TICKERS = ["EURUSD", "GBPUSD", "USDJPY"]
LOT_SIZE = 0.1
SLEEP_SEC = 60   # check every 60 seconds
HORIZON_DAYS = 1
TRANSACTION_COST = 0.0005
TRAIN_END_DATE = "2025-09-20"

# -------------------- MT5 INIT --------------------
if not mt5.initialize():
    print("MT5 initialization failed")
    mt5.shutdown()
    exit()

# -------------------- FETCH HISTORICAL DATA --------------------
def fetch_mt5_data(symbol, n=500):
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, n)
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    df = df['close']
    return df

price_data = pd.DataFrame({sym: fetch_mt5_data(sym) for sym in TICKERS})

# -------------------- TRAIN ML MODEL --------------------
X, y = compute_features(price_data, horizon_days=HORIZON_DAYS, clip=0.002)
model = train_lightgbm(X, y, train_end_date=TRAIN_END_DATE)

# -------------------- LIVE LOOP --------------------
positions = {t: 0 for t in TICKERS}
cash = 100000

print("Starting live MT5 HF trading...")
while True:
    try:
        live_prices = pd.Series({t: mt5.symbol_info_tick(t).ask for t in TICKERS})
        
        signals = {}
        for t in TICKERS:
            # barrier signals
            barrier_df = calculate_barrier_metrics(price_data[t])
            barrier_signal = barrier_df.iloc[-1]['signal']
            # ML signal
            X_live = X.xs(t, level=1).iloc[-1:]
            ml_signal = "BUY" if model.predict(X_live)[0]>0 else "SELL"
            # combined
            signals[t] = barrier_signal if barrier_signal == ml_signal else "HOLD"
        
        # execute trades
        for t, sig in signals.items():
            price = live_prices[t]
            if sig=="BUY":
                mt5.order_send(symbol=t, action=mt5.ORDER_TYPE_BUY, volume=LOT_SIZE, price=price)
                positions[t] += LOT_SIZE
            elif sig=="SELL" and positions[t]>0:
                mt5.order_send(symbol=t, action=mt5.ORDER_TYPE_SELL, volume=LOT_SIZE, price=price)
                positions[t] -= LOT_SIZE
        
        print(datetime.now(), signals)
        time.sleep(SLEEP_SEC)
    except KeyboardInterrupt:
        print("Stopping live MT5 trading")
        break
    except Exception as e:
        print("Error:", e)
        time.sleep(SLEEP_SEC)

mt5.shutdown()
