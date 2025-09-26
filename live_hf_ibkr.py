from ib_insync import IB, Stock, Option, MarketOrder, util
import pandas as pd
import numpy as np
import time
from datetime import datetime
from ml_model import compute_features, train_lightgbm
from signals import calculate_barrier_metrics, calculate_correlations
from volatility import VolatilityFeatures

# -------------------- CONFIG --------------------
STOCK_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN"]  # add as many as needed
OPTION_EXPIRY = "2025-12-20"  # YYYY-MM-DD
OPTION_RIGHT = "C"             # 'C' = Call, 'P' = Put
LOT_SIZE = 1                   # contracts per signal
SLEEP_SEC = 60
HORIZON_DAYS = 1
TRAIN_END_DATE = "2025-09-20"
CORR_THRESHOLD = 0.8           # correlation filter

# -------------------- IBKR INIT --------------------
ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)  # TWS or IB Gateway

# -------------------- FETCH HISTORICAL STOCK DATA --------------------
def fetch_ibkr_stock(symbol, duration="60 D", bar_size="1 day"):
    contract = Stock(symbol, 'SMART', 'USD')
    bars = ib.reqHistoricalData(contract, endDateTime='', durationStr=duration,
                                barSizeSetting=bar_size, whatToShow='TRADES',
                                useRTH=True)
    df = util.df(bars)
    df.set_index('date', inplace=True)
    return df['close']

price_data = pd.DataFrame({s: fetch_ibkr_stock(s) for s in STOCK_TICKERS})
returns_data = price_data.pct_change().dropna()

# -------------------- TRAIN ML MODEL --------------------
X, y = compute_features(price_data, horizon_days=HORIZON_DAYS, clip=0.02)
model = train_lightgbm(X, y, train_end_date=TRAIN_END_DATE)

# -------------------- LIVE TRADING LOOP --------------------
positions = {s: 0 for s in STOCK_TICKERS}
option_positions = {s: 0 for s in STOCK_TICKERS}
cash = 100000

print("Starting live IBKR stocks + options HF trading with correlation filter...")

while True:
    try:
        # Fetch latest prices
        live_prices = {}
        for s in STOCK_TICKERS:
            df = fetch_ibkr_stock(s, duration="1 D", bar_size="1 min")
            live_prices[s] = df.iloc[-1]

        # Generate signals (barrier + ML)
        signals = {}
        for s in STOCK_TICKERS:
            barrier_df = calculate_barrier_metrics(price_data[s])
            barrier_signal = barrier_df.iloc[-1]['signal']
            X_live = X.xs(s, level=1).iloc[-1:]
            ml_signal = "BUY" if model.predict(X_live)[0] > 0 else "SELL"
            signals[s] = barrier_signal if barrier_signal == ml_signal else "HOLD"

        # -------------------- Apply correlation filter --------------------
        correlations = calculate_correlations(returns_data)
        latest_signals = signals.copy()
        for (s1, s2), corr in correlations.items():
            if corr > CORR_THRESHOLD:
                # if both signals are BUY, keep only one
                if latest_signals[s1] == latest_signals[s2] == "BUY":
                    latest_signals[s2] = "HOLD"

        # Execute trades
        for s, sig in latest_signals.items():
            stock_price = live_prices[s]
            stock_contract = Stock(s, 'SMART', 'USD')
            strike_price = round(stock_price)
            option_contract = Option(s, OPTION_EXPIRY, strike_price, OPTION_RIGHT, 'SMART')

            if sig == "BUY":
                ib.placeOrder(stock_contract, MarketOrder('BUY', LOT_SIZE))
                ib.placeOrder(option_contract, MarketOrder('BUY', LOT_SIZE))
                positions[s] += LOT_SIZE
                option_positions[s] += LOT_SIZE
            elif sig == "SELL":
                if positions[s] > 0:
                    ib.placeOrder(stock_contract, MarketOrder('SELL', LOT_SIZE))
                    positions[s] -= LOT_SIZE
                if option_positions[s] > 0:
                    ib.placeOrder(option_contract, MarketOrder('SELL', LOT_SIZE))
                    option_positions[s] -= LOT_SIZE

        print(datetime.now(), latest_signals)
        time.sleep(SLEEP_SEC)

    except KeyboardInterrupt:
        print("Stopping live IBKR trading...")
        break
    except Exception as e:
        print("Error:", e)
        time.sleep(SLEEP_SEC)

ib.disconnect()
