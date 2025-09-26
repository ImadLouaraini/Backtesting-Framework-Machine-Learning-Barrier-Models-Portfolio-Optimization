import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
from datetime import datetime
from ml_model import compute_features, train_lightgbm
from signals import calculate_barrier_metrics, calculate_correlations
from volatility import VolatilityFeatures
from money_management_mt5 import MoneyManagerMT5

# -------------------- CONFIG --------------------
TICKERS = ["EURUSD","GBPUSD","USDJPY"]
SLEEP_SEC = 60
HORIZON_DAYS = 1
TRAIN_END_DATE = "2025-09-20"
CORR_THRESHOLD = 0.8
RISK_PER_TRADE = 0.01
DEFAULT_LOT = 0.1
SL_PCT = 0.002
TP_PCT = 0.004

# -------------------- MT5 INIT --------------------
if not mt5.initialize():
    print("MT5 initialization failed")
    mt5.shutdown()
    exit()

# -------------------- MONEY MANAGEMENT --------------------
mm = MoneyManagerMT5(account_size=100000, risk_per_trade=RISK_PER_TRADE, default_lot=DEFAULT_LOT)

# -------------------- FETCH HISTORICAL DATA --------------------
def fetch_mt5_data(symbol, n=500):
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, n)
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df['close']

price_data = pd.DataFrame({sym: fetch_mt5_data(sym) for sym in TICKERS})
returns_data = price_data.pct_change().dropna()

# -------------------- TRAIN ML MODEL --------------------
X, y = compute_features(price_data, horizon_days=HORIZON_DAYS, clip=0.002)
model = train_lightgbm(X, y, train_end_date=TRAIN_END_DATE)

# -------------------- LIVE LOOP --------------------
positions = {t: 0 for t in TICKERS}

print("Starting live MT5 HF trading with money management & correlation filter...")
while True:
    try:
        live_prices = pd.Series({t: mt5.symbol_info_tick(t).ask for t in TICKERS})

        # Generate signals
        signals = {}
        for t in TICKERS:
            barrier_df = calculate_barrier_metrics(price_data[t])
            barrier_signal = barrier_df.iloc[-1]['signal']
            X_live = X.xs(t, level=1).iloc[-1:]
            ml_signal = "BUY" if model.predict(X_live)[0] > 0 else "SELL"
            signals[t] = barrier_signal if barrier_signal == ml_signal else "HOLD"

        # -------------------- Apply correlation filter --------------------
        correlations = calculate_correlations(returns_data)
        latest_signals = signals.copy()
        for (s1, s2), corr in correlations.items():
            if corr > CORR_THRESHOLD:
                if latest_signals[s1] == latest_signals[s2] == "BUY":
                    latest_signals[s2] = "HOLD"

        # -------------------- Execute trades --------------------
        for t, sig in latest_signals.items():
            price = live_prices[t]
            sl, tp = mm.get_sl_tp(price, sig, sl_pct=SL_PCT, tp_pct=TP_PCT)
            lot = mm.calculate_lot(price, sl)

            if sig == "BUY":
                mt5.order_send(symbol=t, action=mt5.ORDER_TYPE_BUY, volume=lot, price=price, sl=sl, tp=tp)
                positions[t] += lot
            elif sig == "SELL" and positions[t] > 0:
                mt5.order_send(symbol=t, action=mt5.ORDER_TYPE_SELL, volume=lot, price=price, sl=sl, tp=tp)
                positions[t] -= lot

        print(datetime.now(), latest_signals)
        time.sleep(SLEEP_SEC)

    except KeyboardInterrupt:
        print("Stopping live MT5 trading...")
        break
    except Exception as e:
        print("Error:", e)
        time.sleep(SLEEP_SEC)

mt5.shutdown()
