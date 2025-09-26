import pandas as pd
import numpy as np
from .data import fetch_multiple_stocks
from .signals import calculate_barrier_metrics, portfolio_signals
from .volatility import VolatilityFeatures
from .ml_model import compute_features, train_lightgbm

class BacktestEngineHF:
    def __init__(self, initial_capital=100000, transaction_cost=0.0005):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.results = {}

    def run(self, tickers, start_date, end_date, train_end_date, freq="1H"):
        """
        freq: '1H' = hourly, '1T' = minute, '1D' = daily
        """
        # 1️⃣ Fetch intraday data
        prices = fetch_multiple_stocks(tickers, start_date, end_date)
        prices = prices.asfreq(freq).ffill()  # resample to desired frequency

        returns = prices.pct_change().dropna()
        barrier_results = {}
        vol_features = pd.DataFrame(index=prices.index)

        # 2️⃣ Volatilité et barrier signals
        for ticker in tickers:
            df = prices[[ticker]].copy()
            df["realized_vol"] = VolatilityFeatures.realized_volatility(df[ticker].pct_change())
            df["garch_vol"] = df[ticker].pct_change().rolling(21).std()  # simple GARCH proxy
            barrier_df = calculate_barrier_metrics(df[ticker])
            barrier_results[ticker] = barrier_df
            vol_features = pd.concat([vol_features, df[["realized_vol","garch_vol"]]], axis=1)

        # 3️⃣ Features ML et LightGBM
        X, y = compute_features(prices, horizon_days=1, clip=0.02)  # horizon = 1 period for HF
        model = train_lightgbm(X, y, train_end_date=train_end_date, n_estimators=200, lr=0.05)

        # 4️⃣ Predict HF returns
        ml_signals = {}
        for ticker in tickers:
            X_ticker = X.xs(ticker, level=1)
            y_pred = model.predict(X_ticker)
            ml_signals[ticker] = pd.Series(y_pred, index=X_ticker.index).apply(
                lambda x: "BUY" if x > 0 else "SELL"
            )

        # 5️⃣ Combine ML + barrier signals
        combined_signals = {}
        for ticker in tickers:
            barrier_df = barrier_results[ticker].set_index("date")
            ml_df = ml_signals[ticker].to_frame(name="ml_signal")
            combined = pd.concat([barrier_df["signal"], ml_df], axis=1, join="inner")
            combined["signal"] = np.where(combined["signal"]==combined["ml_signal"], combined["signal"], "HOLD")
            combined_signals[ticker] = combined

        portfolio_sig = portfolio_signals(combined_signals, returns)

        # 6️⃣ High-frequency backtesting
        portfolio_values = []
        benchmark_values = []
        cash = self.initial_capital
        positions = {t: 0 for t in tickers}

        for date in prices.index[1:]:
            daily_prices = prices.loc[date]
            signals = {t: combined_signals[t].loc[date, "signal"] if date in combined_signals[t].index else "HOLD"
                       for t in tickers}

            # HF allocation: 1 unit per signal
            for t, sig in signals.items():
                price = daily_prices[t]
                if sig == "BUY" and cash >= price:
                    positions[t] += 1
                    cash -= price * (1 + self.transaction_cost)
                elif sig == "SELL" and positions[t] > 0:
                    positions[t] -= 1
                    cash += price * (1 - self.transaction_cost)

            portfolio_val = cash + sum(positions[t]*daily_prices[t] for t in tickers)
            portfolio_values.append(portfolio_val)
            benchmark_values.append(sum(self.initial_capital/len(tickers)/prices[t].iloc[0]*daily_prices[t] for t in tickers))

        self.results = {
            "portfolio_values": portfolio_values,
            "benchmark_values": benchmark_values,
            "dates": prices.index[1:],
            "portfolio_signals": portfolio_sig,
            "volatility_features": vol_features,
            "barrier_results": barrier_results,
            "ml_signals": ml_signals,
            "combined_signals": combined_signals,
            "model": model
        }

        return self.results
