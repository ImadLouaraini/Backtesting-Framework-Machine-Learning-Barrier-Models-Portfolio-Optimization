import pandas as pd
from .data import fetch_multiple_stocks
from .signals import calculate_barrier_metrics, portfolio_signals
from .volatility import VolatilityFeatures
from .ml_model import compute_features, train_lightgbm
from .report import plot_portfolio_results, print_report
import numpy as np

class BacktestEngineML:
    def __init__(self, initial_capital=100000, transaction_cost=0.001):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.results = {}

    def run(self, tickers, start_date, end_date, train_end_date):
        # 1️⃣ Fetch prices
        prices = fetch_multiple_stocks(tickers, start_date, end_date)
        returns = prices.pct_change().dropna()

        barrier_results = {}
        vol_features = pd.DataFrame(index=prices.index)

        # 2️⃣ Compute volatility features and barrier signals
        for ticker in tickers:
            df = prices[[ticker]].copy()

            # Volatilités avancées
            df["realized_vol"] = VolatilityFeatures.realized_volatility(df[ticker].pct_change())
            df["parkinson_vol"] = VolatilityFeatures.parkinson_volatility(df[ticker], df[ticker])
            df["garman_klass_vol"] = VolatilityFeatures.garman_klass_volatility(df[ticker], df[ticker], df[ticker], df[ticker])
            try:
                df["garch_vol"] = VolatilityFeatures.garch_volatility(df[ticker])
            except:
                df["garch_vol"] = df[ticker].pct_change().rolling(20).std()

            barrier_df = calculate_barrier_metrics(df[ticker])
            barrier_results[ticker] = barrier_df
            vol_features = pd.concat([vol_features, df[["realized_vol","parkinson_vol","garman_klass_vol","garch_vol"]]], axis=1)

        # 3️⃣ Compute ML features and train LightGBM
        X, y = compute_features(prices, horizon_days=30, clip=0.3)
        model = train_lightgbm(X, y, train_end_date=train_end_date)

        # 4️⃣ Predict future returns for all tickers
        ml_signals = {}
        for ticker in tickers:
            X_ticker = X.xs(ticker, level=1)
            y_pred = model.predict(X_ticker)
            ml_signals[ticker] = pd.Series(y_pred, index=X_ticker.index).apply(lambda x: "BUY" if x>0 else "SELL")

        # 5️⃣ Combine ML signals with barrier signals using correlation filter
        combined_signals = {}
        for ticker in tickers:
            barrier_df = barrier_results[ticker].set_index("date")
            ml_df = ml_signals[ticker].to_frame(name="ml_signal")
            combined = pd.concat([barrier_df["signal"], ml_df], axis=1, join="inner")
            # simple rule: if both agree, keep; else HOLD
            combined["signal"] = np.where(combined["signal"]==combined["ml_signal"], combined["signal"], "HOLD")
            combined_signals[ticker] = combined

        portfolio_sig = portfolio_signals(combined_signals, returns)

        # 6️⃣ Simplified backtesting
        portfolio_values = []
        benchmark_values = []
        cash = self.initial_capital
        positions = {t: 0 for t in tickers}

        for date in prices.index[1:]:
            daily_prices = prices.loc[date]
            signals = {t: combined_signals[t].loc[date, "signal"] if date in combined_signals[t].index else "HOLD"
                       for t in tickers}

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
