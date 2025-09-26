# backtest.py
import pandas as pd
from .data import fetch_multiple_stocks
from .signals import calculate_barrier_metrics, calculate_correlations, portfolio_signals
from .volatility import VolatilityFeatures
from .report import plot_portfolio_results, print_report

class BacktestEngine:
    def __init__(self, initial_capital=100000, transaction_cost=0.001):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.results = {}

    def run(self, tickers, start_date, end_date):
        # 1️⃣ Fetch prices
        prices = fetch_multiple_stocks(tickers, start_date, end_date)
        returns = prices.pct_change().dropna()

        barrier_results = {}
        vol_features = pd.DataFrame(index=prices.index)

        # 2️⃣ Compute volatility features and barrier signals per ticker
        for ticker in tickers:
            df = prices[[ticker]].copy()
            
            # Multi-volatility
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

        # 3️⃣ Compute portfolio signals with correlation filter
        portfolio_sig = portfolio_signals(barrier_results, returns)

        # 4️⃣ Simplified portfolio backtesting
        portfolio_values = []
        benchmark_values = []
        cash = self.initial_capital
        positions = {t: 0 for t in tickers}

        for date in prices.index[1:]:
            daily_prices = prices.loc[date]
            signals = {t: barrier_results[t].loc[barrier_results[t]['date']==date, 'signal'].values[0]
                       if not barrier_results[t].loc[barrier_results[t]['date']==date].empty else "HOLD"
                       for t in tickers}

            # naive allocation: buy/sell 1 unit per signal
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
            "barrier_results": barrier_results
        }

        return self.results
