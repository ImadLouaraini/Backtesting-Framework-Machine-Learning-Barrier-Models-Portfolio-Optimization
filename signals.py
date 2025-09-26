import numpy as np
import pandas as pd
from scipy.stats import norm
import itertools

def calculate_barrier_metrics(prices, lookback_days=126, horizon_days=30):
    metrics = []
    for i in range(lookback_days, len(prices) - horizon_days):
        current_price = prices.iloc[i]
        hist = prices.iloc[i - lookback_days:i]
        returns = np.log(hist / hist.shift(1)).dropna()
        vol = returns.std() * np.sqrt(252) or 0.3
        barrier_levels = [current_price * (1 - d) for d in [0.05, 0.10, 0.15]]
        probs = []
        for barrier in barrier_levels:
            if current_price <= barrier:
                prob = 1.0
            else:
                distance = np.log(current_price / barrier)
                prob = norm.cdf(-distance / (vol * np.sqrt(horizon_days / 252)))
            probs.append(prob)
        
        avg_prob = np.mean(probs)

        # 🔑 Generate trading signal
        if avg_prob > 0.7:
            signal = "SELL"
        elif avg_prob < 0.3:
            signal = "BUY"
        else:
            signal = "HOLD"

        metrics.append({
            'date': prices.index[i],
            'price': current_price,
            'volatility': vol,
            'barrier_5pct': probs[0],
            'barrier_10pct': probs[1],
            'barrier_15pct': probs[2],
            'avg_barrier_prob': avg_prob,
            'signal': signal
        })
    return pd.DataFrame(metrics)


def calculate_correlations(returns_data, lookback=63):
    correlations = {}
    for s1, s2 in itertools.combinations(returns_data.columns, 2):
        r = returns_data[[s1, s2]].dropna()
        if len(r) > lookback:
            corr = r[s1].rolling(lookback).corr(r[s2]).iloc[-1]
            correlations[(s1, s2)] = corr if not np.isnan(corr) else 0
        else:
            correlations[(s1, s2)] = 0
    return correlations


def portfolio_signals(barrier_dfs, returns_data, lookback=63):
    """
    Combine individual asset signals with correlation filter.
    """
    correlations = calculate_correlations(returns_data, lookback)
    latest_signals = {asset: df.iloc[-1]['signal'] for asset, df in barrier_dfs.items()}

    # Apply correlation filter
    for (a1, a2), corr in correlations.items():
        if corr > 0.8 and latest_signals[a1] == latest_signals[a2] == "BUY":
            # Too correlated → avoid doubling risk
            latest_signals[a2] = "HOLD"

    return latest_signals
