
import numpy as np
from arch import arch_model

class VolatilityFeatures:
    """Calcul avancé de volatilité pour un actif."""

    @staticmethod
    def realized_volatility(returns, window=20):
        return returns.rolling(window).std()

    @staticmethod
    def garch_volatility(prices, p=1, q=1):
        returns = np.log(prices / prices.shift(1)).dropna() * 100
        model = arch_model(returns, vol="Garch", p=p, q=q)
        res = model.fit(disp="off")
        return res.conditional_volatility / 100

    @staticmethod
    def parkinson_volatility(high, low, window=20):
        rs = (1 / (4 * np.log(2))) * (np.log(high / low)) ** 2
        return np.sqrt(rs.rolling(window).mean())

    @staticmethod
    def garman_klass_volatility(open_, high, low, close, window=20):
        log_hl = np.log(high / low)
        log_co = np.log(close / open_)
        rs = 0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2
        return np.sqrt(rs.rolling(window).mean())

    @staticmethod
    def close_to_close_volatility(prices, window=20):
        returns = np.log(prices / prices.shift(1))
        return returns.rolling(window).std()
