import pandas as pd

class MoneyManager:
    def __init__(self, account_size=100000, risk_per_trade=0.01, default_lot=1):
        """
        account_size: total cash in USD
        risk_per_trade: % of account to risk per trade
        default_lot: default contract/stock size
        """
        self.account_size = account_size
        self.risk_per_trade = risk_per_trade
        self.default_lot = default_lot

    def calculate_lot(self, price, stop_loss):
        """
        Calculate lot size based on stop-loss distance and risk per trade.
        """
        risk_amount = self.account_size * self.risk_per_trade
        if stop_loss is None or stop_loss == price:
            return self.default_lot
        lot = max(1, round(risk_amount / abs(price - stop_loss)))
        return lot

    def get_sl_tp(self, price, direction, sl_pct=0.02, tp_pct=0.04):
        """
        Calculate Stop-Loss and Take-Profit prices.
        direction: "BUY" or "SELL"
        sl_pct: stop-loss percentage
        tp_pct: take-profit percentage
        """
        if direction == "BUY":
            sl = price * (1 - sl_pct)
            tp = price * (1 + tp_pct)
        elif direction == "SELL":
            sl = price * (1 + sl_pct)
            tp = price * (1 - tp_pct)
        else:
            sl = tp = None
        return sl, tp
