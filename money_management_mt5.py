class MoneyManagerMT5:
    def __init__(self, account_size=100000, risk_per_trade=0.01, default_lot=0.1):
        self.account_size = account_size
        self.risk_per_trade = risk_per_trade
        self.default_lot = default_lot

    def calculate_lot(self, price, stop_loss):
        risk_amount = self.account_size * self.risk_per_trade
        if stop_loss is None or stop_loss == price:
            return self.default_lot
        lot = max(self.default_lot, round(risk_amount / abs(price - stop_loss), 2))
        return lot

    def get_sl_tp(self, price, direction, sl_pct=0.002, tp_pct=0.004):
        if direction == "BUY":
            sl = price * (1 - sl_pct)
            tp = price * (1 + tp_pct)
        elif direction == "SELL":
            sl = price * (1 + sl_pct)
            tp = price * (1 - tp_pct)
        else:
            sl = tp = None
        return sl, tp
