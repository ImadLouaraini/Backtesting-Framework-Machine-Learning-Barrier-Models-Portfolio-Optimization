# main.py
from backtest import BacktestEngine

if __name__ == "__main__":
    engine = BacktestEngine()
    res = engine.run(
        tickers=['AAPL','MSFT','GOOGL'],
        start_date="2020-01-01",
        end_date="2024-01-01"
    )

    # Afficher quelques résultats
    print("📌 Portfolio signals:", res["portfolio_signals"])
    for ticker, df in res["barrier_results"].items():
        print(f"\n📌 {ticker} barrier signals (dernieres lignes):\n", df.tail(3))
