import yfinance as yf
import pandas as pd
import numpy as np

def fetch_multiple_stocks(tickers, start_date, end_date, log=print):
    """
    Télécharge les prix de clôture ajustés pour plusieurs tickers via yfinance.
    
    Parameters:
        tickers (list or str): Liste de tickers ou un ticker unique.
        start_date (str): Date de début "YYYY-MM-DD".
        end_date (str): Date de fin "YYYY-MM-DD".
        log (callable): fonction pour logging (default=print).
    
    Returns:
        pd.DataFrame: DataFrame avec les colonnes des tickers et les prix ajustés.
    """
    if isinstance(tickers, str):
        tickers = [tickers]
    
    log(f"Fetching data for {len(tickers)} tickers...")

    raw = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        group_by='ticker',
        progress=False,
        threads=True
    )

    if raw is None or raw.empty:
        raise ValueError("No price data returned from yfinance.")

    closes = {}
    # MultiIndex (tickers séparés)
    if isinstance(raw.columns, pd.MultiIndex):
        for ticker in tickers:
            found = False
            for col in [('Close', ticker), ('Adj Close', ticker), (ticker, 'Close'), (ticker, 'Adj Close')]:
                if col in raw.columns:
                    closes[ticker] = raw[col]
                    found = True
                    break
            if not found:
                log(f"⚠️ Close/Adj Close not found for {ticker}, skipping.")
    # Single ticker or simple columns
    else:
        col_name = 'Close' if 'Close' in raw.columns else 'Adj Close'
        for ticker in tickers:
            closes[ticker] = raw[col_name]

    data = pd.DataFrame(closes)
    data = data.dropna(axis=1, how='all').dropna(how='all')

    if data.empty:
        raise ValueError("No valid price columns found after processing.")

    log(f"Data fetched successfully: {data.shape[0]} rows, {data.shape[1]} tickers.")
    return data
