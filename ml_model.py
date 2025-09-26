import pandas as pd
import numpy as np
import lightgbm as lgb

def compute_features(price_data, benchmark_prices, horizon_days=30, clip=0.3):
    feats_list = []
    bench_ret = benchmark_prices.pct_change()
    bench_var_63 = bench_ret.rolling(63).var()

    for ticker in price_data.columns:
        s = price_data[ticker].dropna()
        if s.empty: continue
        df = pd.DataFrame({
            "ret_1": s.pct_change(),
            "ret_5": s.pct_change(5),
            "ma_20_div": s / s.rolling(20).mean() - 1,
            "vol_21": s.pct_change().rolling(21).std(),
        })
        future_ret = s.shift(-horizon_days) / s - 1
        df['target'] = future_ret.clip(-clip, clip)
        df['ticker'] = ticker
        feats_list.append(df)
    big = pd.concat(feats_list).dropna()
    big = big.set_index(['ticker'], append=True).sort_index()
    return big.drop(columns=['target']), big['target']

def train_lightgbm(X, y, train_end_date, recency_lambda=0.002):
    idx_dates = X.index.get_level_values(0)
    mask = idx_dates <= pd.Timestamp(train_end_date)
    X_train, y_train = X[mask], y[mask]
    tr_dates = X_train.index.get_level_values(0)
    age_days = (pd.Timestamp(train_end_date) - pd.to_datetime(tr_dates)).days
    w = np.exp(-recency_lambda * age_days)
    model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.03)
    model.fit(X_train, y_train, sample_weight=w)
    return model
