import numpy as np
import pandas as pd


class Portfolio:
    def __init__(self, price_frame: pd.DataFrame):
        if price_frame is None or price_frame.empty:
            raise ValueError("Price data is required.")
        self.data = price_frame.copy()
        self.daily_returns = None

    def compute_returns(self, use_log=True):
        source = self.data
        if use_log:
            r = np.log(source.div(source.shift(1)))
        else:
            r = source.pct_change()
        self.daily_returns = r.dropna()
        return self.daily_returns
