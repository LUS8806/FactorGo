
# encoding=utf-8

import copy

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from backup.alpha_miner.base import BaseFactor
from FactorModel.constant import *


# TODO 修改代码，使其也适用于增量数据的因子计算，通过一个变量控制

# TODO 修改代码，transform 的参数不应该叫factor_frame


class IndustryFactor(BaseFactor):
    """哑变量行业因子"""

    def __init__(self, factor_name=None, data_loader=None, keep_ind_col=False):
        """
        Parameters
        ----------
        factor_name: str, 因子的名称, 默认'industry_dummy'
        data_loader: 数据加载器
        keep_ind_col: 是否保留INDUSTRY_COL
        """
        factor_name = 'industry_dummy' if not factor_name else factor_name
        super().__init__(factor_name=factor_name, data_loader=data_loader)
        self._keep_ind_col = keep_ind_col

    def fit(self, factor_frame):
        return self

    def prepare_data(self, factor_frame):
        factor_data = factor_frame.factor_data.copy()
        start_date = factor_data[DATE_COL].min()
        if not isinstance(start_date, str):
            start_date = str(start_date.date())

        end_date = factor_data[DATE_COL].max()
        if not isinstance(end_date, str):
            end_date = str(end_date.date())

        all_codes = list(set(factor_data[CODE_COL]))
        self._data = self._data_loader.get_stock_industries(all_codes, start_date, end_date)

    def transform(self, factor_frame):
        if self._data.empty:
            self.prepare_data(factor_frame)

        factor_frame = copy.copy(factor_frame)
        factor_data = factor_frame.factor_data
        factor_data = pd.merge(factor_data, self._data, how='left')
        factor_data = factor_data.dropna(subset=[INDUSTRY_COL])

        def add_industry_fac(df):
            """行业因子哑变量处理"""
            oht_coder = OneHotEncoder()
            ind_oht = oht_coder.fit_transform(df[[INDUSTRY_COL]])
            df_ind_fac = pd.DataFrame(data=ind_oht.toarray(), index=df.index, columns=oht_coder.categories_[0])
            return df_ind_fac

        factor_data_gp = factor_data.set_index([DATE_COL, CODE_COL]).groupby(level=0, sort=True)
        ind_data = factor_data_gp.apply(add_industry_fac)
        ind_data = ind_data.fillna(0).reset_index()
        factor_data = pd.merge(factor_data, ind_data, how='left')

        if not self._keep_ind_col:
            factor_data = factor_data.drop(columns=INDUSTRY_COL)

        factor_frame.update_data(factor_data)
        return factor_frame


class LogMarketCapFactor(BaseFactor):

    def __init__(self, factor_name=None, data_loader=None):
        factor_name = 'market_cap' if not factor_name else factor_name
        super().__init__(factor_name=factor_name, data_loader=data_loader)

    def fit(self, factor_frame):
        return self

    def prepare_data(self, factor_frame):
        factor_data = factor_frame.factor_data.copy()
        start_date = factor_data[DATE_COL].min()
        if not isinstance(start_date, str):
            start_date = str(start_date.date())

        end_date = factor_data[DATE_COL].max()
        if not isinstance(end_date, str):
            end_date = str(end_date.date())

        all_codes = list(set(factor_data[CODE_COL]))
        self._data = self._data_loader.get_stock_market_cap(all_codes, start_date, end_date)

    def transform(self, factor_frame):
        if self._data.empty:
            self.prepare_data(factor_frame)
        factor_frame = copy.copy(factor_frame)
        factor_data = factor_frame.factor_data
        factor_data = pd.merge(factor_data, self._data, how='left')
        factor_data[self.factor_name] = np.log(factor_data[self.factor_name])
        factor_frame.update_data(factor_data)
        return factor_frame


class TurnOverFactor(BaseFactor):

    def __init__(self, periods=None, factor_name=None, data_loader=None):
        factor_name = 'turnover' if not factor_name else factor_name
        super().__init__(factor_name=factor_name, data_loader=data_loader)
        self._periods = 5 if not periods else periods

        if isinstance(self._periods, int):
            self._periods = [self._periods]

    def fit(self, factor_frame):
        return self

    def prepare_data(self, factor_frame):
        factor_data = factor_frame.factor_data

        start_date = str(factor_data[DATE_COL].min().date())
        start_date = self._data_loader.get_trade_date_offset(start_date, -1*max(self._periods))

        end_date = str(factor_data[DATE_COL].max().date())

        all_codes = list(set(factor_data[CODE_COL]))
        df_turnover = self._data_loader.get_stock_turnover(
            all_codes, start_date, end_date)
        self._data = df_turnover

    def transform(self, factor_frame):
        if self._data.empty:
            self.prepare_data(factor_frame)

        factor_frame = copy.copy(factor_frame)
        pivot_data = self._data.pivot(index=DATE_COL,
                                      columns=CODE_COL,
                                      values='turnover_ratio')
        factor_data = factor_frame.factor_data.copy()

        for p in self._periods:
            mean_tv = pivot_data.rolling(p).mean()
            mean_tv = mean_tv.stack().reset_index()
            mean_tv.columns = [DATE_COL, CODE_COL, 'turnover_ratio_{}'.format(p)]
            factor_data = pd.merge(factor_data, mean_tv, how='left')

        factor_frame.update_data(factor_data)
        return factor_frame


class RateReturnFactor(BaseFactor):

    def __init__(self, periods=None, factor_name=None, data_loader=None):
        factor_name = 'ret' if not factor_name else factor_name
        super().__init__(factor_name=factor_name, data_loader=data_loader)
        self._periods = 5 if not periods else periods

        if isinstance(self._periods, int):
            self._periods = [self._periods]

    def fit(self, factor_frame):
        return self

    def prepare_data(self, factor_frame):
        factor_data = factor_frame.factor_data

        start_date = str(factor_data[DATE_COL].min().date())
        start_date = self._data_loader.get_trade_date_offset(start_date, -1*max(self._periods))

        end_date = str(factor_data[DATE_COL].max().date())

        all_codes = list(set(factor_data[CODE_COL]))
        df_close = self._data_loader.get_price_data(all_codes, start_date, end_date)
        self._data = df_close

    def transform(self, factor_frame):
        if self._data.empty:
            self.prepare_data(factor_frame)

        factor_frame = copy.copy(factor_frame)
        factor_data = factor_frame.factor_data

        for p in self._periods:
            p_col = str().join([self.factor_name, '_', str(p)])
            df_return = self._data.pct_change(p)
            df_return = df_return.stack().reset_index()
            df_return.columns = [DATE_COL, CODE_COL, p_col]
            factor_data = pd.merge(factor_data, df_return, how='left')

        factor_frame.update_data(factor_data)
        return factor_frame

    def set_period(self, periods):
        self._periods = periods
        if isinstance(self._periods, int):
            self._periods = [self._periods]


class StDevFactor(BaseFactor):

    def __init__(self, periods=None, factor_name=None, data_loader=None):
        factor_name = 'std' if not factor_name else factor_name
        super().__init__(factor_name=factor_name, data_loader=data_loader)
        self._periods = 5 if not periods else periods

        if isinstance(self._periods, int):
            self._periods = [self._periods]

    def fit(self, factor_frame):
        return self

    def prepare_data(self, factor_frame):
        factor_data = factor_frame.factor_data

        start_date = str(factor_data[DATE_COL].min().date())
        start_date = self._data_loader.get_trade_date_offset(start_date, -1*max(self._periods))
        end_date = str(factor_data[DATE_COL].max().date())
        all_codes = list(set(factor_data[CODE_COL]))
        df_close = self._data_loader.get_price_data(all_codes, start_date, end_date)
        self._data = df_close

    def transform(self, factor_frame):
        if self._data.empty:
            self.prepare_data(factor_frame)

        df_return = self._data.pct_change()
        factor_frame = copy.copy(factor_frame)
        factor_data = factor_frame.factor_data

        for p in self._periods:
            p_col = str().join([self.factor_name, '_', str(p)])
            df_std = df_return.rolling(p).std()
            df_std = df_std.stack().reset_index()
            df_std.columns = [DATE_COL, CODE_COL, p_col]
            factor_data = pd.merge(factor_data, df_std, how='left')

        factor_frame.update_data(factor_data)
        return factor_frame

    def set_period(self, periods):
        self._periods = periods
        if isinstance(self._periods, int):
            self._periods = [self._periods]


class BookPriceFactor(BaseFactor):

    """
    对于包含财务数据的指标，可以灵活选择是PIT还是按照财报对其

    """
    def __init__(self):

        super().__init__()


if  __name__ == '__main__':

    import time
    from FactorModel.get_data import DemoDataLoader
    from backup.alpha_miner.base import BaseFactorData


    data_loader = DemoDataLoader()

    # 股票池：沪深300
    df_fac = data_loader.get_stock_date_list(index_code='000300.XSHG',
                                             start_date='2019-01-01',
                                             end_date='2019-12-31')

    base_frame = BaseFactorData(factor_data=df_fac, base_index='000300.XSHG')

    st = time.time()
    ind_trans = IndustryFactor(data_loader=data_loader)
    ind_fac = ind_trans.fit_transform(base_frame)

    et = time.time()

    print("Ind Time Spend: {}".format((et-st)))
    print(ind_fac.factor_data.head())

    st = time.time()
    ind_trans = StDevFactor(data_loader=data_loader)
    ind_fac = ind_trans.fit_transform(base_frame)

    et = time.time()

    print("Std Time Spend: {}".format((et-st)))
    print(ind_fac.factor_data.head())


    st = time.time()
    ind_trans = TurnOverFactor(data_loader=data_loader)
    ind_fac = ind_trans.fit_transform(base_frame)

    et = time.time()

    print("Turnover Time Spend: {}".format((et-st)))
    print(ind_fac.factor_data.head())