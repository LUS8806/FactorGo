"""
因子算法

因子算法 + 基础数据库 --> FactorDataStruct

算法+计算+存储+测试 一步完成

"""
import abc
import pandas as pd
import numpy as np
import statsmodels.api as sm

from datetime import datetime, date
from pandas import DataFrame, MultiIndex, Index
from typing import Union, List
from sklearn.base import BaseEstimator, TransformerMixin
from FactorGo.factor_base import FactorDataStruct
from FactorGo.factor_test.util import exp_weight
from FactorGo.data_loader import data_api, BaseDataLoader


class BaseFactorAlgo(object, metaclass=abc.ABCMeta):

    """
    因子计算基础类：

    @属性

    @方法
    1. 增量更新: update_daily
    2. 批量更新: update_batch

    """

    _data_loader = data_api
    params = {}

    def __init__(self, data_loader: BaseDataLoader = None):
        if data_loader is not None:
            self._data_loader = data_loader

    @abc.ABCMeta
    def _update_daily(self, start_date=None):
        """日度更新"""
        ...

    @abc.ABCMeta
    def _update_batch(self, start_date, end_date):
        """批量更新"""
        ...

    def update(self, start_date, end_date):
        """更新到最新数据"""
        ...

    def save_to_db(self, conn, table_name):
        """保存至数据库"""
        ...

    def load_data(self, sec_codes, start_date, end_date):
        """加载数据"""
        ...

    @classmethod
    def set_params(cls, **kwargs):
        """设置计算参数"""
        for k, v in kwargs.items():
            cls.params[k] = v


class BetaSigma(BaseFactorAlgo):

    params = {
        'beta_ws': 252,
        'beta_hl': 63,
    }

    def __init__(self, data_loader: BaseDataLoader = None):
        super().__init__(data_loader)

    def prepare_data(self,
                     codes_date_index: Union[MultiIndex, DataFrame] = None,
                     sec_codes: List[str] = None,
                     market_index: str = '000300.XSHG',
                     dates: Union[str, datetime, date, list, Index] = None,
                     start_date: Union[str, datetime, date] = None,
                     end_date: Union[str, datetime, date] = None) -> DataFrame:

        stock_price = self._data_loader.get_stock_price(codes_date_index,
                                                        sec_codes,
                                                        dates,
                                                        start_date,
                                                        end_date,
                                                        fields=['close'])

        index_price = self._data_loader.get_index_price(market_index, start_date, end_date, fields=['close'])

        stock_ret = stock_price['close'].unstack().pct_change().dropna()
        index_ret = index_price['close'].unstack().pct_change().dropna()
        df_ret = pd.concat([stock_ret, index_ret], axis=1)
        return df_ret

    def _update_daily(self, start_date=None):
        ...

    def _update_batch(self,
                      data: DataFrame = None,
                      codes_date_index: Union[MultiIndex, DataFrame] = None,
                      sec_codes: List[str] = None,
                      market_index: str = '000300.XSHG',
                      dates: Union[str, datetime, date, list, Index] = None,
                      start_date: Union[str, datetime, date] = None,
                      end_date: Union[str, datetime, date] = None,
                      rf_rate: float = 0.03):
        """
        Beta >>> BETA
        Residual Volatility >>> HSIGMA
        """

        if data is None:
            data = self.prepare_data(codes_date_index, sec_codes, market_index, dates, start_date, end_date)

        if 'rf_rate' not in data:
            data['rf_rate'] = rf_rate

        ws = self.params['beta_ws']
        hl = self.params['beta_hl']
        weight = exp_weight(hl, ws)

        ret = pd.concat([data['mkt_ret'],
                         data['sec_ret'],
                         data['trade_status'],
                         data['rf_rate']
                         ], join='inner', axis=1)

        ret.columns = ['mkt_ret', 'sec_ret', 'trade_status', 'rf_rate']

        df_beta = {}
        df_hsigma = {}

        for sec in data:
            if sec not in [market_index, 'rf_rate']:
                ex_ret = data[[sec, market_index]].dropna()
                ex_ret = ex_ret.sub(data['rf_rate'], axis=0)
                data_len = len(ex_ret)

                def _regress(_x, _y):
                    x_notna = _x[~np.isnan(_x)]
                    w_notna = weight[~np.isnan(_x)]
                    w = w_notna / w_notna.sum()
                    y_notna = _y[~np.isnan(_x)]

                    if len(x_notna) < 42:
                        return np.nan, np.nan

                    w_mat = np.diag(w)
                    X = sm.add_constant(x_notna)
                    coef = np.linalg.inv(X.T @ w_mat @ X) @ X.T @ w_mat @ y_notna
                    res = y_notna - X @ coef
                    return coef[1], np.std(res)

                beta_arr = np.full(data_len, np.nan)
                hsigma_arr = np.full(data_len, np.nan)

                for i in range(ws - 1, data_len):
                    x = ex_ret[sec].iloc[i - ws:i].values
                    y = ex_ret[market_index].iloc[i - ws:i].values
                    beta_arr[i], hsigma_arr[i] = _regress(x, y)

                beta = pd.Series(beta_arr, index=ex_ret.index)
                hsigma = pd.Series(hsigma_arr, index=ex_ret.index)

                df_beta[sec] = beta
                df_hsigma[sec] = hsigma

        return pd.DataFrame(df_beta), pd.DataFrame(df_hsigma)

    def _calculate_factor_for_each_sec(self):
        """TODO 价格数据内存数据库 或者使用 @lru_cache缓存"""
        ...


class RSTA(BaseFactorAlgo):
    """
    Barra风格因子的描述变量
    """
    params = {
        'hl': 504,
        'ws': 126,
        'lag': 21
    }

    def __init__(self):
        super().__init__()

    def transform(self, data):
        """Momentum >>> RSTR"""
        hl = self.params['hl']
        ws = self.params['ws']
        lag = self.params['lag']
        roll_window = ws + lag

        log_ex_ret = np.log(1 + data['sec_ret']) - np.log(1 + data['rf_rate'])
        weight = exp_weight(hl, ws)

        _rstr = lambda x: np.sum(x[:-1*lag] * weight)
        rstr = log_ex_ret.rolling(roll_window).apply(_rstr, raw=True)

        rstr.name = "RSTR"


class LNCAP(BaseFactorAlgo):

    def transform(self, data):
        """Size >>> LNCAP"""
        ret = np.log(data['mkt_cap'])
        ret.name = 'LNCAP'

        return ret.reindex(self.desc_index)


class DASTD(BaseFactorAlgo):
    def transform(self, data):
        """Residual Volatility >>> DASTD"""

        hl = self.params['dastd_hl']
        ws = self.params['dastd_ws']

        ex_ret = self.data['sec_ret'] - self.data['rf_rate']
        ex_ret[self.data['trade_status'] == 1] = np.nan

        weight = self.exp_weight(hl, ws)

        def _dastd(x):
            x_notna = x[~np.isnan(x)]
            w_notna = weight[~np.isnan(x)]
            w_notna = w_notna / w_notna.sum()
            if len(x_notna) < 42:
                return np.nan
            else:
                x_w = x_notna * np.sqrt(w_notna)
                return np.nanstd(x_w)

        ret = ex_ret.rolling(ws).apply(_dastd, raw=True)
        ret.name = 'DASTD'

        return ret.reindex(self.desc_index)


class CMRA(BaseFactorAlgo):
    params = {
        'prd': None
    }
    def transform(self, data):
        """Residual Volatility >>> CMRA """

        prd = self.params['cmra_prd']
        ws = prd * 21

        log_ex_ret = np.log(1 + self.data['sec_ret']) - np.log(1 + self.data['rf_rate'])

        def _cmra(x):
            z = np.full(prd, np.nan)
            for p in range(0, prd):
                z[p] = np.sum(x[-(p + 1) * 21:])
            res = z.max() - z.min()
            return res

        ret = log_ex_ret.rolling(ws).apply(_cmra, raw=True).dropna()
        ret.name = 'CMRA'

        return ret.reindex(self.desc_index)

    def SGRO(self):

        """Growth >>> SGRO"""

        data = self.data['sales_per_share'].copy().dropna()
        roll_res = data.rolling(5, min_periods=2).apply(self._growth_func, raw=True)
        roll_res.name = "SGRO"

        roll_res.index = roll_res.index.date
        roll_res.index.name = 'rpt_date'
        dct = {k: v for v, k in self.rpt_pub.iteritems()}
        roll_res.index = roll_res.index.map(dct)
        roll_res = roll_res.drop_duplicates()
        join_index = roll_res.index.join(self.desc_index, how='outer')
        ret = roll_res.reindex(join_index).fillna(method='ffill')
        ret.name = "SGRO"

        return ret.reindex(self.desc_index)

    def EGRO(self):
        """Growth >>> EGRO"""

        data = self.data['earning_per_share'].copy().dropna()
        roll_res = data.rolling(5, min_periods=2).apply(self._growth_func, raw=True)
        roll_res.name = "EGRO"
        roll_res.index = roll_res.index.date
        roll_res.index.name = 'rpt_date'

        dct = {k: v for v, k in self.rpt_pub.iteritems()}
        roll_res.index = roll_res.index.map(dct)
        roll_res = roll_res.drop_duplicates()
        join_index = roll_res.index.join(self.desc_index, how='outer')
        ret = roll_res.reindex(join_index).fillna(method='ffill')
        ret.name = "EGRO"

        return ret.reindex(self.desc_index)

    def BTOP(self):
        """Book-to-Price >>> BTOP"""

        bv = self.data['book_value'].copy()
        mkt_cap = self.data['mkt_cap'].copy()
        bv.name = 'book_value'

        bv_td = self._rpt_td_merge(bv)
        ret = bv_td / mkt_cap
        ret.name = 'BTOP'

        return ret.reindex(self.desc_index)

    def ETOP(self):
        """Earnings Yield >>> ETOP"""

        data = self.data['pe_ttm'].copy()
        ret = 1 / data
        ret.name = "ETOP"

        return ret.reindex(self.desc_index)

    def CETOP(self):
        """Earnings Yield >>> ETOP"""

        cf = self.data['cashflow_ttm'].copy()
        mkt_cap = self.data['mkt_cap'].copy()
        cf_td = self._rpt_td_merge(cf)
        ret = cf_td / mkt_cap
        ret.name = 'CETOP'

        return ret.reindex(self.desc_index)

    def MLEV(self):
        """Leverge >>> MLEV"""

        ld = self.data['long_debt'].copy()
        mkt_cap = self.data['mkt_cap'].copy()

        ld_td = self._rpt_td_merge(ld)

        ret = (mkt_cap + ld_td) / mkt_cap
        ret.name = 'MLEV'

        return ret.reindex(self.desc_index)

    def DTOA(self):
        """Leverge >>> DTOA"""

        data = self.data['debt_asset_ratio'].copy()
        #         data.index = pd.to_datetime(data.index)
        data.name = 'DTOA'
        ret = self._rpt_td_merge(data) / 100
        ret.name = 'DTOA'

        return ret.reindex(self.desc_index)

    def BLEV(self):
        """Leverge >>> BLEV"""

        ld = self.data['long_debt'].copy()
        be = self.data['book_value'].copy()

        ret = (be + ld) / be
        ret.index = pd.to_datetime(ret.index)
        ret.name = "BLEV"
        ret = self._rpt_td_merge(ret)
        ret.name = "BLEV"
        return ret

    def STOM(self):
        """Liquidity >>> STOM"""

        data = self.data['turnover'].copy()
        ret = data.rolling(21).apply(lambda x: self._sto_func(x, 1), raw=True).dropna()
        ret.name = "STOM"

        return ret.reindex(self.desc_index)

    def STOQ(self):
        """Liquidity >>> STOQ"""

        data = self.data['turnover'].copy()
        ret = data.rolling(63).apply(lambda x: self._sto_func(x, 3), raw=True).dropna()
        ret.name = "STOQ"

        return ret.reindex(self.desc_index)

    def STOA(self):
        """Liquidity >>> STOA"""

        data = self.data['turnover'].copy()
        ret = data.rolling(252).apply(lambda x: self._sto_func(x, 12), raw=True).dropna()
        ret.name = "STOA"

        return ret.reindex(self.desc_index)

    def NLSIZE(self):
        """Non-linear Size >>> NLSIZE"""

        data = self.data['mkt_cap'].dropna()

        lg_cap = np.log(data)
        cube_cap = np.power(lg_cap, 3)

        model = sms.OLS(cube_cap, lg_cap)
        res = model.fit()
        ret = pd.Series(res.resid, index=lg_cap.index)

        ret.name = "NLSIZE"

        return ret.reindex(self.desc_index)

    @staticmethod
    def _sto_func(x, months):

        sto = np.log(np.sum(x) / months)
        if sto == - np.inf:
            sto = 0
        return sto

    @staticmethod
    def _growth_func(y):

        x = np.arange(1, len(y) + 1)
        model = sms.OLS(y, x)
        res = model.fit(params_only=True)
        ret = res.params[0] / y.mean()
        if ret == np.inf or ret == -np.inf:
            ret = np.nan
        return ret

    def _rpt_td_merge(self, data, rpt_col=None, index_rpt=True, data_name=None):
        """以财报日期为Index"""

        if not data_name:
            data_name = 'DESCRIPTOR'

        if not data.name:
            data.name = data_name

        data_name = data.name

        data.index = data.index.date
        data = data.to_frame()

        if index_rpt:
            data.index.name = 'rpt_date'
            data = data.reset_index()

        left = self.td_rpt_s.to_frame(name='rpt_date').reset_index()
        comb = pd.merge(left, data, on='rpt_date', how='left').set_index('trade_date')

        return comb[data_name]