# encoding=utf8

import numpy as np
import pandas as pd
import pyfolio as pf
from typing import Union
from pandas import DataFrame, Series
from functools import wraps
from datetime import datetime
from dateutil.parser import parse
from scipy.stats import spearmanr
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from sklearn.preprocessing import OneHotEncoder
from FactorGo.factor_test.constant import *
from DataGo.fetch import get_trade_date_between

"""
TODO check_ret_col函数，检查ret_period是否符合要求
"""


def to_dummy_variable(data: Union[DataFrame, Series]) -> DataFrame:
    """
    转换为哑变量

    Parameters
    ----------
    data: DataFrame or Series, 需要转变为哑变量的数据

    Returns
    -------
    DataFrame

    """
    if isinstance(data, DataFrame):
        data = data.iloc[0]

    oht_coder = OneHotEncoder()
    ind_oht = oht_coder.fit_transform(data)
    df_dummy = pd.DataFrame(data=ind_oht.toarray(), index=data.index, columns=oht_coder.categories_[0])
    return df_dummy


def sec_code_reformat(sec_codes: Union[Series, np.ndarray, list]) -> Series:
    """
    股票代码转换为.XSHE, .XSHG结尾

    Parameters
    ----------
    sec_codes: Union[Series, np.ndarray, list], 股票代码序列

    Returns
    -------

    """

    if isinstance(sec_codes, (np.ndarray, list)):
        sec_codes = pd.Series(sec_codes)

    sec_codes = sec_codes.str[0:6] + sec_codes.str[0].map({'0': '.XSHE',
                                                           '3': '.XSHE',
                                                           '6': '.XSHG'})

    return sec_codes


def df_winsorize_std(df: Union[Series, DataFrame],
                     ex_num: int = 3) -> Union[Series, DataFrame]:
    """
    标准差去极端值

    Parameters
    ----------
    df: DataFrame, 数据
    ex_num: int, 极值判断的范围，如果ex_num=3，表示数据到均值距离超过3倍标准差，会被修正为离均值3倍标准差的大小

    Returns
    -------
    DataFrame, 处理后的数据
    """
    mu = df.mean()
    std = df.std()
    upper = mu + ex_num * std
    lower = mu - ex_num * std
    return df.clip(lower=lower, upper=upper)


def df_winsorize_mad(df: Union[Series, DataFrame],
                     ex_num: int = 5) -> Union[Series, DataFrame]:
    """
    MAD(median absolute deviation)去极值

    Parameters
    ----------
    df: DataFrame, 数据
    ex_num: int, 极值判断的范围

    Returns
    -------
    DataFrame, 处理后的数据
    """
    median_1 = df.median()
    median_2 = (df - median_1).abs().median()

    upper = median_1 + ex_num * median_2
    lower = median_1 - ex_num * median_2

    return df.clip(lower=lower, upper=upper)


def df_winsorize_quantile(df: Union[Series, DataFrame],
                          low: float = 0.05,
                          up: float = 0.95,
                          ) -> Union[Series, DataFrame]:
    """
    分位数去极值

    Parameters
    ----------
    df: DataFrame, 数据
    low: float, 默认为0.05, 极值判断的下分位数
    up: float, 默认为0.95, 极值判断的上分位数

    Returns
    -------
    DataFrame, 处理后的数据
    """

    upper = df.quantile(up)
    lower = df.quantile(low)

    return df.clip(lower=lower, upper=upper)


def df_standardize_norm(df: Union[Series, DataFrame]) -> Union[Series, DataFrame]:
    """
    单个df标准化, z-score

    Parameters
    ----------
    df: DataFrame, 数据

    Returns
    -------
    DataFrame, 处理后的数据
    """
    mu = df.mean()
    std = df.std()
    return df.subtract(mu).divide(std)


def demean_forward_returns(forward_ret: Series,
                           grouper: list = None) -> Series:
    """
    收益率去均值

    Parameters
    ----------
    forward_ret: 收益率数据
    grouper: 分组, 默认按日期分组

    Returns
    -------

    """

    forward_ret = forward_ret.copy()

    if not grouper:
        grouper = forward_ret.index.get_level_values(DATE_COL)

    forward_ret = forward_ret.groupby(grouper).transform(lambda x: x - x.mean())

    return forward_ret


def quantize_factor(factor_data: Union[Series, DataFrame],
                    quantiles: Union[int, list] = 5,
                    bins: Union[int, list] = None,
                    by_group: Union[DataFrame, Series] = None,
                    no_raise: bool = False,
                    zero_aware: bool = False):

    """
    因子分组

    Parameters
    ----------
    factor_data
    quantiles
    bins
    by_group
    no_raise
    zero_aware

    Returns
    -------

    """

    if isinstance(factor_data, DataFrame):
        factor_data = factor_data.iloc[0]

    if isinstance(by_group, DataFrame):
        by_group = by_group.iloc[0]

    if not ((quantiles is not None and bins is None) or
            (quantiles is None and bins is not None)):
        raise ValueError('Either quantiles or bins should be provided')

    if zero_aware and not (isinstance(quantiles, int)
                           or isinstance(bins, int)):
        msg = ("zero_aware should only be True when quantiles or bins is an"
               " integer")
        raise ValueError(msg)

    def quantile_calc(x, _quantiles, _bins, _zero_aware, _no_raise):
        try:
            if _quantiles is not None and _bins is None and not _zero_aware:
                return pd.qcut(x, _quantiles, labels=False) + 1
            elif _quantiles is not None and _bins is None and _zero_aware:
                pos_quantiles = pd.qcut(x[x >= 0], _quantiles // 2,
                                        labels=False) + _quantiles // 2 + 1
                neg_quantiles = pd.qcut(x[x < 0], _quantiles // 2,
                                        labels=False) + 1
                return pd.concat([pos_quantiles, neg_quantiles]).sort_index()
            elif _bins is not None and _quantiles is None and not _zero_aware:
                return pd.cut(x, _bins, labels=False) + 1
            elif _bins is not None and _quantiles is None and _zero_aware:
                pos_bins = pd.cut(x[x >= 0], _bins // 2,
                                  labels=False) + _bins // 2 + 1
                neg_bins = pd.cut(x[x < 0], _bins // 2,
                                  labels=False) + 1
                return pd.concat([pos_bins, neg_bins]).sort_index()
        except Exception as e:
            if _no_raise:
                return pd.Series(index=x.index)
            raise e

    grouper = [factor_data.index.get_level_values(DATE_COL)]  # 按日期分组

    if by_group:
        grouper.append(by_group)

    factor_quantile = factor_data.groupby(grouper).apply(quantile_calc, quantiles, bins, zero_aware, no_raise)

    return factor_quantile


def factor_ic(factor_data: Union[Series, DataFrame],
              forward_ret: Union[Series, DataFrame],
              group_adjust: bool = False,
              by_group: Union[Series, DataFrame] = None,
              ) -> Series:
    """
    计算IC

    Parameters
    ----------
    factor_data: Series or DataFrame, 因子数据, 如果是DataFrame，则会默认以第一列作为因子
    forward_ret: Series or DataFrame, 收益率数据, 如果是DataFrame，则会默认以第一列作为收益率
    group_adjust: bool, 默认False, 收益率去均值
    by_group: Series or DataFrame, 分组的依据

    Returns
    -------

    """
    if isinstance(factor_data, DataFrame):
        factor_data = factor_data.iloc[0]

    if isinstance(forward_ret, DataFrame):
        forward_ret = forward_ret.iloc[0]

    if isinstance(by_group, DataFrame):
        by_group = by_group.iloc[0]

    factor_name = factor_data.name
    return_name = forward_ret.name

    def src_ic(group):
        fac_value = group[factor_name]
        _ic = group[return_name].apply(lambda x: spearmanr(x, fac_value)[0])
        return _ic

    all_data = pd.merge(factor_data, forward_ret, left_index=True, right_index=True, how='left').dropna()
    grouper = [all_data.index.get_level_values(DATE_COL)]

    if isinstance(by_group, Series):
        group_name = by_group.name
        all_data = pd.merge(all_data, by_group, left_index=True, right_index=True)
        all_data = all_data.dropna(subset=[group_name])
        grouper = [all_data.index.get_level_values(DATE_COL), group_name]

    if group_adjust:
        all_data = demean_forward_returns(all_data, grouper)

    ic = all_data.groupby(grouper).apply(src_ic)

    return ic


def exp_weight(half_life, window_size, reverse=True):
    """根据半衰期计算指数加权的权重"""
    alpha = 1 - np.exp(np.log(0.5) / half_life)
    ew = np.zeros(window_size)
    ew[0] = alpha
    ew[1:] = np.cumprod(np.full(window_size - 1, 1 - alpha)) * alpha

    if reverse:
        ew = ew[::-1]
    ret = ew / ew.sum()

    return ret


def match_trade_date(dates: Union[Series, list, np.ndarray],
                     cut_hour: int = 15,
                     cut_minute: int = 0,
                     trade_calendar: Union[list, np.ndarray] = None):
    """
    match trade_date of
    Parameters
    ----------
    dates
    cut_hour
    cut_minute
    trade_calendar

    Returns
    -------

    """

    if not isinstance(dates, Series):
        dates = pd.Series(dates)

    dates = pd.to_datetime(dates)

    start_date = dates.min()
    start_date = start_date.strftime("%Y-%m-%d")

    end_date = dates.max()
    end_date = end_date.strftime("%Y-%m-%d")

    if not trade_calendar:
        trade_calendar = get_trade_date_between('1990-01-01', '2020-12-31', return_type='str')

    dts_arr = np.array(trade_calendar)

    st_idx = np.searchsorted(dts_arr, start_date)
    et_idx = np.searchsorted(dts_arr, end_date)

    trade_dts = [parse(dt).date() for dt in dts_arr[st_idx - 1:et_idx + 2]]
    ti = datetime(2019, 1, 1, hour=cut_hour, minute=cut_minute).time()
    trade_dts = [datetime.combine(dt, ti) for dt in trade_dts]
    ts_trade_dts = pd.Series(trade_dts)
    trade_date = pd.cut(dates.astype(np.int64) // 10 ** 9,
                        bins=ts_trade_dts.astype(np.int64) // 10 ** 9)
    trade_date = pd.to_datetime([it.right for it in trade_date], unit='s')
    res = pd.to_datetime(pd.DatetimeIndex(trade_date).date)

    return res


