from __future__ import annotations

import copy
import abc
import statsmodels.api as sm
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Callable
from FactorGo.factor_test.util import *
from functools import partial
from datetime import timedelta
from FactorGo.data_loader import data_api, BaseDataLoader
from sklearn.impute import SimpleImputer
from copy import deepcopy
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from FactorGo.factor_base import FactorDataStruct


class FactorProcess(BaseEstimator, TransformerMixin, metaclass=abc.ABCMeta):
    _data_loader = data_api

    def __init__(self, data_loader: BaseDataLoader = None, inplace=False):
        if data_loader is not None:
            self._data_loader = data_loader
        self._inplace = inplace

    def fit(self, factor_struct):
        return self

    @abc.abstractmethod
    def transform(self, factor_struct: FactorDataStruct):
        ...


class FactorMatchIndex(FactorProcess):
    """因子匹配指数成分股"""

    def __init__(self,
                 data_loader: BaseDataLoader = None,
                 index_code: str = None,
                 drop_na: bool = True,
                 inplace: bool = False):
        """

        Parameters
        ----------
        data_loader: BaseDataLoader, 数据加载器
        index_code: str, 指数代码
        drop_na: bool, 未匹配的股票是否删除
        inplace: bool, 默认False, 返回新的FactorDataStruct, 为'True'则在原有的FactorDataStruct上更新数据
        """
        super().__init__(data_loader, inplace)
        self.index_code = index_code
        self._drop_na = drop_na

    def transform(self, factor_struct: FactorDataStruct) -> Union[None, FactorDataStruct]:
        index_component = self._data_loader.get_index_components(
            index_code=self.index_code,
            dates=factor_struct.all_dates).set_index([DATE_COL, CODE_COL])

        merge_data = pd.merge(factor_struct.factor_data,
                              index_component,
                              how='right',
                              left_index=True,
                              right_index=True)
        if self._drop_na:
            merge_data = merge_data.dropna()
        if self._inplace:
            factor_struct.update_data('factor_data', merge_data, is_na_processed=self._drop_na)
        else:
            factor_struct_cp = copy.deepcopy(factor_struct)
            factor_struct_cp.update_data('factor_data', merge_data, is_na_processed=self._drop_na)
            return factor_struct_cp


class FactorNanProcess(FactorProcess):
    """缺失值处理"""

    def __init__(self,
                 data_loader: BaseDataLoader = None,
                 method: str = 'drop',
                 constant_val: float = None,
                 inplace: bool = False):
        """

        Parameters
        ----------
        data_loader: BaseDataLoader, 数据加载器
        method: str, 缺失值处理的方法, ['drop', 'mean', 'median', 'most_frequent', 'constant']
        constant_val: float, 当method='constant'时，设定的固定的值
        inplace: bool, 默认False, 返回新的FactorDataStruct, 为'True'则在原有的FactorDataStruct上更新数据
        """
        super().__init__(data_loader, inplace)
        self._method = method
        self._constant_val = constant_val

    def transform(self, factor_struct: FactorDataStruct) -> Union[None, FactorDataStruct]:
        factor_data = factor_struct.factor_data.copy()
        if self._method == 'drop':
            factor_data = factor_data.dropna()
        if self._method in ['mean', 'median', 'most_frequent', 'constant']:
            if self._method == 'constant' and self._constant_val is None:
                raise ValueError("Must set a value for constant_val when method is 'constant'!")

            def _nan_process(df):
                imputer = SimpleImputer(strategy=self._method)
                imputer.fit(df)
                return imputer.transform(df)

            factor_data = factor_data.groupby(level=0).apply(_nan_process)

        if self._inplace:
            factor_struct.update_data('factor_data', factor_data, is_na_processed=True)
        else:
            factor_struct_cp = copy.deepcopy(factor_struct)
            factor_struct_cp.update_data('factor_data', factor_data, is_na_processed=True)
            return factor_struct_cp


class FactorWinsorize(FactorProcess):
    """因子去极值处理"""
    exec_func_dict = {
        'std': df_winsorize_std,
        'median': df_winsorize_mad,
        'quantile': df_winsorize_quantile
    }

    def __init__(self,
                 data_loader: BaseDataLoader = None,
                 method: str = 'median',
                 ex_num: int = 5,
                 up: float = 0.95,
                 low: float = 0.05,
                 exec_func: Callable = None,
                 inplace: bool = False):
        """
        Parameters
        ----------
        data_loader: BaseDataLoader, 数据加载器
        method: str, 极值处理的方法, 可以是'std', 'median' 或'quantile'
        ex_num: int, 默认ex_num=5, 极端值的判定范围, 当method为std，median时有效
        up: float, 默认up=0.95, method为percentile时有效
        low: float, 默认low=0.05, method为percentile时有效
        exec_func: 自定义处理函数
        inplace: bool, 默认False, 返回新的FactorDataStruct, 为'True'则在原有的FactorDataStruct上更新数据
        """
        super().__init__(data_loader, inplace)
        self._method = method
        self._ex_num = ex_num

        if exec_func:
            self._exec_func = exec_func
        else:
            if method in ['std', 'median']:
                self._exec_func = partial(self.exec_func_dict[method], ex_num=ex_num)
            elif method == 'quantile':
                self._exec_func = partial(self.exec_func_dict[method], up=up, low=low)
            else:
                raise ValueError("Method can only be one of 'std', 'median' or 'quantile'")

    def transform(self, factor_struct: FactorDataStruct) -> Union[None, FactorDataStruct]:

        factor_data = factor_struct.factor_data.copy()
        factor_data = factor_data.groupby(level=DATE_COL, sort=True).apply(self._exec_func)

        if self._inplace:
            factor_struct.update_data('factor_data', factor_data, is_winsorized=True)
        else:
            factor_struct_cp = copy.deepcopy(factor_struct)
            factor_struct_cp.update_data('factor_data', factor_data, is_winsorized=True)
            return factor_struct_cp


class FactorStandardize(FactorProcess):
    """因子去标准化处理 """

    exec_func_dict = {
        'z_score': df_standardize_norm
    }

    def __init__(self,
                 data_loader: BaseDataLoader = None,
                 method: str = 'z_score',
                 exec_func: Callable = None,
                 inplace: bool = False):
        """
        Parameters
        ----------
        data_loader: BaseDataLoader, 数据加载器
        method: str, 默认method='z_score', 标准化的方法
        exec_func: 自定义处理函数
        inplace: bool, 默认False, 返回新的FactorDataStruct, 为'True'则在原有的FactorDataStruct上更新数据
        """
        super().__init__(data_loader, inplace)
        self._method = method

        if exec_func:
            self._exec_func = exec_func
        else:
            if method in ['z_score']:
                self._exec_func = self.exec_func_dict[method]
            else:
                raise ValueError("Method can only be one of 'z_score', 'median' or 'percentile'")

    def transform(self, factor_struct: FactorDataStruct) -> Union[None, FactorDataStruct]:
        factor_data = factor_struct.factor_data.copy()
        factor_data = factor_data.groupby(level=DATE_COL, sort=True).apply(self._exec_func)

        if self._inplace:
            factor_struct.update_data('factor_data', factor_data, is_standardized=self._method)
        else:
            factor_struct_cp = copy.deepcopy(factor_struct)
            factor_struct_cp.update_data('factor_data', factor_data, is_standardized=self._method)
            return factor_struct_cp


class FactorCodeFilter(FactorProcess):
    """股票过滤期，根据上市时间过滤"""

    def __init__(self,
                 data_loader: BaseDataLoader = None,
                 days_threshold: int = 40,
                 inplace: bool = False):
        """

        Parameters
        ----------
        data_loader: BaseDataLoader, 数据加载器
        days_threshold: int, 默认40，去掉上市未满这个天数的股票
        inplace: bool, 默认False, 返回新的FactorDataStruct, 为'True'则在原有的FactorDataStruct上更新数据
        """
        super().__init__(data_loader, inplace)
        self.days_threshold = days_threshold

    def transform(self, factor_struct: FactorDataStruct) -> Union[None, FactorDataStruct]:
        sec_info = self._data_loader.get_stock_list_days(sec_codes=factor_struct.all_sec_codes)
        df_factor = factor_struct.factor_data.copy().reset_index()
        days_delta = timedelta(days=self.days_threshold)
        df_factor_list = pd.merge(df_factor, sec_info, how='left')
        list_con = (df_factor_list[DATE_COL] - df_factor_list[LIST_DATE_COL]) >= days_delta
        dlist_con = df_factor_list[DATE_COL] < df_factor_list[DLIST_DATE_COL]

        # 数据过滤与处理
        df_factor = df_factor.loc[list_con & dlist_con].reset_index(drop=True)
        df_factor = df_factor.set_index([DATE_COL, CODE_COL])

        if self._inplace:
            factor_struct.update_data('factor_data', df_factor, is_filtered=self.days_threshold)
        else:
            factor_struct_cp = copy.deepcopy(factor_struct)
            factor_struct_cp.update_data('factor_data', df_factor, is_filtered=self.days_threshold)
            return factor_struct_cp


class FactorNeutralize(FactorProcess):
    """因子中性化处理, 中性化处理之前因子都已经标准化处理过"""
    def __init__(self,
                 data_loader: BaseDataLoader = None,
                 base_factor: Union[list, DataFrame] = None,
                 industry: str = None,
                 normalize: bool = True,
                 inplace: bool = False):
        """
        Parameters
        ----------
        data_loader: BaseDataLoader, 数据加载器
        base_factor: BaseFactorData, 基于base_factor做中性化处理, 默认为None时做市值行业中性化处理
        normalize: bool, 默认True, 中性化之后是否需要标准化处理
        inplace: bool, 默认False, 返回新的FactorDataStruct, 为'True'则在原有的FactorDataStruct上更新数据
        """
        super().__init__(data_loader, inplace)
        self._normalize = normalize
        self._industry = industry if industry else 'sw'
        self.base_factor = base_factor

    def transform(self, factor_struct: FactorDataStruct) -> Union[None, FactorDataStruct]:
        tar_df_factor = factor_struct.factor_data.copy()
        tar_col = [factor_struct.factor_name]
        base_cols = []

        base_factor = self.base_factor

        if base_factor is None:
            factor_struct_cp = deepcopy(factor_struct)
            if factor_struct_cp.industry_cat.empty or self._industry not in factor_struct_cp.industry_cat:
                factor_struct_cp.match_industry(industry=self._industry, inplace=True)

            if factor_struct_cp.market_cap.empty:
                factor_struct_cp.match_cap(inplace=True)

            factor_data_gp = factor_struct_cp.industry_cat[self._industry].groupby(level=0, sort=True)
            ind_data = factor_data_gp.apply(to_dummy_variable)
            base_factor = pd.merge(ind_data, np.log(factor_struct_cp.market_cap),
                                   right_index=True, left_index=True, how='left')

        if isinstance(base_factor, list):
            for fac in base_factor:
                if isinstance(fac, FactorDataStruct):
                    to_merge = fac.factor_data
                    base_col = fac.factor_name
                elif isinstance(fac, pd.DataFrame):
                    to_merge = fac
                    base_col = fac.columns.tolist()
                else:
                    raise ValueError("Base Factors can only be DataFrame of FactorDataStruct")

                tar_df_factor = pd.merge(
                    tar_df_factor,
                    to_merge,
                    how='left', left_index=True, right_index=True)
                if isinstance(base_col, str):
                    base_col = [base_col]
                base_cols.extend(base_col)

        elif isinstance(base_factor, pd.DataFrame):
            tar_df_factor = pd.merge(tar_df_factor, base_factor, how='left', left_index=True, right_index=True)
            base_cols = base_factor.columns.tolist()

        else:
            raise ValueError("base_factor can only be list or DataFrame!")

        def _neutralize(df):
            df = df.reset_index(level=0, drop=True)
            df_tar_temp = df[tar_col]
            df_base = df[base_cols]

            try:
                model = sm.OLS(df_tar_temp, df_base, missing='drop')
                results = model.fit()
            except Exception as e:
                print(e)
                return pd.DataFrame()

            residual = results.resid.reindex(df.index)

            if self._normalize:
                residual = (residual - residual.mean()) / residual.std()
            return residual

        all_fac_data_gp = tar_df_factor.groupby(level=0)
        all_residual = all_fac_data_gp.apply(_neutralize)

        if self._inplace:
            factor_struct.update_data('factor_data', all_residual, is_neutralized=base_cols)
        else:
            factor_struct_cp = deepcopy(factor_struct)
            factor_struct_cp.update_data('factor_data', all_residual, is_neutralized=base_cols)
            return factor_struct_cp


class FactorQuantize(FactorProcess):
    """因子分组"""

    def __init__(self,
                 data_loader: BaseDataLoader = None,
                 quantiles: Union[int, list] = 5,
                 bins: Union[int, list] = None,
                 by_group: Series = None,
                 no_raise: bool = False,
                 zero_aware: bool = False,
                 inplace: bool = False):
        """
        Parameters
        ----------
        data_loader: BaseDataLoader, 数据加载器
        quantiles: int or list of float, 根据分位数分层, 设定为整数，表示等分为N组；也可以用数组表示具体的分割分位数。
        bins: list of float, 表示按具体的因子数值分割
        by_group: Series, 组内进行分层
        no_raise: bool, 默认False, 设置为True时, 发生错误时, 值设定为nan
        zero_aware: bool, 默认False, 根据因子值的正负分别进行分组
        inplace: bool, 默认False, 返回新的FactorDataStruct, 为'True'则在原有的FactorDataStruct上更新数据
        """
        super().__init__(data_loader, inplace)
        if quantiles and bins:
            raise ValueError("quantiles和bins同时只能指定一个参数")

        self._quantiles = quantiles
        self._bins = bins
        self._by_group = by_group
        self._no_raise = no_raise
        self._zero_aware = zero_aware

    @property
    def quantize_num(self):
        """分组的数量"""
        if isinstance(self._quantiles, int):
            quantize_num = self._quantiles
        elif isinstance(self._quantiles, list):
            quantize_num = len(self._quantiles) - 1
        elif isinstance(self._bins, list):
            quantize_num = len(self._bins) - 1
        else:
            quantize_num = None
        return quantize_num

    def transform(self, factor_struct: FactorDataStruct) -> Union[None, FactorDataStruct]:
        factor_data = factor_struct.factor_data.copy()
        factor_quantile = quantize_factor(factor_data,
                                          factor_name=factor_struct.factor_name,
                                          quantiles=self._quantiles,
                                          bins=self._bins,
                                          by_group=self._by_group,
                                          no_raise=self._no_raise,
                                          zero_aware=self._zero_aware)

        if self._inplace:
            factor_struct.update_data('factor_quantile', factor_quantile)

        else:
            factor_struct_cp = copy.deepcopy(factor_struct)
            factor_struct_cp.update_data('factor_quantile', factor_quantile)
            return factor_struct_cp


class FactorReturnMatch(FactorProcess):
    """匹配股票下期收益率"""
    def __init__(self,
                 data_loader: BaseDataLoader = None,
                 periods: Union[str, List[str]] = None,
                 price_type: str = 'close',
                 if_exists: str = 'append',
                 inplace: bool = False):
        """
        Parameters
        ----------
        data_loader: BaseDataLoader, 数据加载器
        periods: list of str, 如果periods为None, 则根据factor_struct的factor_freq匹配
        price_type: str, 计算收益率的价格类型，'open' 或 'close'
        if_exists: str, 默认'append', 如果已有收益率数据, 'append'添加新收益率数据, 'replace'替换
        inplace: bool, 默认False, 返回新的FactorDataStruct, 为'True'则在原有的FactorDataStruct上更新数据
        """
        super().__init__(data_loader, inplace)
        self._periods = periods
        self._price_type = price_type
        self._if_exists = if_exists
        if isinstance(self._periods, str):
            self._periods = [self._periods]

    def transform(self, factor_struct: FactorDataStruct) -> Union[None, FactorDataStruct]:
        forward_ret = self._data_loader.get_stock_return(
            codes_date_index=factor_struct.factor_data.index,
            periods=self._periods)

        origin_forward = factor_struct.forward_ret.copy()

        if self._if_exists == 'append' and not origin_forward.empty:
            for col in origin_forward:
                if col not in forward_ret:
                    forward_ret[col] = origin_forward[col]

        if self._inplace:
            factor_struct.update_data('forward_ret', forward_ret)
        else:
            factor_struct_cp = copy.deepcopy(factor_struct)
            factor_struct_cp.update_data('forward_ret', forward_ret)
            return factor_struct_cp

    def set_period(self, periods: Union[str, List[str]]):
        self._periods = periods


class FactorIndustryMatch(FactorProcess):
    """匹配股票行业代码"""
    def __init__(self,
                 data_loader: BaseDataLoader = None,
                 industry: str = 'sw',
                 if_exists: str = None,
                 inplace: bool = False):
        """

        Parameters
        ----------
        data_loader: BaseDataLoader, 数据加载器
        industry: str, 默认'sw', 股票匹配行业
        if_exists: str, 默认'append', 如果已有收益率数据, 'append'添加新收益率数据, 'replace'替换
        inplace: bool, 默认False, 返回新的FactorDataStruct, 为'True'则在原有的FactorDataStruct上更新数据
        """

        super().__init__(data_loader, inplace)
        self._industry = industry
        self._if_exists = if_exists

    def transform(self, factor_struct: FactorDataStruct) -> Union[None, FactorDataStruct]:

        stock_industry = self._data_loader.get_stock_industries(
            codes_date_index=factor_struct.factor_data.index,
            industry=self._industry)

        origin_industry = factor_struct.industry_cat.copy()

        if self._if_exists == 'append' and not origin_industry.empty:
            for col in origin_industry:
                if col not in stock_industry:
                    stock_industry[col] = origin_industry[col]

        if self._inplace:
            factor_struct.update_data('industry_cat', stock_industry)

        else:
            factor_struct_cp = copy.deepcopy(factor_struct)
            factor_struct_cp.update_data('industry_cat', stock_industry)
            return factor_struct_cp


class FactorMarketCapMatch(FactorProcess):
    """匹配市值"""
    def __init__(self,
                 data_loader: BaseDataLoader = None,
                 inplace: bool = False):
        """

        Parameters
        ----------
        data_loader: BaseDataLoader, 数据加载器
        inplace: bool, 默认False, 返回新的FactorDataStruct, 为'True'则在原有的FactorDataStruct上更新数据
        """
        super().__init__(data_loader, inplace)

    def transform(self, factor_struct: FactorDataStruct) -> Union[None, FactorDataStruct]:

        market_cap = self._data_loader.get_stock_cap(
            codes_date_index=factor_struct.factor_data.index)['market_cap']

        if self._inplace:
            factor_struct.update_data('market_cap', market_cap)
        else:
            factor_struct_cp = copy.deepcopy(factor_struct)
            factor_struct_cp.update_data('market_cap', market_cap)
            return factor_struct_cp
