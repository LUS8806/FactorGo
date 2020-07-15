from pandas import DataFrame, Series
from typing import Union, List, Callable

from FactorGo.factor_test.constant import *
from FactorGo.factor_process import FactorWinsorize, FactorStandardize, FactorCodeFilter, FactorNeutralize
from FactorGo.factor_process import FactorMatchIndex, FactorNanProcess, FactorReturnMatch
from FactorGo.factor_process import FactorQuantize, FactorIndustryMatch, FactorMarketCapMatch
from FactorGo.factor_test.factor_test import FactorICTest, FactorStats
from FactorGo.factor_test.plotting import *


# TODO 整合linearmodels模块

class FactorDataStruct(object):
    """因子数据类，可以进行因子分析

    因子具有以下属性:
        * 身份属性:
            * 因子名称: factor_name

        * 数据属性:
            * 因子数据: factor_data
            * 收益率数据: forward_ret
            * 市值数据: market_cap
            * 行业分类数据: industry_cat
            * 周期: factor_freq

            @property
            * 所有涉及日期: all_dates
            * 所有涉及股票: all_sec_codes

        * 状态属性：
            * 标准化: is_standardized
            * 去极值: is_winsorized
            * 空值处理: is_na_processed
            * 中性化: is_neutralized
            * 上市日期过滤: is_filtered

    因子具有以下方法：
        * 计算与其他因子的相关性: corr_with
        * 基于指定指数计算覆盖度: coverage_stats
        * 匹配指数成分股: merge_with_index

    """
    def __init__(self,
                 factor_data: Union[Series, DataFrame],
                 factor_freq: str = None,
                 ret_periods: Union[str, List[str]] = None,
                 factor_name: str = None,
                 factor_col: Union[str, int] = None,
                 forward_ret: DataFrame = None,
                 market_cap: Series = None,
                 industry_cat: Series = None,
                 factor_quantile: Series = None,
                 init_data: bool = False):

        # 数据属性
        self.factor_data, self.factor_name = self.__factor_data(factor_data, factor_name, factor_col)
        self.factor_freq = factor_freq  # 因子频率
        self.ret_periods = ret_periods

        self.factor_quantile = pd.Series(index=self.factor_data.index) if not factor_quantile else factor_quantile   # 因子分组
        self.forward_ret = pd.DataFrame(index=self.factor_data.index) if not forward_ret else forward_ret   # 收益率数据
        self.market_cap = pd.Series(index=self.factor_data.index) if not market_cap else market_cap  # 市值数据
        self.industry_cat = pd.DataFrame(index=self.factor_data.index) if not industry_cat else industry_cat    # 行业分类

        # 状态属性
        self.is_winsorized = False
        self.is_standardized = False
        self.is_neutralized = False
        self.is_na_processed = False
        self.is_filtered = False

        self.__check_datetime_format()

        if init_data:
            self.__data_init()

    @staticmethod
    def __factor_data(factor_data: Union[Series, DataFrame],
                      factor_name: str = None,
                      factor_col: Union[str, int] = None):

        if isinstance(factor_data, DataFrame):
            if not factor_col:
                factor_col = 0
            factor_data = factor_data[factor_col]

            if factor_name:
                factor_data.name = factor_name
            else:
                factor_name = factor_data.name

        elif isinstance(factor_data, Series):
            if factor_name:
                factor_data.name = factor_name
            else:
                if not factor_data.name:
                    factor_data.name = 'factor'
                factor_name = factor_data.name
        factor_data.index.names = [DATE_COL, CODE_COL]

        return factor_data, factor_name

    def __status_init(self):
        self.is_winsorized = False
        self.is_normalized = False
        self.is_neutralized = False
        self.is_filtered = False
        self.is_na_processed = False

    def __data_init(self):
        """数据初始化"""
        self.match_cap(inplace=True)
        self.match_industry(inplace=True)
        self.match_return(inplace=True)

    def __check_index_name(self):
        self.factor_data.index.names = [DATE_COL, CODE_COL]

    def __check_datetime_format(self):
        """
        检查日期的格式
        """
        dt_index = self.factor_data.index.get_level_values(0).unique()
        if not isinstance(dt_index, pd.DatetimeIndex):
            self.factor_data.index.set_levels(pd.to_datetime(dt_index), level=0, inplace=False)

    def __align_data_index(self):
        """
        对齐index
        """
        idx = self.factor_data.index
        if not self.forward_ret.empty:
            self.forward_ret = self.forward_ret.reindex(idx)
        if not self.market_cap.empty:
            self.market_cap = self.market_cap.reindex(idx)
        if not self.industry_cat.empty:
            self.industry_cat = self.industry_cat.reindex(idx)
        if not self.factor_quantile.empty:
            self.factor_quantile = self.factor_quantile.reindex(idx)

    def update_data(self,
                    name: str,
                    data: Union[DataFrame, Series],
                    is_na_processed: Union[bool, str] = None,
                    is_winsorized: Union[bool, str] = None,
                    is_standardized: Union[bool, str] = None,
                    is_neutralized: Union[bool, str] = None,
                    is_filtered: Union[bool, int] = None):
        setattr(self, name, data)
        self.__align_data_index()
        self.is_na_processed = self.is_na_processed if is_na_processed is None else is_na_processed
        self.is_winsorized = self.is_winsorized if is_winsorized is None else is_winsorized
        self.is_standardized = self.is_standardized if is_standardized is None else is_standardized
        self.is_neutralized = self.is_neutralized if is_neutralized is None else is_neutralized
        self.is_filtered = self.is_filtered if is_filtered is None else is_filtered

    def __str__(self):
        return self.factor_data.__repr__()

    @property
    def start_date(self):
        return self.all_dates[0]

    @property
    def end_date(self):
        return self.all_dates[-1]

    @property
    def all_dates(self):
        """因子所有的日期"""
        return self.factor_data.index.get_level_values(DATE_COL).unique().tolist()

    @property
    def all_sec_codes(self):
        """因子所有覆盖的股票"""
        return self.factor_data.index.get_level_values(CODE_COL).unique().tolist()

    @property
    def quantize_num(self):
        """分组的数量"""
        if not self.factor_quantile.empty:
            return self.factor_quantile.max()

    def factor_stats(self, index_code: str = None):
        """因子数据统计"""
        tester = FactorStats()
        res_stats = tester.fit_transform(self)
        return res_stats

    def nan_process(self,
                    method: str = 'drop',
                    constant_val: float = None,
                    inplace: bool = False):
        processer = FactorNanProcess(method=method, constant_val=constant_val, inplace=inplace)
        return processer.fit_transform(self)

    def winsorize(self,
                  method: str = 'median',
                  ex_num: int = 5,
                  up: float = 0.95,
                  low: float = 0.05,
                  exec_func: Callable = None,
                  inplace: bool = False):
        """去极值"""
        handler = FactorWinsorize(method=method, ex_num=ex_num, up=up, low=low, exec_func=exec_func,
                                  inplace=inplace)

        return handler.fit_transform(self)

    def standardize(self,
                    method: str = 'z_score',
                    exec_func: Callable = None,
                    inplace: bool = False):
        """标准化"""
        handler = FactorStandardize(method=method, exec_func=exec_func, inplace=inplace)
        return handler.fit_transform(self)

    def neutralize(self,
                   factors: Union[DataFrame, list] = None,
                   inplace: bool = False):
        """中性化"""

        handler = FactorNeutralize(base_factor=factors, inplace=inplace)
        return handler.fit_transform(self)

    def match_cap(self, inplace: bool = False):
        """匹配市值数据"""
        cap_match = FactorMarketCapMatch(inplace=inplace)
        return cap_match.fit_transform(self)

    def match_industry(self,
                       industry: str = 'sw',
                       inplace: bool = False):
        """匹配行业"""
        ind_match = FactorIndustryMatch(industry=industry, inplace=inplace)
        return ind_match.fit_transform(self)

    def match_return(self,
                     periods: Union[str, List[str]] = None,
                     price_type: str = 'close',
                     if_exists: str = 'append',
                     inplace: bool = False):
        """
        Forward收益率
        Parameters
        ----------
        periods
        price_type
        if_exists
        inplace

        Returns
        -------

        """
        periods = self.ret_periods if not periods else periods

        ret_match = FactorReturnMatch(periods=periods,
                                      price_type=price_type,
                                      if_exists=if_exists,
                                      inplace=inplace)
        return ret_match.fit_transform(self)

    def corr_with(self, other_factor, method='spearman'):
        """计算与其他因子的相关性
           返回相关性序列及均值
        """
        if isinstance(other_factor, FactorDataStruct):
            other_factor = other_factor.factor_data

        all_data = pd.merge(self.factor_data, other_factor, left_index=True, right_index=True)
        all_corr = all_data.groupby(level=0).apply(lambda x: x.corr(method=method).iloc[0, 1])
        return all_corr, all_corr.mean()

    def merge_with_index(self, index_code, drop_na=True, inplace=False):
        """与指数的成分股对齐"""
        index_match = FactorMatchIndex(index_code=index_code, drop_na=drop_na, inplace=inplace)
        return index_match.fit_transform(self)

    def filter_with_list_days(self, days=60, inplace=False):
        """根据上市天数过滤"""
        days_filter = FactorCodeFilter(days_threshold=days, inplace=inplace)
        return days_filter.fit_transform(self)

    def factor_stats(self, plot=True):
        """因子统计，直方图等"""

        pass

    def ic_test(self, ret_period=None, by_group=None, plot=True):
        """计算ic
        返回IC分析的数据对象
        """
        if by_group == INDUSTRY_COL:
            by_group = self.industry_cat.copy()
        ic_test = FactorICTest(by_group=by_group, ret_period=ret_period)
        test_result = ic_test.fit_transform(self)
        if plot:
            test_result.plot()
        # 格式化输出ic测试的结果
        return test_result

    def quantize(self, quantiles=5, bins=None, by_group=None, no_raise=False, zero_aware=False, inplace=False):
        """分组"""
        quantizer = FactorQuantize(quantiles=quantiles,
                                   bins=bins,
                                   by_group=by_group,
                                   no_raise=no_raise,
                                   zero_aware=zero_aware,
                                   inplace=inplace
                                   )

        return quantizer.fit_transform(self)

    def regress_test(self, ret_period=None):
        """回归测试"""
        pass

    def quantize_test(self, ret_period=None, mode='vector'):
        """分层测试"""
        tester = FactorQuantize

    def one_click_test(self):
        """
        一键测试
        Returns
        -------

        """
        pass

