from __future__ import annotations
import abc
import statsmodels.api as sm
import backtrader as bt

from collections import OrderedDict
from typing import List, TYPE_CHECKING
from copy import deepcopy

from pyfolio.timeseries import perf_stats

from sklearn.base import BaseEstimator, TransformerMixin
from FactorGo.factor_test.util import *
from FactorGo.data_loader import data_api, BaseDataLoader
from FactorGo.factor_test.constant import *
from FactorGo.factor_test.plotting import *
from FactorGo.factor_test.bt_rebalance_strategy import PortfolioRebalanceStrategy, AmountPandasFeed
from DataGo.fetch import estimate_freq
from FactorGo.factor_process import FactorMatchIndex

if TYPE_CHECKING:
    from FactorGo.factor_base import FactorDataStruct


class FactorTest(BaseEstimator, TransformerMixin, metaclass=abc.ABCMeta):
    _data_loader = data_api

    def __init__(self, data_loader: BaseDataLoader = None):
        if data_loader is not None:
            self._data_loader = data_loader

    def fit(self, factor_struct):
        return self

    @abc.abstractmethod
    def transform(self, factor_struct: FactorDataStruct):
        ...


class ICTestStruct(object):

    # TODO ic_decay 分组计算

    def __init__(self, ic_series,
                 ic_series_gp=None,
                 ic_decay=None,
                 factor_name=None,
                 by_group=None):
        self.factor_name = factor_name
        self.ic_series = ic_series
        self.ic_series_gp = ic_series_gp
        self.ic_decay = ic_decay
        self.by_group = by_group

    def ic_mean(self, by_group=False):
        if by_group:
            return self.ic_series_gp.mean(level=self.by_group)
        else:
            return self.ic_series.mean()

    def ic_std(self, by_group=False):
        if by_group:
            return self.ic_series_gp.std(level=self.by_group)
        else:
            return self.ic_series.std()

    def ic_ir(self, by_group=False):
        return self.ic_mean(by_group) / self.ic_std(by_group)

    def ic_ratio(self, value=0, by_group=False):
        if by_group:
            return self.ic_series_gp.groupby(level=self.by_group).apply(lambda x: (x > value).sum() / len(x))
        else:
            return (self.ic_series > value).sum() / len(self.ic_series)

    def ic_cum(self, by_group=False):
        if by_group:
            return self.ic_series_gp.cumsum(level=self.by_group)
        else:
            return self.ic_series.cumsum()

    def ic_half_decay(self):
        ...

    def plot(self, show=True):
        fig = plot_ic_test(self)
        if show:
            fig.show()
        return fig


class RegressTestStruct(object):

    def __init__(self,
                 factor_return,
                 t_series,
                 factor_return_gp=None,
                 t_series_gp=None,
                 by_group=None,
                 factor_name=None):

        self.factor_name = factor_name if factor_name else 'factor'
        self.factor_return = factor_return
        self.t_series = t_series
        self.t_series_gp = t_series_gp
        self.factor_return_gp = factor_return_gp
        self.by_group = by_group

    def tvalue_ratio(self, value=0, by_group=False, abs_value=True):

        df = self.t_series.copy() if not by_group else self.t_series_gp.copy()
        if abs_value:
            df = df.abs()

        if by_group:
            return df.groupby(level=self.by_group).apply(lambda x: (x > value).sum() / len(x))
        else:
            return (df > value).sum() / len(df)

    def tvalue_mean(self, by_group=False, abs_value=True):
        df = self.t_series.copy() if not by_group else self.t_series_gp.copy()
        if abs_value:
            df = df.abs()
        return df.mean()

    def factor_return_mean(self, by_group=False):
        if by_group:
            return self.factor_return_gp.mean()
        else:
            return self.factor_return.mean()


class FactorStatsStruct(object):
    ...


class QuantizeTestStruct(object):
    """分层收益测试数据"""

    def __init__(self, quantile_return, freq_days='1d', factor_name=None, benchmark=None, long_q=None, short_q=None,):

        self.quantile_return = quantile_return.copy()
        long_q = self.quantile_return.columns.max() if not long_q else long_q
        short_q = self.quantile_return.columns.min() if not short_q else short_q
        ls_ret = self.quantile_return[long_q] - self.quantile_return[short_q]
        self.quantile_return.columns = ['Q{}'.format(int(q)) for q in self.quantile_return.columns]
        self.quantile_return['Q_LS'] = ls_ret

        self.freq_days = freq_days
        self.factor_name = factor_name
        self._benchmark = benchmark

    @property
    def quantile_mean_return(self):
        """分组平均收益"""
        return self.quantile_return.mean()

    @property
    def quantile_cum_return(self):
        return (1+self.quantile_return).cumprod()

    @property
    def return_stats(self):
        ret = pd.DataFrame()
        for col in self.quantile_return:
            if self.freq_days == '1d':
                ret[col] = perf_stats(self.quantile_return[col])

        return ret

    def plot(self, show=True):
        fig = plot_quantize_test(self)
        if show:
            fig.show()
        return fig


class FactorStats(FactorTest):
    """
    因子基础统计
    # 因子覆盖度
    # 因子相关性
    """

    def __init__(self, cov_index_codes: Union[list, str] = None,
                 corr_base_factors: Union[list, DataFrame] = None):
        self._cov_index_codes = cov_index_codes
        self._corr_base_factors = corr_base_factors

    def transform(self, factor_struct: FactorDataStruct) -> Union[None, FactorStatsStruct]:
        ...

    @staticmethod
    def _cover_ratio(factor_struct, index_code):
        """因子数据统计"""
        index_match = FactorMatchIndex(index_code=index_code, inplace=False, drop_na=False)
        new_fac = index_match.fit_transform(factor_struct)
        cov_ratio = new_fac.factor_data.groupby(level=0)[factor_struct.factor_name].apply(lambda x: len(x.dropna()) / len(x))
        return cov_ratio


class FactorCorrelation(FactorTest):
    """因子相关性分析"""

    def __init__(self,
                 base_factor: Union[list, DataFrame] = None):

        self.base_factor = base_factor

    def transform(self, factor_struct: FactorDataStruct = None) -> DataFrame:
        base_factor = deepcopy(self.base_factor)
        tar_df_factor = factor_struct.factor_data.copy()
        fac_col = factor_struct.factor_name

        if isinstance(base_factor, list):
            for fac in base_factor:
                if isinstance(fac, FactorDataStruct):
                    to_merge = fac.factor_data
                elif isinstance(fac, pd.DataFrame):
                    to_merge = fac
                else:
                    raise ValueError("Base Factors can only be DataFrame of FactorDataStruct")

                tar_df_factor = pd.merge(
                    tar_df_factor,
                    to_merge,
                    how='left', left_index=True, right_index=True)

        elif isinstance(base_factor, pd.DataFrame):
            tar_df_factor = pd.merge(tar_df_factor, base_factor, how='left', left_index=True, right_index=True)
        else:
            raise ValueError("base_factor can only be list or DataFrame!")

        corr_series = tar_df_factor.groupby(level=DATE_COL).apply(lambda x: x.corr()[fac_col])

        return corr_series


class FactorICTest(FactorTest):
    """IC测试"""

    def __init__(self,
                 data_loader: BaseDataLoader = None,
                 by_group: Union[str, DataFrame] = None,
                 group_adjust: bool = False,
                 ret_period: str = None):
        """
        因子IC测试

        Parameters
        ----------
        data_loader: BaseDataLoader, 数据加载器
        by_group:
        group_adjust
        ret_period: str, 收益率的周期
        """
        super().__init__(data_loader)
        self._group_adjust = group_adjust
        self._by_group = by_group
        self._ret_period = ret_period

    def transform(self, factor_struct: FactorDataStruct) -> ICTestStruct:
        factor_struct_cp = deepcopy(factor_struct)
        factor_data = factor_struct_cp.factor_data.copy()

        # 收益率数据
        if not self._ret_period:
            ret_name = NEXT_RET_PREFIX + 'default'
        else:
            ret_name = NEXT_RET_PREFIX + self._ret_period

        if ret_name not in factor_struct_cp.forward_ret:
            factor_struct_cp.match_return(periods=self._ret_period, if_exists='append', inplace=True)

        ret_data = factor_struct_cp.forward_ret[ret_name]

        # 不分组IC
        ic_data = factor_ic(factor_data,
                            ret_data,
                            group_adjust=self._group_adjust,
                            by_group=None)

        # 分组IC
        ic_data_gp = None
        by_group = None

        if self._by_group is not False:
            if self._by_group is None or self._by_group in ['sw']:
                if factor_struct_cp.industry_cat.empty:
                    factor_struct_cp.match_industry(inplace=True)
                    by_group = factor_struct_cp.industry_cat['sw']
                else:
                    by_group = factor_struct_cp.industry_cat['sw']
            else:
                by_group = self._by_group

            ic_data_gp = factor_ic(factor_data,
                                   ret_data,
                                   group_adjust=self._group_adjust,
                                   by_group=by_group)

        # ic_decay = self._ic_time_decay(factor_struct)
        # ic_decay[0] = ic_data.mean()

        ic_res = ICTestStruct(ic_series=ic_data,
                              ic_series_gp=ic_data_gp,
                              # ic_decay=ic_decay,
                              factor_name=factor_struct.factor_name,
                              by_group=by_group.name)

        return ic_res

    def _ic_time_decay(self, factor_struct, decay_days=240, by_group=None):
        decay_dct = OrderedDict()
        all_dts = factor_struct.all_dates
        factor_freq = factor_struct.factor_freq
        factor_data = factor_struct.factor_data
        if factor_freq is None:
            factor_freq = estimate_freq(factor_struct.all_dates)

        last_date = self._data_loader.get_trade_date_offset(all_dts[-1], factor_freq)
        last_date = self._data_loader.get_trade_date_offset(last_date, decay_days)
        price_data = self._data_loader.get_stock_price(sec_codes=factor_struct.all_sec_codes,
                                                        start_date=all_dts[0],
                                                        end_date=last_date,
                                                        fields=['close'])

        for i in range(1, decay_days+1):
            st_dts = self._data_loader.get_trade_date_offset(all_dts, i)
            ed_dts = self._data_loader.get_trade_date_offset(st_dts, factor_freq)
            nan_idx = np.isnan(ed_dts)
            st_dts, ed_dts, all_dts = st_dts[nan_idx], ed_dts[nan_idx], all_dts[nan_idx]
            st_dct = dict(zip(st_dts, all_dts))
            ed_dct = dict(zip(ed_dts, all_dts))
            st_price = price_data.loc[st_dts]
            st_price = st_price.rename(st_dct, axis=0)
            et_price = price_data.loc[ed_dts]
            et_price = et_price.rename(ed_dct, axis=0)
            ret_data = et_price.divide(st_price)
            temp_factor = factor_data.loc[all_dts, :]

            ic_data = factor_ic(temp_factor,
                                ret_data,
                                group_adjust=self._group_adjust,
                                by_group=by_group)

            decay_dct[i] = ic_data.mean()

        return decay_dct


class FactorRegressTest(FactorTest):
    """回归测试: 1. 收益率与因子回归（[可选]在回归模型中加入行业和市值因子）
                2. 可以单独测试一个因子，也可以用于多个因子同时进行回归测试
                3. *默认不需要进行数据加载与读取
                4. 不改变原先因子结构
    """

    def __init__(self,
                 data_loader: BaseDataLoader = None,
                 ret_period: str = None,
                 base_factor: Union[list, DataFrame] = None,
                 by_group: Union[str, DataFrame] = None):
        """

        Parameters
        ----------
        data_loader: function, 数据加载器
        ret_period: str
        base_factor: BaseFactorData, 基于base_factor做中性化处理
        by_group
        """
        super().__init__(data_loader)
        self._by_group = by_group
        self.base_factor = base_factor
        self.ret_period = ret_period

    def transform(self, factor_struct: FactorDataStruct):

        tar_df_factor = factor_struct.factor_data.copy()
        fac_col = factor_struct.factor_name

        if not self.ret_period:
            ret_col = NEXT_RET_PREFIX + 'default'
        else:
            ret_col = NEXT_RET_PREFIX + self.ret_period

        if ret_col in factor_struct.forward_ret:
            ret_data = factor_struct.forward_ret[[ret_col]]
        else:
            temp_struct = factor_struct.match_return(periods=self.ret_period, inplace=False)
            ret_data = temp_struct.forward_ret[[ret_col]]

        # 如果 base_factor为空，则使用factor_struct的行业和市值
        base_factor = deepcopy(self.base_factor)

        if base_factor is None:
            factor_struct_cp = deepcopy(factor_struct)
            if factor_struct_cp.market_cap.empty:
                factor_struct_cp.match_cap(inplace=True)
            if factor_struct_cp.industry_cat.empty:
                factor_struct_cp.match_industry(inplace=True)
            ind_exposure = factor_struct_cp.industry_cat['industry'].groupby(level=0).apply(to_dummy_variable)
            base_factor = pd.merge(factor_struct_cp.market_cap, ind_exposure, left_index=True, right_index=True)

        if isinstance(base_factor, list):
            for fac in base_factor:
                if isinstance(fac, FactorDataStruct):
                    to_merge = fac.factor_data
                elif isinstance(fac, pd.DataFrame):
                    to_merge = fac
                else:
                    raise ValueError("Base Factors can only be DataFrame of FactorDataStruct")

                tar_df_factor = pd.merge(
                    tar_df_factor,
                    to_merge,
                    how='left', left_index=True, right_index=True)

        elif isinstance(base_factor, pd.DataFrame):
            tar_df_factor = pd.merge(tar_df_factor, base_factor, how='left', left_index=True, right_index=True)
        else:
            raise ValueError("base_factor can only be list or DataFrame!")

        base_cols = tar_df_factor.columns.tolist()

        all_df = pd.merge(ret_data, tar_df_factor, left_index=True, right_index=True, how='left')

        def _regress(df):
            """对下期收益率与因子（及base因子如行业市值）做回归"""
            df = df.dropna()
            if df.empty:
                return np.nan
            model = sm.OLS(df[ret_col], df[base_cols])
            results = model.fit()
            res = {'coefficient': results.params[fac_col], 't_value': results.tvalues[fac_col]}
            return res

        all_df_gp = all_df.groupby(level=DATE_COL)
        tot_res = all_df_gp.apply(_regress)
        df_coef = pd.DataFrame(tot_res).applymap(lambda x: x['coefficient'])
        df_tvalue = pd.DataFrame(tot_res).applymap(lambda x: x['t_value'])

        df_coef_gp = None
        df_tvalue_gp = None
        by_group = None
        if self._by_group is not False:
            if self._by_group is None:
                if factor_struct.industry_cat.empty:
                    temp_fac = factor_struct.match_industry(inplace=False)
                    by_group = temp_fac.industry_cat
                else:
                    by_group = factor_struct.industry_cat
            else:
                by_group = self._by_group

            group_name = by_group.columns[0]
            all_df = pd.merge(all_df, by_group, left_index=True, right_index=True)
            all_df = all_df.dropna(subset=[group_name])
            grouper = [all_df.index.get_level_values(DATE_COL), group_name]
            tot_res = all_df.groupby(grouper).apply(_regress)
            df_coef_gp = pd.DataFrame(tot_res).applymap(lambda x: x['coefficient'])
            df_tvalue_gp = pd.DataFrame(tot_res).applymap(lambda x: x['t_value'])

        regress_data = RegressTestStruct(df_coef, df_tvalue, df_coef_gp, df_tvalue_gp, by_group)

        return regress_data


class FactorQuantizeTest(FactorTest):
    """分层收益率测试

    分层收益率分析分为三种情况：
    1. 因子的频率等于持仓周期
    2. TODO 因子的频率大于持仓周期
    3. TODO 因子的频率小于持仓周期
    """

    def __init__(self,
                 data_loader: BaseDataLoader = data_api,
                 rebalance_days: str = None,
                 benchmark: str = None,
                 trade_cost: float = 0,
                 is_demean: bool = False,
                 quantiles: Union[int, List[float]] = None,
                 bins: Union[int, List[float]] = None,
                 by_group: Union[str, bool, DataFrame, Series] = None,
                 no_raise: bool = False,
                 zero_aware: bool = False,
                 quantize: bool = True,
                 test_mode: str = 'vector'  # vector/event 向量式或事件驱动回测
                 ):
        """

        Parameters
        ----------
        data_loader: BaseDataLoader, 数据加载器
        rebalance_days: 调仓周期，为None使用因子区间的间隔
        benchmark: 基准，如果为None，则以所有股票的平均收益为基准
        trade_cost: 交易成本，默认千分之二
        is_demean: 是否减去组合平均收益
        quantiles: 按quantile分组
        bins: 按因子值区间分组
        by_group: 分组测试
        no_raise
        zero_aware
        quantize: 强制分组
        """
        super().__init__(data_loader)
        self._benchmark = benchmark
        self._trade_cost = trade_cost
        self._is_demean = is_demean
        self._quantiles = quantiles
        self._bins = bins
        self._by_group = by_group
        self._no_raise = no_raise
        self._zero_aware = zero_aware
        self._quantize = quantize  # 强制重新分组
        self._price_data = None  # 用来缓存价格数据
        self._rebalance_days = rebalance_days
        self.test_mode = test_mode

    def transform(self, factor_struct: FactorDataStruct) -> QuantizeTestStruct:
        # 如果因子没有进行分组或强制重新分组，先进行分组
        if factor_struct.factor_quantile.empty:
            factor_struct = factor_struct.quantize(quantiles=self._quantiles,
                                                   bins=self._bins,
                                                   by_group=self._by_group,
                                                   no_raise=self._no_raise,
                                                   zero_aware=self._zero_aware,
                                                   inplace=False
                                                   )
        if self.test_mode == 'vector':
            ret = self._vectorize_backtest(factor_struct)

        elif self.test_mode == 'event':
            tester = FactorBackTest(data_loader=self._data_loader, trade_cost=self._trade_cost)
            ret = tester.fit_transform(factor_struct)

        else:
            raise ValueError('Invalid Parameter for test_mode!')

        quantile_test = QuantizeTestStruct(quantile_return=ret,
                                           factor_name=factor_struct.factor_name,
                                           benchmark=self._benchmark)

        return quantile_test

    @staticmethod
    def _quantile_to_portfolio_target(factor_quantile: DataFrame) -> dict:
        """TODO 根据不同的参数设置返回目标组合权重
           TODO 路径依赖情况下的因子测试
        """
        port_weight = OrderedDict()
        for (dt, q), df in factor_quantile.groupby(['trade_date', factor_quantile]):
            df = df.reset_index(level=0, drop=True)
            if q not in port_weight:
                port_weight[q] = OrderedDict()
            port_weight[q][dt] = dict(zip(df.index, [1/len(df)]*len(df)))

        return port_weight

    def _vectorize_backtest(self, factor_struct: FactorDataStruct) -> DataFrame:
        """向量式回测"""
        st = factor_struct.all_dates[0]
        et = factor_struct.all_dates[-1]
        offset = estimate_freq(factor_struct.all_dates)
        et = self._data_loader.get_trade_date_offset(et, freq=offset)
        all_dts = self._data_loader.get_trade_date_between(st, et, return_type='datetime')
        if self._price_data is None:
            print('Loading Price Data...')
            self._price_data = self._data_loader.get_stock_price(sec_codes=factor_struct.all_sec_codes,
                                                                   start_date=st,
                                                                   end_date=et,
                                                                   fields=['close'])
            self._price_data = self._price_data.reset_index().pivot(index=DATE_COL, columns=CODE_COL, values='close')
            print('Price Data Loaded...')

        portfolio_tar = self._quantile_to_portfolio_target(factor_struct.factor_quantile)

        def _cum_ret_for_q(_portfolio_tar):
            """TODO 考虑交易成本；权重，当前默认等权
            """
            nvq = OrderedDict()
            nv = 1
            # _nv = 1
            st_price = None
            st_codes = None
            cur_price = None
            st_weight = None
            st_nv = None

            for dt in all_dts:
                if st_codes is not None:
                    cur_price = self._price_data.loc[dt, st_codes]

                if st_price is not None:
                    nv = (cur_price/st_price).mean()*st_nv

                if dt in _portfolio_tar:
                    st_codes = _portfolio_tar[dt].keys()
                    st_weight = pd.Series(_portfolio_tar[dt])
                    st_price = self._price_data.loc[dt, st_codes]
                    st_nv = nv*(1-self._trade_cost)

                nvq[dt] = nv

            return nvq

        cum_ret = {q: _cum_ret_for_q(v) for q, v in portfolio_tar.items()}
        ret = pd.DataFrame(cum_ret).pct_change().fillna(0)

        return ret


class FactorBackTest(FactorTest):
    """利用Backtrader回测"""

    def __init__(self,
                 data_loader: BaseDataLoader = data_api,
                 initial_capital: float = None,  # 初始资金
                 trade_cost: float = None,  # 交易成本
                 rebalance_days: int = None,
                 cash_keep: float = None):
        """
        因子收益回测（用回测框架Backtrader进行测试）

        Parameters
        ----------
        data_loader
        initial_capital
        trade_cost
        rebalance_days
        cash_keep
        """

        super().__init__(data_loader)
        self._initial_capital = initial_capital
        self._trade_cost = trade_cost
        self._rebalance_days = rebalance_days
        self._cash_keep = cash_keep

        self.cerebro: bt.Cerebro = None

    def _cerebro_init(self, factor_struct: FactorDataStruct):
        """初始化cerebro，加载数据"""
        print("Initializing Cerebro...")
        cerebro = bt.Cerebro()
        sec_codes = factor_struct.all_sec_codes
        start_date = factor_struct.all_dates[0]
        end_date = factor_struct.all_dates[-1]  # + factor_struct.factor_freq

        # 1. 加载数据
        print("Loading Stock Price Data...")
        price_dct = self._load_price_data(sec_codes, start_date, end_date)
        for sec_code, sec_df in price_dct.items():
            print(sec_df)
            data_feed = AmountPandasFeed(dataname=sec_df)
            cerebro.adddata(data_feed, name=sec_code)

        # 2. 设置参数
        # 初始资金
        cerebro.broker.setcash(self._initial_capital)
        # 交易成本
        cerebro.broker.setcommission(self._trade_cost)

        # 成交价格（当前收盘价/下一开盘/下根VWAP）
        # cerebro.broker.set_coc()

        # 3. 添加Analyzer
        # TODO 增加换手率Analyzer
        cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='pnl')
        self.cerebro = cerebro
        print("Cerebro Initialized...")

    def _load_price_data(self,
                          sec_codes: list,
                          start_date: str,
                          end_date: str,
                          freq: str = '1d') -> dict:
        """

        Parameters
        ----------
        sec_codes
        start_date
        end_date
        freq

        Returns
        -------

        """
        price_data = self._data_loader.get_stock_price(sec_codes=sec_codes,
                                                        start_date=start_date,
                                                        end_date=end_date,
                                                        freq=freq)

        trade_dts = pd.to_datetime(get_trade_date_between(start_date, end_date, return_type='str'))

        fields_p = ['open', 'high', 'low', 'close']
        fields_v = ['volume', 'amount']

        price_dct = {}
        for sec_code, df in price_data.groupby('sec_code'):
            df = df.reset_index(level=1, drop=True)
            df = df.reindex(trade_dts)
            df.loc[:, fields_p] = df[fields_p].fillna(method='pad', inplace=False)
            df.loc[:, fields_v] = df[fields_v].fillna(value=0, inplace=False)
            price_dct[sec_code] = df

        return price_dct

    @staticmethod
    def _quantile_to_portfolio_target(factor_quantile: DataFrame) -> dict:

        port_weight = OrderedDict()
        for (dt, q), df in factor_quantile.groupby(['trade_date', factor_quantile]):
            if q not in port_weight:
                port_weight[q] = OrderedDict()
            port_weight[q][dt] = dict(zip(df.index, [1/len(df)]*len(df)))

        return port_weight

    def transform(self, factor_struct: FactorDataStruct) -> DataFrame:

        if self.cerebro is None:
            self._cerebro_init(factor_struct)

        port_weight = self._quantile_to_portfolio_target(factor_struct.factor_quantile)

        q_ret = OrderedDict()

        for q, v in port_weight.items():
            cerebro = deepcopy(self.cerebro)
            cerebro.addstrategy(PortfolioRebalanceStrategy, v, self._cash_keep)
            strats = cerebro.run()
            strat = strats[0]
            daily_return = strat.analyzers.pnl.get_analysis()
            daily_return = pd.Series(daily_return)
            q_ret[q] = daily_return

        return pd.DataFrame(q_ret)



