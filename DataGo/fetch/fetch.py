import pandas as pd
import numpy as np
from math import ceil
from typing import List, Union
from pandas import DataFrame, Index, MultiIndex
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
from dateutil.parser import parse
from DataGo.model import sql_session
from DataGo.model import Security, IndexComponent, IndustryComponent, FADataValuation
from DataGo.model import StockPriceDay, StockAdjFactor, IndexPriceDay
from DataGo.fetch.trade_date import trade_date_day
from DataGo.fetch.util import dates_transform, _docs_to_df
from DataGo.fetch.sw_industry_name import sw_l1_name


def estimate_freq(dates: list) -> str:
    """
    估计日期序列的平均间隔
    Parameters
    ----------
    dates: 日期序列

    Returns
    -------

    """
    t_delta = []
    for i in range(len(dates)-1):
        t_delta.append(get_trade_date_count(dates[i], dates[i+1]))

    return '{}d'.format(ceil(np.mean(t_delta)))


def _get_fetch_dates(dates: Union[List[str], List[datetime], List[date], Index] = None,
                     start_date: Union[str, datetime, date] = None,
                     end_date: Union[str, datetime, date] = None,
                     freq: str = 'd',
                     offset: int = None,
                     return_type: str = None
                     ) -> list:
    """
    如果给定dates，则会忽略其他参数；
    如果dates为None，则会根据start_date, end_date, freq计算日期
    Parameters
    ----------
    dates
    start_date
    end_date
    freq
    offset
    return_type

    Returns
    -------

    """
    if dates:
        res = dates
    elif not start_date or not end_date:
        raise ValueError("参数错误")
    else:
        res = get_trade_date_between(start_date, end_date, freq, offset)
    res = dates_transform(res, return_type)

    return res


def get_trade_date_between(
        start_date: Union[str, datetime, date],
        end_date: Union[str, datetime, date],
        freq: str = 'd',
        offset: int = None,
        return_type: str = None) -> list:
    """
    自定义频率的交易日，比如每月最后一天: freq='m', offset=-1
    Parameters
    ----------
    start_date
    end_date
    freq
    offset
    return_type

    Returns
    -------

    """
    start_date, end_date = dates_transform([start_date, end_date], to_type='str')
    offset = -1 if not offset else offset
    if freq[-1] not in ['d', 'w', 'm']:
        raise ValueError("Unsupported freq value of '{}'".format(freq))

    ts_dates = pd.Series(index=pd.to_datetime(trade_date_day), data=trade_date_day)
    if freq[-1] != 'd':
        ts_dates = ts_dates.groupby(ts_dates.index.to_period(freq=freq)).apply(lambda x: x.iloc[offset])

    res = ts_dates[ts_dates.between(start_date, end_date)].tolist()
    res = dates_transform(res, return_type)

    return res


def get_trade_date_count(
        start_date: Union[str, datetime, date],
        end_date: Union[str, datetime, date]) -> int:
    """计算两个交易日之间相隔的天数"""

    return len(get_trade_date_between(start_date, end_date))


def get_trade_date_offset(dates: Union[str, datetime, date, list],
                          freq: Union[str, int] = '1d',
                          return_type: str = None,
                          return_max: bool = True,
                          ) -> list:
    """
    当前日后N天/周/月的交易日期
    Parameters
    ----------
    dates
    freq
    return_type
    return_max: 当日期超过范围时，True返回最大可用日期，False返回Nan

    Returns
    -------

    """
    dates = dates_transform(dates, 'str')
    if isinstance(freq, int):
        freq = str(freq)+'d'
    offset = int(freq[:-1])
    direction = np.sign(offset)
    offset = np.abs(offset)
    freq = freq[-1]

    ts_dates = pd.Series(trade_date_day)
    freq = freq.lower()

    def _get_trade_date_offset(dt):

        if freq == 'd':
            st_idx = ts_dates[ts_dates <= dt].tail(-1).index + offset * direction

        elif freq == 'w':
            dt = parse(dt)
            offset_date = dt + relativedelta(weeks=offset) * direction
            offset_date = str(offset_date.date())
            st_idx = ts_dates[ts_dates <= offset_date].tail(-1).index

        elif freq == 'm':
            dt = parse(dt)
            offset_date = dt + relativedelta(months=offset) * direction
            offset_date = str(offset_date.date())
            st_idx = ts_dates[ts_dates <= offset_date].tail(-1).index

        else:
            raise ValueError("Unsupported freq value {}".format(freq))

        if st_idx.tolist()[-1] > len(ts_dates)-1 and not return_max:
            return np.nan
        else:
            st_idx = min(st_idx.tolist()[-1], len(ts_dates)-1)
            offset_date = ts_dates[st_idx]
            return offset_date

    if not isinstance(dates, list):
        dates = [dates]
    res = [_get_trade_date_offset(dt) for dt in dates]
    res = dates_transform(res, return_type)

    return res


def get_stock_list_days(sec_codes: Union[str, List[str]] = None,
                        fields: Union[str, List[str]] = None) -> DataFrame:
    """
    股票上市（退市）日期数据
    Parameters
    ----------
    sec_codes: stock codes
    fields: ['list_date', 'dlist_date']

    Returns
    -------

    """

    if isinstance(sec_codes, str):
        sec_codes = [sec_codes]

    if isinstance(fields, str):
        fields = [fields]

    if not sec_codes:
        df_query = sql_session.query(
            Security.sec_code,
            Security.list_date,
            Security.dlist_date
        ).filter(
            Security.sec_type == 'stock',
        ).statement
    else:
        df_query = sql_session.query(
            Security.sec_code,
            Security.list_date,
            Security.dlist_date
        ).filter(
            Security.sec_type == 'stock',
            Security.sec_code.in_(sec_codes)
        ).statement

    res = pd.read_sql(df_query, con=sql_session.bind, index_col='sec_code')

    if fields:
        res = res[fields]

    return res


def get_index_components(index_code: str,
                         dates: Union[str, datetime, date, list, Index] = None,
                         start_date: Union[str, datetime, date] = None,
                         end_date: Union[str, datetime, date] = None,
                         freq: str = '1d',
                         offset: int = None,
                         weight: bool = False,
                         return_index: bool = False,
                         ) -> Union[DataFrame, MultiIndex]:
    """
    指数成分股数据
    Parameters
    ----------
    index_code: 指数代码
    dates: 日期列表
    start_date: 开始日期
    end_date: 结束日期
    freq: 周期
    return_index: 是否返回Index格式
    offset
    return_index

    Returns
    -------

    """
    dates = _get_fetch_dates(dates, start_date, end_date, freq, return_type='str', offset=offset)
    df_query = sql_session.query(
        IndexComponent.trade_date,
        IndexComponent.sec_code,
        IndexComponent.weight,
    ).filter(
        IndexComponent.index_code == index_code,
        IndexComponent.trade_date.in_(dates)
    ).statement

    res = pd.read_sql(df_query, con=sql_session.bind)
    res['trade_date'] = pd.to_datetime(res['trade_date'])
    res = res.sort_values(by=['trade_date', 'sec_code']).reset_index(drop=True)

    if not weight:
        res = res.drop('weight', axis=1)
    if return_index:
        res = res.set_index(['trade_date', 'sec_code']).index
    return res


def get_stock_industries(codes_date_index: Union[MultiIndex, DataFrame] = None,
                         sec_codes: List[str] = None,
                         dates: Union[str, datetime, date, list, Index] = None,
                         start_date: Union[str, datetime, date] = None,
                         end_date: Union[str, datetime, date] = None,
                         freq: str = '1d',
                         offset: str = None,
                         industry: str = None) -> DataFrame:
    """
    默认申万一级行业
    TODO 加上SAM行业
    股票所属行业
    Parameters
    ----------
    codes_date_index
    sec_codes
    dates
    start_date
    end_date
    freq
    offset
    industry: 行业分类

    Returns
    -------

    """
    industry = 'sw' if not industry else industry

    if codes_date_index is not None:
        if isinstance(codes_date_index, DataFrame):
            codes_date_index = codes_date_index.set_index(['trade_date', 'sec_code'])
        dates = codes_date_index.get_level_values(0).tolist()
        sec_codes = codes_date_index.get_level_values(1).tolist()

    dates = _get_fetch_dates(dates, start_date, end_date, freq, offset=offset, return_type='str')

    df_query = sql_session.query(
        IndustryComponent.date,
        IndustryComponent.sec_code,
        IndustryComponent.sw_l1
    ).filter(
        IndustryComponent.sec_code.in_(sec_codes),
        IndustryComponent.date.in_(dates),
    ).statement

    res = pd.read_sql(df_query, con=sql_session.bind)
    res = res.rename({'date': 'trade_date', 'sw_l1': industry}, axis=1).set_index(['trade_date', 'sec_code'])
    res[industry] = res[industry].map(sw_l1_name)

    if codes_date_index is not None:
        codes_date_index.names = ['trade_date', 'sec_code']
        temp = pd.DataFrame(index=codes_date_index)
        res = pd.merge(temp, res, left_index=True, right_index=True, how='left')

    return res


def get_stock_fin_indicators(indicators: List[str],
                             codes_date_index: Union[MultiIndex, DataFrame] = None,
                             sec_codes: List[str] = None,
                             dates: Union[str, datetime, date, list, Index] = None,
                             start_date: Union[str, datetime, date] = None,
                             end_date: Union[str, datetime, date] = None,
                             freq: str = '1d',
                             offset: int = None) -> DataFrame:
    """
    财务指标，包括日度估值类指标及PIT财务指标
    Parameters
    ----------
    indicators
    codes_date_index
    sec_codes
    dates
    start_date
    end_date
    freq
    offset

    Returns
    -------

    """

    return _get_stock_daily_fin_indicators(indicators, codes_date_index, sec_codes, dates, start_date, end_date, freq, offset)


def _get_stock_daily_fin_indicators(indicators: Union[str, List[str]],
                                    codes_date_index: Union[MultiIndex, DataFrame] = None,
                                    sec_codes: List[str] = None,
                                    dates: Union[str, datetime, date, list, Index] = None,
                                    start_date: Union[str, datetime, date] = None,
                                    end_date: Union[str, datetime, date] = None,
                                    freq: str = '1d',
                                    offset: int = None) -> DataFrame:

    if isinstance(indicators, str):
        indicators = [indicators]

    if codes_date_index is not None:
        if isinstance(codes_date_index, DataFrame):
            codes_date_index = codes_date_index.set_index(['trade_date', 'sec_code'])
        dates = codes_date_index.get_level_values(0).unique().tolist()
        sec_codes = codes_date_index.get_level_values(1).unique().tolist()

    dates = _get_fetch_dates(dates, start_date, end_date, freq, offset, return_type='str')

    df_query = sql_session.query(
        FADataValuation
    ).filter(
        FADataValuation.sec_code.in_(sec_codes),
        FADataValuation.trade_date.in_(dates),
    ).statement

    res = pd.read_sql(df_query, con=sql_session.bind).set_index(['trade_date', 'sec_code'])
    res = res[indicators]

    if codes_date_index is not None:
        codes_date_index.names = ['trade_date', 'sec_code']
        temp = pd.DataFrame(index=codes_date_index)
        res = pd.merge(temp, res, left_index=True, right_index=True, how='left')

    return res


def _get_stock_report_fin_indicators(codes_date_index: Union[MultiIndex, DataFrame] = None,
                                     sec_codes: List[str] = None,
                                     dates: Union[str, datetime, date, list, Index] = None,
                                     start_date: Union[str, datetime, date] = None,
                                     end_date: Union[str, datetime, date] = None,
                                     freq: str = '1d',
                                     offset: int = None) -> DataFrame:
    # TODO 解决指标PIT查询的问题

    if codes_date_index is not None:
        dates = codes_date_index.get_level_values(0).unique().tolist()
        sec_codes = codes_date_index.get_level_values(1).unique().tolist()

    dates = _get_fetch_dates(dates, start_date, end_date, freq, offset=offset, return_type='str')

    ...


def get_stock_price(codes_date_index: Union[MultiIndex, DataFrame] = None,
                    sec_codes: List[str] = None,
                    dates: Union[str, datetime, date, list, Index] = None,
                    start_date: Union[str, datetime, date] = None,
                    end_date: Union[str, datetime, date] = None,
                    freq: str = '1d',
                    offset: int = None,
                    fields: List[str] = None,
                    adj_type: str = 'qfq',
                    ) -> DataFrame:
    """
    股票价格
    Parameters
    ----------
    codes_date_index
    sec_codes
    dates
    start_date
    end_date
    freq
    offset
    fields: [open, high, low, close, volume, amount]
    adj_type: 前复权qfq, 后复权hfq, 不复权None

    Returns
    -------

    """
    all_fields = ['open', 'high', 'low', 'close', 'volume', 'amount']

    if not fields:
        fields = all_fields
    else:
        fields = [field for field in fields if field in all_fields]

    if codes_date_index is not None:
        if isinstance(codes_date_index, DataFrame):
            codes_date_index = codes_date_index.set_index(['trade_date', 'sec_code'])
        dates = codes_date_index.get_level_values(0).unique().tolist()
        sec_codes = codes_date_index.get_level_values(1).unique().tolist()

    qa_codes = [code[0:6] for code in sec_codes]
    qa_codes_dct = dict(zip(qa_codes, sec_codes))

    dates = _get_fetch_dates(dates, start_date, end_date, freq, offset, return_type='str')
    price_docs = StockPriceDay.objects.filter(code__in=qa_codes, date__in=dates)
    df_price = _docs_to_df(price_docs, index_col=['trade_date', 'sec_code'])

    df_price = df_price[fields]

    if adj_type is not None:
        # 获取复权因子
        adj_docs = StockAdjFactor.objects.filter(code__in=qa_codes, date__in=dates)
        df_adj = _docs_to_df(adj_docs, index_col=['trade_date', 'sec_code'])

        df_price = pd.merge(df_price, df_adj, left_index=True, right_index=True, how='left')

        if adj_type == 'qfq':    # 前复权
            for col in fields:
                if col == 'volume':
                    df_price['volume'] = df_price['volume'] / df_price['adj']
                else:
                    df_price[col] = df_price[col] * df_price['adj']

        elif adj_type == 'hfq':
            # TODO 完成后复权计算逻辑
            raise ValueError("暂时还不支持后复权计算")
            pass

        df_price = df_price.drop('adj', axis=1)

    df_price = df_price.rename(qa_codes_dct, level=1, axis=0).sort_index()

    return df_price


def get_stock_return(codes_date_index: Union[MultiIndex, DataFrame] = None,
                     sec_codes: List[str] = None,
                     dates: Union[str, datetime, date, list, Index] = None,
                     start_date: Union[str, datetime, date] = None,
                     end_date: Union[str, datetime, date] = None,
                     freq: str = '1d',
                     offset: int = None,
                     periods: Union[str, List[str]] = None,
                     price_type: str = 'close',
                     columns: str = None
                     ) -> DataFrame:
    """
    TODO 多个period的收益率数据
    TODO Mongo中根据datestamp查询
    股票收益率数据
    Parameters
    ----------
    codes_date_index
    sec_codes
    dates
    start_date
    end_date
    freq
    offset
    periods
    price_type: 计算收益率价格的类型，只能是'open'或'close'
    columns

    Returns
    -------

    """
    st_price = get_stock_price(codes_date_index, sec_codes, dates,
                               start_date, end_date, freq, offset, [price_type], 'qfq')

    st_price.columns = ['start']

    if codes_date_index is not None:
        if isinstance(codes_date_index, DataFrame):
            codes_date_index = codes_date_index.set_index(['trade_date', 'sec_code'])
        dates = codes_date_index.get_level_values(0).unique().tolist()
        sec_codes = codes_date_index.get_level_values(1).unique().tolist()

    dates = _get_fetch_dates(dates, start_date, end_date, freq, offset)

    def _get_et_ret(_et_dates, _name):
        st_et_dt_map = dict(zip(_et_dates, dates))
        codes_date_index_et = pd.MultiIndex.from_product([_et_dates, sec_codes], names=['trade_date', 'sec_code'])
        et_price = get_stock_price(codes_date_index=codes_date_index_et, fields=[price_type], adj_type='qfq')
        et_price = et_price.rename(st_et_dt_map, level=0, axis=0)
        et_price.columns = ['end']

        all_price = pd.merge(st_price, et_price, left_index=True, right_index=True, how='left')
        result = all_price.eval("end/start - 1")
        result = result.to_frame(name=_name)
        if codes_date_index is not None:
            result = result.reindex(codes_date_index)
        return result

    if not periods:
        et_dates = dates[1:].tolist()
        et_dates.append(get_trade_date_offset(dates[-1], estimate_freq(dates)))
        res = _get_et_ret(et_dates, 'forward_ret_default')

    else:
        if isinstance(periods, str):
            periods = [periods]
        res = pd.DataFrame()
        for period in periods:
            et_dates = get_trade_date_offset(dates, period)
            name = 'forward_ret_{}'.format(period)
            et_ret = _get_et_ret(et_dates, name)
            if res.empty:
                res = et_ret
            else:
                res[name] = et_ret

    if columns:
        res.columns = columns

    return res


def get_index_price(codes_date_index: Union[MultiIndex, DataFrame] = None,
                    sec_codes: List[str] = None,
                    dates: Union[str, datetime, date, list, Index] = None,
                    start_date: Union[str, datetime, date] = None,
                    end_date: Union[str, datetime, date] = None,
                    freq: str = '1d',
                    offset: int = None,
                    fields: List[str] = None,
                    ) -> DataFrame:
    """
    股票价格
    Parameters
    ----------
    codes_date_index
    sec_codes
    dates
    start_date
    end_date
    freq
    offset
    fields: [open, high, low, close, volume, amount]
    adj_type: 前复权qfq, 后复权hfq, 不复权None

    Returns
    -------

    """
    all_fields = ['open', 'high', 'low', 'close', 'volume', 'amount']

    if not fields:
        fields = all_fields
    else:
        fields = [field for field in fields if field in all_fields]

    if codes_date_index is not None:
        if isinstance(codes_date_index, DataFrame):
            codes_date_index = codes_date_index.set_index(['trade_date', 'sec_code'])
        dates = codes_date_index.get_level_values(0).unique().tolist()
        sec_codes = codes_date_index.get_level_values(1).unique().tolist()

    qa_codes = [code[0:6] for code in sec_codes]
    qa_codes_dct = dict(zip(qa_codes, sec_codes))

    dates = _get_fetch_dates(dates, start_date, end_date, freq, offset, return_type='str')
    price_docs = StockPriceDay.objects.filter(code__in=qa_codes, date__in=dates)
    df_price = _docs_to_df(price_docs, index_col=['trade_date', 'sec_code'])

    df_price = df_price[fields]

    df_price = df_price.rename(qa_codes_dct, level=1, axis=0).sort_index()

    return df_price


def get_index_return(codes_date_index: Union[MultiIndex, DataFrame] = None,
                     sec_codes: List[str] = None,
                     dates: Union[str, datetime, date, list, Index] = None,
                     start_date: Union[str, datetime, date] = None,
                     end_date: Union[str, datetime, date] = None,
                     freq: str = '1d',
                     offset: int = None,
                     periods: Union[str, List[str]] = None,
                     price_type: str = 'close',
                     columns: str = None
                     ) -> DataFrame:
    """
    TODO 多个period的收益率数据
    股票收益率数据
    Parameters
    ----------
    codes_date_index
    sec_codes
    dates
    start_date
    end_date
    freq
    offset
    periods
    price_type: 计算收益率价格的类型，只能是'open'或'close'
    columns

    Returns
    -------

    """
    st_price = get_index_price(codes_date_index, sec_codes, dates,
                               start_date, end_date, freq, offset, [price_type])

    st_price.columns = ['start']

    if codes_date_index is not None:
        if isinstance(codes_date_index, DataFrame):
            codes_date_index = codes_date_index.set_index(['trade_date', 'sec_code'])
        dates = codes_date_index.get_level_values(0).unique().tolist()
        sec_codes = codes_date_index.get_level_values(1).unique().tolist()

    dates = _get_fetch_dates(dates, start_date, end_date, freq, offset)

    def _get_et_ret(_et_dates, _name):
        st_et_dt_map = dict(zip(_et_dates, dates))
        codes_date_index_et = pd.MultiIndex.from_product([_et_dates, sec_codes], names=['trade_date', 'sec_code'])
        et_price = get_index_price(codes_date_index=codes_date_index_et, fields=[price_type], adj_type='qfq')
        et_price = et_price.rename(st_et_dt_map, level=0, axis=0)
        et_price.columns = ['end']

        all_price = pd.merge(st_price, et_price, left_index=True, right_index=True, how='left')
        result = all_price.eval("end/start - 1")
        result = result.to_frame(name=_name)
        return result

    if not periods:
        et_dates = dates[1:].tolist()
        et_dates.append(get_trade_date_offset(dates[-1], estimate_freq(dates)))
        res = _get_et_ret(et_dates, 'forward_ret_default')

    else:
        if isinstance(periods, str):
            periods = [periods]
        res = pd.DataFrame()
        for period in periods:
            et_dates = get_trade_date_offset(dates, period)
            name = 'forward_ret_{}'.format(period)
            et_ret = _get_et_ret(et_dates, name)
            if res.empty:
                res = et_ret
            else:
                res[name] = et_ret

    if columns:
        res.columns = columns

    return res


def get_all_sec_codes(types=None):
    """获取所有已上市的股票及指数信息"""
    types = ['stock', 'index'] if not types else types

    df_query = sql_session.query(Security.sec_code, Security.list_date, Security.dlist_date).filter(
        Security.sec_type.in_(types)).statement

    res = pd.read_sql(df_query, con=sql_session.bind)['sec_code'].tolist()

    return res


