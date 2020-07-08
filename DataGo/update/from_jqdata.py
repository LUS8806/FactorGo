import jqdatasdk as jq
import pandas as pd
import numpy as np
from DataGo.model import engine
from DataGo.fetch import get_all_sec_codes

# TODO 参考Quantaxis->QASU实现数据更新模块


def update_stock_valuation(start_date, end_date, conn=None):
    """更新股票估值数据，包括市值、市盈率等"""

    if not conn:
        conn = engine

    dts = jq.get_trade_days(start_date, end_date)

    for dt in dts:
        dt_str = dt.strftime("%Y-%m-%d")
        df = jq.get_fundamentals(jq.query(jq.valuation), dt_str)
        cols = list(set(df.columns) - {'id'})
        df = df[cols]
        df = df.rename({'code': 'sec_code', 'day': 'trade_date'}, axis='columns')
        df.to_sql(name='fa_data_valuation', con=conn, index=False, if_exists='append')
        print(dt_str)


def update_industry_com(start_date, end_date=None):
    """更新股票行业分类数据"""
    dts = pd.date_range(start_date, end_date)
    dt_len = len(dts)
    codes = get_all_sec_codes(types=['stock'])

    for idx, dt in enumerate(dts):
        dt = dt.strftime("%Y-%m-%d")
        ind_data = jq.get_industry(codes,dt)
        df = pd.DataFrame(ind_data).T
        df = df[~df['sw_l1'].isna()]
        df = df.applymap(lambda x: x['industry_code'] if x is not np.nan else x)
        cols = ['zjw','sw_l1','sw_l2', 'sw_l3']
        df = df[cols]
        df.index.name = 'sec_code'
        df = df.reset_index()
        df['date'] = dt
        df.to_sql(name='industry_component', con=engine, if_exists='append', index=False)
        print('{:03d}/{}--{} UPDATED.'.format(idx, dt_len, dt))


if __name__ == "__main__":
    # TODO 命令行参数解析
    from jqdatasdk import auth
    auth('18660537822', '537822')
    # update_stock_valuation('2014-12-31', '2014-12-31')
    update_industry_com('2014-12-31', '2014-12-31')