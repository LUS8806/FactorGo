
import backtrader as bt
import pandas as pd
from typing import Union
from pandas import DataFrame
from DataGo import get_stock_price, get_trade_date_between


class AmountPandasFeed(bt.feeds.PandasData):
    lines = ('amount', )
    params = dict(
        amount=5,  # npy field is in the 6th column (0 based index)
        openinterest=-1,  # -1 indicates there is no openinterest field
    )

# class PortfolioRebalanceStrategy(bt.Strategy):
#
#     def log(self, arg):
#         print('{} {}'.format(self.datetime.date(), arg))
#
#     def __init__(self,
#                  portfolio_target: dict,
#                  cash_keep: float = 0.05,
#                  trade_limit: float = 0.098,
#                  order_type: str = 'close'):
#         """
#
#         Parameters
#         ----------
#         portfolio_target: 目标持仓数据，包括股票代码与对应的权重
#          {'2020-05-31': {'000001.XSHE': 0.01, '600000.XSHG': 0.02},
#           '2020-06-03': {'000001.XSHE': 0.01, '600000.XSHG': 0.02},...}
#
#         cash_keep: 现金的比率
#         trade_limit: 涨跌停无法下单比率, 默认0.98
#         order_type: 'close' 当前收盘，'open' 下一开盘，'vwap' 成交量加权平均
#
#
#         """
#
#         self.portfolio_target = portfolio_target
#         self.cash_keep = cash_keep
#         self.trade_limit = trade_limit
#         self.order_type = order_type
#
#     def next(self):
#         cur_date = self.datetime.date(ago=0)  # 0 is the default
#         cur_target = self.portfolio_target.get(cur_date)
#         if cur_target is not None:
#
#             to_sell = []
#             to_buy = []
#
#             # 当前账户总价值
#             cur_tot_value = self.broker.get_value()*(1-self.cash_keep)
#
#             # 当前持仓
#             pos_data = [d for d, pos in self.getpositions().items() if pos]
#             pos_value ={d._name: self.broker.get_value(d) for d in pos_data}
#
#             # 现有持仓不在目标持仓中的股票，目标权重设置为0
#             for code in pos_value:
#                 if code not in cur_target:
#                     cur_target[code] = 0
#
#             for code, tar_pct in cur_target.items():
#                 tar_value = cur_tot_value*tar_pct
#                 # 当前有持仓的股票
#                 if code in pos_value:
#                     # 目标金额小于当前持仓金额，需要卖出
#                     if tar_value < pos_value[code]:
#                         to_sell.append(code)
#
#                     elif tar_value > pos_value[code]:
#                         to_buy.append(code)
#
#                 else:  # 当前无持仓的股票
#                     if tar_value > 0:
#                         to_buy.append(code)
#                     elif tar_value < 0:
#                         to_sell.append(code)
#
#             # 先卖
#             for code in to_sell:
#
#                 # 默认设置为下一根K线开盘价买
#                 if self.trade_success(code, direction='sell'):
#                     self.order_target_percent(self.getdatabyname(code), target=cur_target[code])
#                     self.log('Sell Success {}'.format(code))
#                 else:
#                     self.log('Sell Failed {}'.format(code))
#
#             # 再买
#             for code in to_buy:
#                 if self.trade_success(code, direction='buy'):
#                     self.order_target_percent(self.getdatabyname(code), target=cur_target[code])
#                     self.log('Buy Success {}'.format(code))
#                 else:
#                     self.log('Buy Failed {}'.format(code))
#
#     def trade_success(self, code, direction='buy') -> bool:
#         bar = self.getdatabyname(code)
#         if bar.volume[1] > 0:
#             if direction == 'buy':
#                 if bar.open[1] / bar.close[0] >= (1+self.trade_limit):
#                     return False
#             elif direction == 'sell':
#                 if bar.open[1] / bar.close[0] <= (1-self.trade_limit):
#                     return False
#             else:
#                 raise ValueError("direction error!")
#         else:
#             return False
#
#     def put_order(self, direction):
#         """根据下单类型匹配不同的下单函数"""
#
#         ...


class PortfolioRebalanceStrategy(bt.Strategy):
    def __init__(self, name):
        self.dataclose = self.datas[0].close
        self.name = name
        self.order = None

    def next(self):

        # print(self.cerebro.p.cheat_on_open)
        if self.name == 's1':
            self.buy(size=100)
        print(self.name)
        print(self.broker.get_value())
        # print({"current_date":  self.datetime.date(ago=0),
        #        "current_close": self.dataclose[0],
        #        "pre_date": self.datas[0].datetime.date(-1),
        #        "pre_close": self.dataclose[-1],
        #        "amount": self.datas[0].amount[-1],
        #        "cur_value": self.broker.get_value()}
        #       )


def load_price_data(sec_codes: list,
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
    price_data = get_stock_price(sec_codes=sec_codes,
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


def run_portfolio_balance(sec_codes: list,
                          start_date: str,
                          end_date: str,
                          # portfolio_target: dict,
                          initial_cash: float = 10000000,
                          cash_keep: float = 0.05,
                          trade_limit: float = 0.098,
                          order_type: str = 'close'
                          ):
    """

    Parameters
    ----------
    sec_codes
    start_date
    end_date
    portfolio_target
    initial_cash
    cash_keep
    trade_limit
    order_type

    Returns
    -------

    """
    cerebro = bt.Cerebro()
    # 添加策略
    # cerebro.addstrategy(PortfolioRebalanceStrategy, portfolio_target, cash_keep, trade_limit, order_type)
    cerebro.addstrategy(PortfolioRebalanceStrategy, name='s1')
    cerebro.addstrategy(PortfolioRebalanceStrategy, name='s2')


    # 加载数据
    price_data = load_price_data(sec_codes, start_date, end_date)
    for sec_code, sec_df in price_data.items():
        data_feed = AmountPandasFeed(dataname=sec_df)
        cerebro.adddata(data_feed, name=sec_code)

    # 设置初始资金
    cerebro.broker.setcash(initial_cash)
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='pnl')
    strats = cerebro.run()
    strat = strats[0]
    daily_return = strat.analyzers.pnl.get_analysis()
    daily_return = pd.Series(daily_return)
    return daily_return
    # Basic performance evaluation ... final value ... minus starting cash


if __name__ == '__main__':

    # df_price = load_price_data(['600000.XSHG', '000001.XSHE'], '2010-01-01', '2010-01-20')
    ret = run_portfolio_balance(['600000.XSHG', '000001.XSHE'], '2010-01-01', '2010-01-20')

    print('done')