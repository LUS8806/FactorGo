import abc
from DataGo.fetch import *


class BaseDataLoader(metaclass=abc.ABCMeta):
    @staticmethod
    def get_stock_list_days(*args, **kwargs):
        ...

    @staticmethod
    def get_stock_price(*args, **kwargs):
        ...

    @staticmethod
    def get_stock_return(*args, **kwargs):
        ...

    @staticmethod
    def get_index_components(*args, **kwargs):
        ...

    @staticmethod
    def get_stock_industries(*args, **kwargs):
        ...

    @staticmethod
    def get_stock_turnover(*args, **kwargs):
        ...

    @staticmethod
    def get_stock_cap(*args, **kwargs):
        ...

    @staticmethod
    def get_trade_date_offset(*args, **kwargs):
        ...

    @staticmethod
    def get_trade_date_between(*args, **kwargs):
        ...


class DemoDataLoader(BaseDataLoader):

    @staticmethod
    def get_stock_list_days(*args, **kwargs):
        return get_stock_list_days(*args, **kwargs)

    @staticmethod
    def get_stock_price(*args, **kwargs):
        return get_stock_price(*args, **kwargs)

    @staticmethod
    def get_stock_return(*args, **kwargs):
        return get_stock_return(*args, **kwargs)

    @staticmethod
    def get_index_components(*args, **kwargs):
        return get_index_components(*args, **kwargs)

    @staticmethod
    def get_stock_industries(*args, **kwargs):
        return get_stock_industries(*args, **kwargs)

    @staticmethod
    def get_stock_turnover(*args, **kwargs):
        return get_stock_fin_indicators(*args, **kwargs, indicators=['turnover_ratio'])

    @staticmethod
    def get_stock_cap(*args, **kwargs):
        return get_stock_fin_indicators(*args, **kwargs, indicators=['market_cap'])

    @staticmethod
    def get_trade_date_offset(*args, **kwargs):
        return get_trade_date_offset(*args, **kwargs)

    @staticmethod
    def get_trade_date_between(*args, **kwargs):
        return get_trade_date_between(*args, **kwargs)


data_api = DemoDataLoader()

