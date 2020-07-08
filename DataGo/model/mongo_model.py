from mongoengine import Document, FloatField, StringField, IntField
from mongoengine import connect

connect(alias='qa', db='quantaxis')
connect(alias='barra_des', db='factor')

class StockPriceDay(Document):
    open = FloatField()
    close = FloatField()
    high = FloatField()
    low = FloatField()
    vol = FloatField()
    amount = FloatField()
    date = StringField()
    code = StringField()
    date_stamp = FloatField()
    meta = {'db_alias': 'qa', 'collection': 'stock_day'}

    def to_dict(self):
        return {
            'sec_code': self.code,
            'trade_date': self.date,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.vol,
            'amount': self.amount,
        }


class StockAdjFactor(Document):
    date = StringField()
    code = StringField()
    adj = FloatField()
    meta = {'db_alias': 'qa', 'collection': 'stock_adj'}

    def to_dict(self):
        return {
            'sec_code': self.code,
            'trade_date': self.date,
            'adj': self.adj
        }


class StockPriceMinute(Document):
    open = FloatField()
    close = FloatField()
    high = FloatField()
    low = FloatField()
    vol = FloatField()
    amount = FloatField()
    date = StringField()
    code = StringField()
    date_stamp = FloatField()
    meta = {'db_alias': 'qa', 'collection': 'stock_min'}

    def to_dict(self):
        return {
            'sec_code': self.code,
            'trade_date': self.date,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.vol,
            'amount': self.amount,
        }


class IndexPriceDay(Document):
    open = FloatField()
    close = FloatField()
    high = FloatField()
    low = FloatField()
    vol = FloatField()
    amount = FloatField()
    date = StringField()
    code = StringField()
    up_count = IntField()
    down_count = IntField()
    date_stamp = FloatField()
    meta = {'db_alias': 'qa', 'collection': 'index_day'}

    def to_dict(self):
        return {
            'sec_code': self.code,
            'trade_date': self.date,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.vol,
            'amount': self.amount,
            'up_count': self.up_count,
            'down_count': self.down_count
        }


if __name__ == '__main__':
    # from QUANTAXIS.QAData import QADataStruct

    price = StockPriceDay.objects.filter(code__in=['000001', '600000'], date='2019-01-24')

    print(price[1].to_dict())
