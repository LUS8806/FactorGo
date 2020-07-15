import sqlalchemy
from sqlalchemy import UniqueConstraint
from sqlalchemy import Column, String, Integer, Date, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from .config import SQL_CONN

Base = declarative_base()


def connect_sql(user_name=None, host=None, dbname=None, pass_word=None, port=3306):
    """连接数据库，返回engine"""
    conn_url = 'mysql+pymysql://{}@{}:{}/{}'.format(SQL_CONN['user_name'],
                                                    SQL_CONN['host'],
                                                    SQL_CONN['port'],
                                                    SQL_CONN['db_name']
                                                    )
    sql_engine = sqlalchemy.create_engine(conn_url, encoding='utf-8')
    return sql_engine


def to_dict(self):
    return {c.name: getattr(self, c.name, None)
            for c in self.__table__.columns}


Base.to_dict = to_dict
engine = connect_sql()
Session = sessionmaker(bind=engine)
session = Session()


class Security(Base):
    __tablename__ = 'security_info'

    id = Column(Integer, primary_key=True)
    sec_code = Column(String(60))
    name_zh = Column(String(60))
    name_eng = Column(String(20))
    list_date = Column(Date)
    dlist_date = Column(Date)
    sec_type = Column(String(20))


class IndexComponent(Base):
    """指数成分股权重"""

    __tablename__ = 'index_component_weight'

    id = Column(Integer, primary_key=True)
    index_code = Column(String(60))
    sec_code = Column(String(60))
    trade_date = Column(Date)
    weight = Column(Float)


class IndustryComponent(Base):
    """行业成分股（申万、证监会）"""

    __tablename__ = 'industry_component'

    id = Column(Integer, primary_key=True)
    sec_code = Column(String(60))
    date = Column(Date)
    zjw = Column(String(60))
    sw_l1 = Column(String(60))
    sw_l2 = Column(String(60))
    sw_l3 = Column(String(60))


class FADataValuation(Base):
    """市值数据"""
    __tablename__ = 'fa_data_valuation'

    __table_args__ = (
        UniqueConstraint('sec_code', 'trade_date', name='_sec_date_uc'),
                     )

    id = Column(Integer, primary_key=True)
    sec_code = Column(String(60), index=True)
    trade_date = Column(Date, index=True)

    pe_ratio = Column(Float)
    turnover_ratio = Column(Float)
    pb_ratio = Column(Float)
    ps_ratio = Column(Float)
    pcf_ratio = Column(Float)
    capitalization = Column(Float)
    market_cap = Column(Float)
    circulating_cap = Column(Float)
    circulating_market_cap = Column(Float)
    pe_ratio_lyr = Column(Float)

#
# class FAItemDesc(Base):
#     """财务字段说明表"""
#     ...
#
#
# class FAItemOrigin(Base):
#     """原始财务科目表"""
#     __tablename__ = 'fa_item_origin'
#
#     __table_args__ = (
#         UniqueConstraint('sec_code', 'report_date', 'pub_date', name='_sec_date_uc'),
#                      )
#
#     id = Column(Integer, primary_key=True)
#     sec_code = Column(String(60), index=True)
#     report_date = Column(Date, index=True)
#     pub_date = Column(Date, index=True)
#     item_name = Column(String(60), index=True)
#     value = Column(Float)
#     source_sheet = Column(String(60))   # 科目来源
#     release_type = Column(String(60))   # 披露类型
#
#
# class FAItemSeason(Base):
#     """单季度财务科目表"""
#     __tablename__ = 'fa_item_season'
#
#     __table_args__ = (
#         UniqueConstraint('sec_code', 'report_date', 'pub_date', name='_sec_date_uc'),
#     )
#
#     id = Column(Integer, primary_key=True)
#     sec_code = Column(String(60), index=True)
#     report_date = Column(Date, index=True)
#     pub_date = Column(Date, index=True)
#     item_name = Column(String(60), index=True)
#     value = Column(Float)
#     source_sheet = Column(String(60))  # 科目来源
#     release_type = Column(String(60))  # 披露类型
#
#
# class FAItemTTM(Base):
#     """TTM数据表"""
#     __tablename__ = 'fa_item_ttm'
#
#     __table_args__ = (
#         UniqueConstraint('sec_code', 'report_date', 'pub_date', name='_sec_date_uc'),
#                      )
#
#     id = Column(Integer, primary_key=True)
#     sec_code = Column(String(60), index=True)
#     report_date = Column(Date, index=True)
#     pub_date = Column(Date, index=True)
#     release_type = Column(String(60))   # 披露类型
#
#
# class FAIndicator(Base):
#     """财务指标数据"""
#     __tablename__ = 'fa_indicator_ttm'
#
#     id = Column(Integer, primary_key=True)
#     sec_code = Column(String(60), index=True)
#     report_date = Column(Date, index=True)
#     pub_date = Column(Date, index=True)
#     release_type = Column(String(60))   # 披露类型
#     cal_type = Column(String(60))       # 计算类型
#
#
# class FAIndicatorDaily(Base):
#     """估值指标数据"""
#     __tablename__ = 'fa_indicator_ttm'
#
#     id = Column(Integer, primary_key=True)
#     sec_code = Column(String(60), index=True)
#     report_date = Column(Date, index=True)
#     pub_date = Column(Date, index=True)
#     release_type = Column(String(60))   # 披露类型
#     cal_type = Column(String(60))       # 计算类型


# class BarraDescriptor(Base):
#     """BARRA描述变量，日度更新"""
#
#     __table__name = 'barra_descriptor'
#
#     id = Column(Integer, primary_key=True)
#     sec_code = Column(String(60), index=True)
#     trade_date = Column(Date)
#