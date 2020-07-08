import pandas as pd
import numpy as np
from pandas import Index
from typing import Union, List
from datetime import datetime, date
from DataGo.model.mongo_model import Document
# from DataGo.fetch.fetch import get_trade_date_count
from math import ceil


def dates_transform(dates: Union[str, datetime, date, list],
                    to_type: str = None) -> list:
    """
    日期列表转换成指定的统一的格式
    Parameters
    ----------
    dates: 日期序列
    to_type: 输出的日期类型，'str','datetime', 'date'

    Returns
    -------

    """
    if not isinstance(dates, (list, Index)):
        dates = [dates]
    res = pd.to_datetime(dates)
    if not to_type:
        pass
    elif to_type == 'str':
        res = res.strftime("%Y-%m-%d").tolist()
    elif to_type == 'datetime':
        res = res.to_pydatetime().tolist()
    elif to_type == 'date':
        res = list(res.date)
    else:
        raise ValueError("参数错误")
    if len(res) == 1:
        res = res[0]
    return res


def _docs_to_df(docs: List[Document],
                index_col: list = None,
                parse_date: str = None
                ):
    """
    mongo document objects转成DataFrame
    Parameters
    ----------
    docs
    index_col
    parse_date

    Returns
    -------

    """
    # TODO parse_date的问题
    lst_dct = []
    for doc in docs:
        lst_dct.append(doc.to_dict())
    df = pd.DataFrame(lst_dct)
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    # df[prase_date] = pd.to_datetime(df[prase_date])

    if index_col:
        df = df.set_index(index_col)

    return df

