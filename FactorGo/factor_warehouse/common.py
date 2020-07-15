
import abc

from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin
from FactorGo.factor_base import FactorDataStruct
from FactorGo.data_loader import data_api

"""
FactorAlgo以下功能

1. 根据给定的算法计算因子
自动从表里读取数据计算因子/或者根据给定的数据计算因子，更新因子库

2. 读取/保存因子

3. 返回FactorDataStruct数据
"""


class BaseFactorAlgo(BaseEstimator, TransformerMixin, metaclass=abc.ABCMeta):

    __data_loader = data_api

    def __init__(self, factor_name):
        self.factor_name = factor_name

    def fit(self, data):
        return self

    def prepare_data(self):
        ...

    def transform(self, data: DataFrame) -> FactorDataStruct:
        """TODO ❓ 这里的transform该完成什么？"""
        ...

    @abc.ABCMeta
    def run(self):
        ...

    def save(self, conn, table_name):
        """保存至数据库"""
        ...

    def update(self):
        """更新因子"""
        ...


