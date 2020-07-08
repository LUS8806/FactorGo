"""
因子模型

因子模型，如CAPM就是一种模型，BARRA也是

"""
import abc


"""
风险模型主要实现三个功能：

估算协方差矩阵、控制风险暴露和组合绩效归因分析。

多因子选股模型的整个投资流程包括 alpha 模型的构建，风险模型的构建， 交易成本模型的构建

"""


class BaseFactorModel(metaclass=abc.ABCMeta):
    """
    因子模型的基本属性：

    因子模型的基本方法：
        1. 组合优化：cvxpy+ecos
        2. 计算因子收益Factor loadings
        3. 计算协方差矩阵(风险模型)，计算预测Alpha（收益模型）
        4.
    """
    def __init__(self):
        ...


class BaseRiskModel(BaseFactorModel):

    def __init__(self):
        super().__init__()
        ...


class BaseAlphaModel(BaseFactorModel):

    def __init__(self):
        super().__init__()
        ...
