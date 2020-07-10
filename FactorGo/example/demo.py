
from FactorGo.factor_test.factor_base import FactorDataStruct
from DataGo.fetch import get_index_components, get_stock_fin_indicators

# 取指数成分股
csi300 = get_index_components('000300.XSHG', start_date='2019-01-01', end_date='2019-10-01', return_index=True)

# 取市值数据
csi300_cap = get_stock_fin_indicators(codes_date_index=csi300, indicators=['market_cap'])

cap_fac = FactorDataStruct(factor_data=np.log(csi300_cap), factor_col='market_cap')
print(cap_fac.factor_name)

# 数据处理--去空值
cap_fac.nan_process(inplace=True)

# 也可以用以下方法
# nan_process = FactorNanProcess()
# new_fac = nan_process.fit_transform(cap_fac)

cap_fac.winsorize(inplace=True)
cap_fac.standardize(inplace=True)

print("获取收益率数据")
cap_fac.match_return(inplace=True, periods=['5d', '10d','1m','2q'])

print("获取市值数据")
cap_fac.match_cap(inplace=True)

# ic分析
ic_res = cap_fac.ic_test(plot=False)

ic_res.plot()
