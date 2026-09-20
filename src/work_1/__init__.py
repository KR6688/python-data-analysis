import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 安装依赖：pip install ISLP statsmodels pandas
from ISLP import load_data
Carseats = load_data('Carseats')

# 构建多元线性回归
model = smf.ols(
    formula='Sales ~ Price + Income + Advertising + ShelveLoc',
    data=Carseats
).fit()

# 输出回归报告
print(model.summary())

# 计算VIF方差膨胀因子，检验多重共线性
X = model.model.exog[:,1:]
vif_result = pd.DataFrame({
    "变量名": model.model.exog_names[1:],
    "VIF": [variance_inflation_factor(X, i) for i in range(X.shape[1])]
})
print(vif_result)
