import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from io import BytesIO, StringIO
from zipfile import ZipFile
import requests

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ========== 1. 加载 Hitters 数据集 ==========
url = "https://www.statlearning.com/s/Hitters.csv"
df = pd.read_csv(url)
df = df.dropna()  # 删除缺失值
# 自变量、因变量
X = df.drop("Salary", axis=1)
y = df["Salary"]
# 类别变量哑变量编码
X = pd.get_dummies(X, drop_first=True)
feature_names = X.columns

# 划分训练集、测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 标准化！正则化模型必须标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ========== 2. 设置λ候选值 ==========
alphas = np.logspace(-3, 6, 100)

# ========== 3. 系数路径图（Coef Path） ==========
fig, axes = plt.subplots(1,3, figsize=(18,5))

# --- Ridge路径
ridge_coefs = []
for a in alphas:
    ridge = Ridge(alpha=a)
    ridge.fit(X_train_scaled, y_train)
    ridge_coefs.append(ridge.coef_)
axes[0].plot(alphas, ridge_coefs)
axes[0].set_xscale("log")
axes[0].set_title("Ridge 系数路径")
axes[0].set_xlabel(r"$\lambda$")
axes[0].set_ylabel("系数")
axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.5)

# --- Lasso路径
lasso_coefs = []
for a in alphas:
    lasso = Lasso(alpha=a, max_iter=10000)
    lasso.fit(X_train_scaled, y_train)
    lasso_coefs.append(lasso.coef_)
axes[1].plot(alphas, lasso_coefs)
axes[1].set_xscale("log")
axes[1].set_title("Lasso 系数路径")
axes[1].set_xlabel(r"$\lambda$")
axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.5)

# --- ElasticNet路径
enet_coefs = []
for a in alphas:
    enet = ElasticNet(alpha=a, l1_ratio=0.5, max_iter=10000)
    enet.fit(X_train_scaled, y_train)
    enet_coefs.append(enet.coef_)
axes[2].plot(alphas, enet_coefs)
axes[2].set_xscale("log")
axes[2].set_title("ElasticNet(l1_ratio=0.5) 系数路径")
axes[2].set_xlabel(r"$\lambda$")
axes[2].axhline(y=0, color='k', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# ========== 4. CV训练模型 cv=10 ==========
ridge_cv = RidgeCV(alphas=alphas, cv=10, scoring="neg_mean_squared_error")
ridge_cv.fit(X_train_scaled, y_train)

lasso_cv = LassoCV(alphas=alphas, cv=10, max_iter=10000)
lasso_cv.fit(X_train_scaled, y_train)

enet_cv = ElasticNetCV(alphas=alphas, l1_ratio=0.5, cv=10, max_iter=10000)
enet_cv.fit(X_train_scaled, y_train)

# ========== 5. 预测、计算RMSE，统计非零系数 ==========
def get_result(model, Xtest, ytest, name):
    y_pred = model.predict(Xtest)
    rmse = np.sqrt(mean_squared_error(ytest, y_pred))
    non_zero = np.sum(np.abs(model.coef_) > 1e-8)
    print(f"==== {name} ====")
    print(f"最优λ(alpha): {model.alpha_:.4f}")
    print(f"测试集RMSE: {rmse:.2f}")
    print(f"非零变量数量: {non_zero}")
    return rmse, non_zero

ridge_rmse, ridge_nz = get_result(ridge_cv, X_test_scaled, y_test, "RidgeCV")
lasso_rmse, lasso_nz = get_result(lasso_cv, X_test_scaled, y_test, "LassoCV")
enet_rmse, enet_nz = get_result(enet_cv, X_test_scaled, y_test, "ElasticNetCV")

# ========== 6. 1-SE 法则（重点！） ==========
"""
1-SE法则思想：
找到交叉验证误差最小的λ0；
允许选取 验证误差 ≤ min_error + 1*SE 的最大λ；
更大λ意味着更强正则，更少变量，模型更稀疏。
sklearn LassoCV / ElasticNetCV 自带参数 n_alphas + 可手动实现；
Ridge 系数不会归零，1-SE对Ridge意义不大，主要讨论Lasso和ElasticNet。
"""
# 取出LassoCV交叉验证各alpha的MSE
cv_mse = lasso_cv.mse_path_.mean(axis=1)
cv_se = lasso_cv.mse_path_.std(axis=1)/np.sqrt(10)
min_idx = np.argmin(cv_mse)
min_mse = cv_mse[min_idx]
threshold = min_mse + cv_se[min_idx]

# 筛选满足 mse <= threshold，并且alpha最大的索引
candidates = np.where(cv_mse <= threshold)[0]
one_se_idx = candidates[-1]
one_se_alpha = lasso_cv.alphas_[one_se_idx]

print("\n==== Lasso 1-SE法则结果 ====")
print(f"CV最小MSE对应的alpha: {lasso_cv.alpha_:.4f}")
print(f"1-SE法则选出的alpha: {one_se_alpha:.4f}")
# 用1SE的alpha重新训练
lasso_1se = Lasso(alpha=one_se_alpha, max_iter=10000)
lasso_1se.fit(X_train_scaled, y_train)
y_pred_1se = lasso_1se.predict(X_test_scaled)
rmse_1se = np.sqrt(mean_squared_error(y_test, y_pred_1se))
nz_1se = np.sum(np.abs(lasso_1se.coef_)>1e-8)
print(f"1-SE Lasso 测试RMSE: {rmse_1se:.2f}")
print(f"1-SE Lasso 非零变量数: {nz_1se}")
print("\n> 结论：1-SE法则牺牲一点点验证集误差，换取更少特征，降低方差，模型更稀疏，泛化能力更稳。")
