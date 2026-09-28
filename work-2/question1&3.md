# 一、证明：正交设计下 $\boldsymbol{\hat\beta}_{Ridge}=\frac{1}{1+\lambda}\boldsymbol{\hat\beta}_{OLS}$

> **正交设计定义**：设计矩阵 X 满足 $\boldsymbol{X}^T\boldsymbol{X}=\boldsymbol{I}$（单位矩阵）

## 1. 写出两个估计的闭式公式

- OLS 最小二乘估计：

$$
\boldsymbol{\hat\beta}_{OLS}=(\boldsymbol{X}^T\boldsymbol{X})^{-1}\boldsymbol{X}^T\boldsymbol{Y}
$$

- Ridge 岭回归闭式解：

$$
\boldsymbol{\hat\beta}_{Ridge}=(\boldsymbol{X}^T\boldsymbol{X}+\lambda \boldsymbol{I})^{-1}\boldsymbol{X}^T\boldsymbol{Y}
$$

## 2. 代入正交条件 $\boldsymbol{X}^T\boldsymbol{X}=\boldsymbol{I}$

把 $\boldsymbol{X}^T\boldsymbol{X}=\boldsymbol{I}$ 代入 Ridge 公式：

$$
\begin{aligned} \boldsymbol{\hat\beta}_{Ridge} &=(\boldsymbol{I}+\lambda \boldsymbol{I})^{-1}\boldsymbol{X}^T\boldsymbol{Y}\\ &=\big((1+\lambda)\boldsymbol{I}\big)^{-1}\boldsymbol{X}^T\boldsymbol{Y} \end{aligned}
$$

单位矩阵数乘求逆性质：$(k\boldsymbol I)^{-1}=\frac1k \boldsymbol I$

$$
\big((1+\lambda)\boldsymbol{I}\big)^{-1}=\frac{1}{1+\lambda}\boldsymbol I
$$

于是：

$$
\boldsymbol{\hat\beta}_{Ridge}= \frac{1}{1+\lambda}\boldsymbol I \cdot \boldsymbol{X}^T\boldsymbol{Y}
$$

再看 OLS，当 $\boldsymbol X^T\boldsymbol X=\boldsymbol I$：

$$
\boldsymbol{\hat\beta}_{OLS}=(\boldsymbol X^T\boldsymbol X)^{-1}\boldsymbol X^T Y=\boldsymbol I^{-1}\boldsymbol X^T Y=\boldsymbol X^T Y
$$

即 $\boldsymbol X^T Y=\boldsymbol{\hat\beta}_{OLS}$，代入上式：

$$
\boldsymbol{\hat\beta}_{Ridge}= \frac{1}{1+\lambda}\boldsymbol{\hat\beta}_{OLS}
$$





三、为什么正则化前必须 Z-score 标准化？不做的后果？

### 核心原理

Ridge、Lasso、ElasticNet 的惩罚项是对**系数**施加惩罚：

$$
\text{Ridge 损失：} \text{RSS}+\lambda\sum_{j=1}^p \beta_j^2
$$

$\text{Lasso 损失：} \text{RSS}+\lambda\sum_{j=1}^p |\beta_j|$ 惩罚项 $\lambda$ 对**每一个系数 $\beta_j$ 同等惩罚**。 但是：**系数$\beta_j$的大小天然受自变量量纲（单位）影响**。



### 不做标准化的后果

1. **惩罚不公平**：量级大的变量几乎不受惩罚，量级小的变量被过度收缩；Lasso 做变量筛选时，会错误地删掉量纲小但实际重要的变量。
2. **交叉验证选出来的$\lambda$完全不可靠**：CV 选的最优正则强度依赖变量单位，更换变量单位，最优$\lambda$会发生巨大变化。
3. **截距项问题**：若没有中心化（Z-score 自带中心化），截距$\beta_0$也会被惩罚项错误惩罚。**正则回归的惩罚项只惩罚斜率系数，不应该惩罚截距**。