# -*- coding: utf-8 -*-
"""
作业：使用 Elastic Net 对 Kaggle 上的 Netflix 数据做分析
========================================================
数据来源：Kaggle - "Netflix Movies and TV Shows" (shivamb/netflix-shows)
          https://www.kaggle.com/datasets/shivamb/netflix-shows
          共 8807 条影视条目、12 个字段。

分析目标：
    数据集中唯一的连续型变量是「电影时长（分钟）」，
    因此以 Movie 的 duration（分钟）为因变量，用 Elastic Net 做正则化回归，
    考察题材(genre)、国家、分级(rating)、上映年份、上线时间、演职员规模
    等特征对电影时长的影响。

    由于类别变量多、且存在强共线性（例如 内容库龄 = 上线年份 − 上映年份），
    这正是 Elastic Net（L1 变量筛选 + L2 分组效应）最合适的场景。
    最后与 OLS、Ridge、Lasso 对比，说明 Elastic Net 的取舍。

运行方式：python work-3/__init__.py
依赖：pandas  numpy  scikit-learn  matplotlib
"""

from pathlib import Path
import sys
import urllib.request

import matplotlib

matplotlib.use("Agg")  # 无界面保存图片
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import (
    ElasticNet,
    ElasticNetCV,
    LassoCV,
    LinearRegression,
    RidgeCV,
)
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib.ticker import FuncFormatter

# 中文绘图设置
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.formatter.use_mathtext"] = False  # 负号用普通字符，避免缺字形

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "netflix_titles.csv"
FIG_DIR = ROOT / "figures"
SEED = 42

KAGGLE_URL = "https://www.kaggle.com/api/v1/datasets/download/shivamb/netflix-shows"


# ========== 0. 工具函数 ==========
def ensure_data():
    """本地没有数据时，自动从 Kaggle 下载（公开数据集，无需登录）。"""
    if DATA_PATH.exists():
        return DATA_PATH
    import zipfile

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_PATH.parent / "netflix-shows.zip"
    print("本地未找到数据，正在从 Kaggle 下载 ...")
    urllib.request.urlretrieve(KAGGLE_URL, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(DATA_PATH.parent)
    zip_path.unlink()
    return DATA_PATH


def multi_hot(series, categories, prefix):
    """把 "A, B, C" 这类多值字段拆成 0/1 哑变量。

    注意：不能用 str.contains()，因为 "Dramas" 会误命中 "TV Dramas"，
    必须按逗号拆分后再精确匹配。
    """
    items = series.fillna("").str.split(", ")
    data = {
        f"{prefix}_{cat}": items.apply(lambda xs, cat=cat: int(cat in xs))
        for cat in categories
    }
    return pd.DataFrame(data, index=series.index)


def nonzero_count(model):
    return int(np.sum(np.abs(model.coef_) > 1e-8))


# ========== 1. 读取数据 ==========
def load_data():
    df = pd.read_csv(ensure_data())
    print("原始数据形状:", df.shape)
    return df


# ========== 2. 数据清洗 ==========
def clean_data(df):
    df = df.copy()

    # 2.1 文本字段去首尾空格
    for col in ["type", "title", "country", "rating", "duration",
                "listed_in", "cast", "director"]:
        df[col] = df[col].astype("string").str.strip()

    # 2.2 上线日期转成 datetime
    df["date_added"] = pd.to_datetime(
        df["date_added"].astype("string").str.strip(),
        format="%B %d, %Y",
        errors="coerce",
    )

    # 2.3 修正 rating 与 duration 写反的脏数据
    #     （例：Louis C.K. 的脱口秀专场，rating 里填的是 "74 min"）
    swapped = df["rating"].astype("string").str.contains("min", na=False)
    print(f"\n发现 rating / duration 写反的脏数据 {int(swapped.sum())} 条，已交换修复")
    df.loc[swapped, "duration"] = df.loc[swapped, "rating"]
    df.loc[swapped, "rating"] = pd.NA

    # 2.4 取出电影时长（分钟）；TV Show 的 duration 是 "N Seasons"，取不到值
    is_movie = df["type"] == "Movie"
    minutes = (df["duration"].astype("string").str.extract(r"^(\d+)\s*min$")[0]
               .astype(float))
    df["duration_minutes"] = minutes.where(is_movie)

    movies = df[df["duration_minutes"].notna()].copy()
    print(f"电影样本数 {len(movies)}（原始 Movie {int(is_movie.sum())} 条，"
          f"{int((is_movie & df['duration_minutes'].isna()).sum())} 条时长缺失被剔除）")

    # 2.5 衍生特征：上线年份/月份、内容库龄（上线年份 − 上映年份）
    movies["year_added"] = movies["date_added"].dt.year
    movies["month_added"] = movies["date_added"].dt.month
    movies["content_age"] = movies["year_added"] - movies["release_year"]
    movies = movies[movies["year_added"].notna()].copy()

    print("清洗后建模样本:", movies.shape)
    print("关键字段缺失情况:\n",
          movies[["director", "cast", "country", "rating"]].isna().sum().to_string())
    return movies


# ========== 3. 探索性分析（EDA） ==========
def eda(movies):
    print("\n电影时长统计: 均值 %.1f 分钟, 中位数 %.1f, 标准差 %.1f, 范围 %d~%d"
          % (movies["duration_minutes"].mean(), movies["duration_minutes"].median(),
             movies["duration_minutes"].std(), movies["duration_minutes"].min(),
             movies["duration_minutes"].max()))

    year_count = movies.groupby("year_added").size()
    genre_count = movies["listed_in"].str.split(", ").explode().value_counts().head(15)
    rating_count = movies["rating"].value_counts().head(10)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # (a) 时长分布
    axes[0, 0].hist(movies["duration_minutes"], bins=45, color="#4C72B0",
                    edgecolor="white")
    axes[0, 0].axvline(movies["duration_minutes"].mean(), color="crimson",
                       linestyle="--",
                       label="均值 %.0f 分钟" % movies["duration_minutes"].mean())
    axes[0, 0].set_title("(a) 电影时长分布")
    axes[0, 0].set_xlabel("时长（分钟）")
    axes[0, 0].set_ylabel("影片数量")
    axes[0, 0].legend()

    # (b) 上线年份
    axes[0, 1].bar(year_count.index, year_count.values, color="#55A868")
    axes[0, 1].set_title("(b) 每年上线 Netflix 的电影数量")
    axes[0, 1].set_xlabel("上线年份")
    axes[0, 1].set_ylabel("影片数量")

    # (c) 题材
    axes[1, 0].barh(genre_count.index[::-1], genre_count.values[::-1], color="#C44E52")
    axes[1, 0].set_title("(c) 电影题材 Top 15")
    axes[1, 0].set_xlabel("影片数量")

    # (d) 分级
    axes[1, 1].bar(rating_count.index, rating_count.values, color="#8172B2")
    axes[1, 1].set_title("(d) 电影分级分布 Top 10")
    axes[1, 1].set_xlabel("分级")
    axes[1, 1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "01_eda.png", dpi=130)
    plt.close(fig)
    print("已保存 图1 EDA 总览 -> figures/01_eda.png")


# ========== 4. 特征工程 ==========
def build_features(movies, top_genres=15, top_countries=10, min_rating=30):
    y = movies["duration_minutes"].astype(float)

    # 4.1 题材、国家：多值字段 → 哑变量
    genres = (movies["listed_in"].str.split(", ").explode()
              .value_counts().head(top_genres).index)
    countries = (movies["country"].str.split(", ").explode()
                 .value_counts().head(top_countries).index)
    X_genre = multi_hot(movies["listed_in"], genres, "题材")
    X_country = multi_hot(movies["country"], countries, "国家")

    # 4.2 分级：出现次数太少的合并为「其他分级」
    rating = movies["rating"].fillna("未知").astype("string")
    keep_rating = rating.value_counts()[lambda s: s >= min_rating].index
    rating = rating.where(rating.isin(keep_rating), "其他分级")
    X_rating = pd.get_dummies(rating, prefix="分级", dtype=int)

    # 4.3 数值特征
    cast_count = movies["cast"].fillna("").apply(
        lambda s: 0 if not s else len(s.split(", ")))
    X_num = pd.DataFrame({
        "上映年份": movies["release_year"].astype(float),
        "上线年份": movies["year_added"].astype(float),
        "上线月份": movies["month_added"].astype(float),
        "内容库龄": movies["content_age"].astype(float),
        "演员人数": cast_count.astype(float),
        "简介字数": movies["description"].str.len().astype(float),
        "简介词数": movies["description"].str.split().apply(len).astype(float),
        "有导演": movies["director"].notna().astype(int),
        "有国家信息": movies["country"].notna().astype(int),
        "有演员信息": movies["cast"].notna().astype(int),
    }, index=movies.index)

    X = pd.concat([X_num, X_genre, X_country, X_rating], axis=1)
    print(f"\n特征矩阵: {X.shape[0]} 个样本 × {X.shape[1]} 个特征")
    print("  数值 %d 个 / 题材哑变量 %d 个 / 国家哑变量 %d 个 / 分级哑变量 %d 个"
          % (X_num.shape[1], X_genre.shape[1], X_country.shape[1], X_rating.shape[1]))
    return X, y


# ========== 5. 划分数据集 + 标准化 ==========
def scale_split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )
    # 正则化的惩罚项直接作用在系数上，必须做 Z-score 标准化，
    # 否则量纲大的变量（上映年份、简介字数等）会被过度收缩（见作业 2 第 3 题）
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    print(f"训练集 {X_train_s.shape[0]} 条 / 测试集 {X_test_s.shape[0]} 条")
    return X_train_s, X_test_s, y_train, y_test, X.columns.tolist()


# ========== 6. Elastic Net 交叉验证调参 ==========
ENET_ALPHAS = np.logspace(-3, 2, 60)
L1_RATIOS = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]


def fit_elastic_net(X_train, y_train):
    enet_cv = ElasticNetCV(
        alphas=ENET_ALPHAS,
        l1_ratio=L1_RATIOS,
        cv=10,
        max_iter=10000,
        n_jobs=-1,
        random_state=SEED,
    )
    enet_cv.fit(X_train, y_train)

    print("\n==== ElasticNetCV 调参结果（10 折交叉验证） ====")
    print(f"最优 alpha (λ) = {enet_cv.alpha_:.4f}")
    print(f"最优 l1_ratio  = {enet_cv.l1_ratio_:.2f}  （1.0 相当于纯 Lasso，越小越接近 Ridge）")
    return enet_cv


# ========== 7. 与 OLS / Ridge / Lasso 对比 ==========
def compare_models(X_train, X_test, y_train, y_test, columns, enet_cv):
    models = {
        "OLS 普通最小二乘": LinearRegression(),
        "Ridge (L2)": RidgeCV(alphas=ENET_ALPHAS, cv=10),
        "Lasso (L1)": LassoCV(alphas=ENET_ALPHAS, cv=10, max_iter=10000,
                              n_jobs=-1, random_state=SEED),
        "Elastic Net": enet_cv,
    }

    rows = []
    for name, model in models.items():
        if name != "Elastic Net":
            model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rows.append({
            "模型": name,
            "测试集 R²": r2_score(y_test, y_pred),
            "测试集 RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "非零系数个数": nonzero_count(model),
        })

    table = pd.DataFrame(rows)
    table["保留特征比例"] = table["非零系数个数"] / len(columns)
    print("\n==== 四种线性模型对比（同一训练 / 测试划分） ====")
    print(table.to_string(index=False,
                          formatters={"测试集 R²": "{:.4f}".format,
                                      "测试集 RMSE": "{:.3f}".format,
                                      "保留特征比例": "{:.1%}".format}))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(table["模型"], table["测试集 R²"], color="#4C72B0")
    axes[0].set_title("测试集决定系数 R2（越高越好）")
    axes[0].set_ylim(0, max(0.4, table["测试集 R²"].max() * 1.35))
    for i, v in enumerate(table["测试集 R²"]):
        axes[0].text(i, v + 0.008, f"{v:.3f}", ha="center")
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].bar(table["模型"], table["非零系数个数"], color="#C44E52")
    axes[1].axhline(len(columns), color="gray", linestyle="--",
                    label=f"全部特征 {len(columns)} 个")
    axes[1].set_title("非零系数个数（模型稀疏度）")
    for i, v in enumerate(table["非零系数个数"]):
        axes[1].text(i, v + 0.6, str(v), ha="center")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "04_model_compare.png", dpi=130)
    plt.close(fig)
    print("已保存 图4 模型对比 -> figures/04_model_compare.png")
    return table


# ========== 8. 系数路径与 CV 误差曲面 ==========
def plot_paths(enet_cv, X_train, y_train):
    # 8.1 固定 CV 选出的 l1_ratio，画系数路径
    path_alphas = np.logspace(-3, 2, 60)
    coefs, n_nonzero = [], []
    for a in path_alphas:
        m = ElasticNet(alpha=a, l1_ratio=float(enet_cv.l1_ratio_), max_iter=10000)
        m.fit(X_train, y_train)
        coefs.append(m.coef_)
        n_nonzero.append(nonzero_count(m))
    coefs = np.array(coefs)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].plot(path_alphas, coefs, linewidth=1)
    axes[0].set_xscale("log")
    axes[0].xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axes[0].axvline(enet_cv.alpha_, color="k", linestyle="--", alpha=0.7,
                    label=f"CV 最优 α={enet_cv.alpha_:.3f}")
    axes[0].axhline(0, color="gray", linewidth=0.8)
    axes[0].set_xlabel("正则强度 λ (alpha)")
    axes[0].set_ylabel("回归系数（标准化尺度）")
    axes[0].set_title(f"Elastic Net 系数路径（l1_ratio={enet_cv.l1_ratio_:.2f}）")
    axes[0].legend()

    axes[1].plot(path_alphas, n_nonzero, color="#C44E52", linewidth=2)
    axes[1].set_xscale("log")
    axes[1].xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axes[1].axvline(enet_cv.alpha_, color="k", linestyle="--", alpha=0.7)
    axes[1].set_xlabel("正则强度 λ (alpha)")
    axes[1].set_ylabel("非零系数个数")
    axes[1].set_title("λ 越大，被压缩为 0 的特征越多")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "02_coef_path.png", dpi=130)
    plt.close(fig)
    print("已保存 图2 系数路径 -> figures/02_coef_path.png")

    # 8.2 l1_ratio × alpha 的交叉验证误差曲面
    #     l1_ratio 传入列表时，mse_path_ 的形状是 (n_l1_ratio, n_alpha, n_folds)
    mse = enet_cv.mse_path_.mean(axis=2)
    best_row = int(np.argmin(np.abs(np.array(L1_RATIOS) - float(enet_cv.l1_ratio_))))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    mesh = ax.pcolormesh(np.log10(ENET_ALPHAS), np.arange(len(L1_RATIOS)), mse,
                         cmap="viridis_r", shading="nearest")
    ax.scatter([np.log10(enet_cv.alpha_)], [best_row], marker="*", s=320,
               color="red", edgecolor="white", zorder=3, label="CV 最优点")
    ax.set_yticks(np.arange(len(L1_RATIOS)))
    ax.set_yticklabels([f"{r:g}" for r in L1_RATIOS])
    ax.set_xlabel("log10(λ)")
    ax.set_ylabel("l1_ratio（越大越像 Lasso，越小越像 Ridge）")
    ax.set_title("10 折交叉验证 MSE 曲面")
    ax.legend()
    fig.colorbar(mesh, ax=ax, label="CV 均方误差 MSE")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "03_cv_surface.png", dpi=130)
    plt.close(fig)
    print("已保存 图3 CV 误差曲面 -> figures/03_cv_surface.png")


# ========== 9. 1-SE 法则 ==========
def one_se_rule(enet_cv, X_train, y_train, X_test, y_test):
    """在最优 l1_ratio 对应的那一行里，选出「CV 误差 ≤ 最小值 + 1 个标准误」
    中 λ 最大（最稀疏）的模型：牺牲一点验证误差，换取更少特征、更低方差。"""
    best_row = int(np.argmin(np.abs(np.array(L1_RATIOS) - float(enet_cv.l1_ratio_))))
    path = enet_cv.mse_path_[best_row]           # (n_alpha, n_folds)
    cv_mean = path.mean(axis=1)
    cv_se = path.std(axis=1) / np.sqrt(path.shape[1])

    i_min = int(np.argmin(cv_mean))
    threshold = cv_mean[i_min] + cv_se[i_min]
    candidates = np.where(cv_mean <= threshold)[0]
    # 注意：ElasticNetCV 的 alphas_ 是「从大到小」排列的（本轮实测已确认），
    # 所以不能简单取 candidates[-1]，要按 alpha 的实际取值挑最大的那个
    i_1se = int(candidates[np.argmax(enet_cv.alphas_[candidates])])

    alpha_1se = float(enet_cv.alphas_[i_1se])
    model_1se = ElasticNet(alpha=alpha_1se, l1_ratio=float(enet_cv.l1_ratio_),
                           max_iter=10000).fit(X_train, y_train)

    print("\n==== 1-SE 法则 ====")
    print(f"CV 最优    : α = {enet_cv.alpha_:.4f}, CV MSE = {cv_mean[i_min]:.2f}, "
          f"非零系数 = {nonzero_count(enet_cv)}")
    print(f"1-SE 法则  : α = {alpha_1se:.4f}, CV MSE = {cv_mean[i_1se]:.2f}, "
          f"非零系数 = {nonzero_count(model_1se)}")
    print("测试集 RMSE: 最优 α = %.3f 分钟, 1-SE = %.3f 分钟"
          % (np.sqrt(mean_squared_error(y_test, enet_cv.predict(X_test))),
             np.sqrt(mean_squared_error(y_test, model_1se.predict(X_test)))))
    return alpha_1se, nonzero_count(model_1se)


# ========== 10. 系数解读与模型诊断 ==========
def interpret(enet_cv, columns, X_train, X_test, y_train, y_test):
    coef = pd.Series(enet_cv.coef_, index=columns)
    kept = coef[coef.abs() > 1e-8].sort_values(key=np.abs, ascending=False)

    print("\n==== Elastic Net 保留下来的特征（按系数绝对值排序） ====")
    print(kept.round(3).to_string())
    print(f"\n共 {len(columns)} 个特征，保留 {len(kept)} 个，"
          f"压缩掉 {len(columns) - len(kept)} 个"
          f"（{1 - len(kept) / len(columns):.1%} 被剔除）")

    ols = LinearRegression().fit(X_train, y_train)
    ols_coef = pd.Series(ols.coef_, index=columns)
    print("\n对比：OLS（未正则化）系数绝对值最大的 8 个特征")
    print(ols_coef.reindex(ols_coef.abs().sort_values(ascending=False).index)
          .head(8).round(3).to_string())

    show = kept.head(15)[::-1]
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = ["#C44E52" if v < 0 else "#4C72B0" for v in show.values]
    ax.barh(show.index, show.values, color=colors)
    ax.axvline(0, color="k", linewidth=0.8)
    ax.set_xlabel("回归系数（因变量：电影时长 / 分钟；特征已标准化）")
    ax.set_title(f"Elastic Net 系数 Top 15（α={enet_cv.alpha_:.3f}, "
                 f"l1_ratio={enet_cv.l1_ratio_:.2f}）")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "05_coefficients.png", dpi=130)
    plt.close(fig)
    print("已保存 图5 系数图 -> figures/05_coefficients.png")

    y_pred = enet_cv.predict(X_test)
    resid = np.asarray(y_test) - y_pred
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    axes[0].scatter(y_test, y_pred, s=8, alpha=0.35, color="#4C72B0")
    lim = [y_test.min(), y_test.max()]
    axes[0].plot(lim, lim, "r--", label="y = x")
    axes[0].set_xlabel("真实时长（分钟）")
    axes[0].set_ylabel("预测时长（分钟）")
    axes[0].set_title("测试集：预测值 vs 真实值")
    axes[0].legend()

    axes[1].scatter(y_pred, resid, s=8, alpha=0.35, color="#55A868")
    axes[1].axhline(0, color="r", linestyle="--")
    axes[1].set_xlabel("预测时长（分钟）")
    axes[1].set_ylabel("残差")
    axes[1].set_title("残差图")

    axes[2].hist(resid, bins=45, color="#8172B2", edgecolor="white")
    axes[2].set_xlabel("残差（分钟）")
    axes[2].set_ylabel("频数")
    axes[2].set_title("残差分布")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "06_diagnostics.png", dpi=130)
    plt.close(fig)
    print("已保存 图6 模型诊断 -> figures/06_diagnostics.png")

    return kept, ols_coef, y_pred, resid


# ========== 11. 分组效应（Elastic Net 相对 Lasso 的核心优势） ==========
def grouping_effect(X_train, y_train, columns, alpha=0.5):
    """对高度相关的特征，Lasso 往往只随机保留其中一个，
    而 Elastic Net 的 L2 部分会把它们「一起留下或一起剔除」。

    这里挑出两组天然高度相关的特征做对照：
        简介字数 / 简介词数  （几乎是同一信息的两种度量）
        上映年份 / 上线年份 / 内容库龄（内容库龄 = 上线年份 − 上映年份）
    """
    # 前两组是真正的共线特征对，用于画图对照
    pairs = [
        ("上映年份", "内容库龄"),
        ("简介字数", "简介词数"),
        ("上映年份", "上线年份"),
    ]
    ratios = [1.0, 0.7, 0.5, 0.3, 0.1]
    idx = {name: i for i, name in enumerate(columns)}

    print("\n==== 分组效应对照（固定 α = %.2f，只改 l1_ratio） ====" % alpha)
    for f1, f2 in pairs:
        print(f"\n[{f1}] 与 [{f2}]   (两者相关系数 %.3f)"
              % np.corrcoef(X_train[:, idx[f1]], X_train[:, idx[f2]])[0, 1])
        print(f"{'l1_ratio':>9} | {'L1 部分':>8} | {f1:>10} | {f2:>10}")
        for r in ratios:
            m = ElasticNet(alpha=alpha, l1_ratio=r, max_iter=10000)
            m.fit(X_train, y_train)
            tag = "纯 Lasso" if r == 1.0 else ("偏 Ridge" if r <= 0.3 else "两者混合")
            print(f"{r:>9.2f} | {tag:>8} | {m.coef_[idx[f1]]:>10.3f} | "
                  f"{m.coef_[idx[f2]]:>10.3f}")

    # 画图：两组共线特征的系数随 l1_ratio 的变化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (f1, f2) in zip(axes, pairs[:2]):
        corr = np.corrcoef(X_train[:, idx[f1]], X_train[:, idx[f2]])[0, 1]
        c1, c2 = [], []
        for r in ratios:
            m = ElasticNet(alpha=alpha, l1_ratio=r, max_iter=10000).fit(X_train, y_train)
            c1.append(m.coef_[idx[f1]])
            c2.append(m.coef_[idx[f2]])
        x = np.arange(len(ratios))
        ax.bar(x - 0.2, c1, width=0.4, label=f1, color="#4C72B0")
        ax.bar(x + 0.2, c2, width=0.4, label=f2, color="#C44E52")
        ax.axhline(0, color="k", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{r:g}" for r in ratios])
        ax.set_xlabel("l1_ratio（1.0 = 纯 Lasso）")
        ax.set_ylabel("回归系数")
        ax.set_title(f"{f1} 与 {f2}（相关系数 {corr:.2f}）")
        ax.legend()
    fig.suptitle(f"分组效应：共线特征的系数随 l1_ratio 的变化（固定 α = {alpha}）")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "07_grouping_effect.png", dpi=130)
    plt.close(fig)
    print("\n已保存 图7 分组效应 -> figures/07_grouping_effect.png")


# ========== 主流程 ==========
def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    movies = clean_data(df)
    eda(movies)
    X, y = build_features(movies)
    X_train, X_test, y_train, y_test, columns = scale_split(X, y)

    enet_cv = fit_elastic_net(X_train, y_train)
    table = compare_models(X_train, X_test, y_train, y_test, columns, enet_cv)
    plot_paths(enet_cv, X_train, y_train)
    alpha_1se, nz_1se = one_se_rule(enet_cv, X_train, y_train, X_test, y_test)
    kept, ols_coef, y_pred, resid = interpret(
        enet_cv, columns, X_train, X_test, y_train, y_test)
    grouping_effect(X_train, y_train, columns, alpha=float(enet_cv.alpha_))

    print("\n==== 结论 ====")
    print(f"1. Elastic Net（α={enet_cv.alpha_:.3f}, l1_ratio={enet_cv.l1_ratio_:.2f}）"
          f"测试集 R² = {r2_score(y_test, y_pred):.4f}，"
          f"RMSE = {np.sqrt(mean_squared_error(y_test, y_pred)):.2f} 分钟；")
    print(f"2. 它把 {len(columns)} 个特征压缩到 {len(kept)} 个；"
          f"1-SE 法则（α={alpha_1se:.3f}）进一步压缩到 {nz_1se} 个；")
    print("3. 电影时长主要由题材、演员人数、分级等结构性特征决定，"
          "年份类变量的解释力很弱。")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
