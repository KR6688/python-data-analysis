# -*- coding: utf-8 -*-
"""
作业 4：用 SVM 对课件中的 Spambase 数据进行垃圾邮件分类。

数据：UCI Spambase，4601 封邮件，57 个特征，标签 1 表示垃圾邮件。
运行：python work-4/__init__.py
"""

from pathlib import Path
import urllib.request
import zipfile

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "spambase.data"
RESULT_DIR = ROOT / "results"
DATA_URL = "https://archive.ics.uci.edu/static/public/94/spambase.zip"
SEED = 42


def download_data() -> None:
    """本地没有数据时，从 UCI 下载 Spambase。"""
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        DATA_URL, headers={"User-Agent": "Mozilla/5.0 (SVM homework)"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        content = response.read()

    zip_path = ROOT / "data" / "spambase.zip"
    zip_path.write_bytes(content)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extract("spambase.data", ROOT / "data")
    zip_path.unlink()


def main() -> None:
    if not DATA_PATH.exists():
        print("正在从 UCI 下载 Spambase……")
        download_data()

    # UCI 原文件无表头，最后一列是标签：0 正常邮件，1 垃圾邮件。
    frame = pd.read_csv(DATA_PATH, header=None)
    X = frame.iloc[:, :-1]
    y = frame.iloc[:, -1]
    if X.shape != (4601, 57):
        raise ValueError(f"数据形状不符合课件描述：{X.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=SEED,
        stratify=y,
    )

    # 标准化对 SVM 很重要；放进 Pipeline 可保证只使用训练集计算均值和方差。
    model = make_pipeline(
        StandardScaler(),
        SVC(kernel="linear", C=1.0),
    )
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)

    accuracy = accuracy_score(y_test, prediction)
    f1 = f1_score(y_test, prediction)
    matrix = confusion_matrix(y_test, prediction)

    print(f"训练集：{len(X_train)} 封，测试集：{len(X_test)} 封")
    print(f"测试集准确率：{accuracy:.4f}")
    print(f"测试集 F1：{f1:.4f}")
    print("\n混淆矩阵（行=真实，列=预测）：")
    print(matrix)
    print("\n分类报告：")
    print(
        classification_report(
            y_test,
            prediction,
            target_names=["正常邮件", "垃圾邮件"],
            digits=4,
        )
    )

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "指标": ["Accuracy", "F1"],
            "结果": [accuracy, f1],
        }
    ).to_csv(RESULT_DIR / "metrics.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
