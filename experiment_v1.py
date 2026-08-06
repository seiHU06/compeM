from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from compem.feature import prepare_data
from compem.model import train_model
from compem.predict import predict


# ==========================
# データ読み込み
# ==========================

ROOT_DIR = Path.cwd()
DATA_DIR = ROOT_DIR / "data"

train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")

sample_submit = pd.read_csv(
    DATA_DIR / "sample_submit.csv",
    header=None
)


# ==========================
# 前処理
# ==========================

X_train, y_train, X_test = prepare_data(train, test)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape : {X_test.shape}")


# ==========================
# 5-fold Cross Validation
# ==========================

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# 各行のOOF予測確率を保存
oof_prob = np.zeros(len(X_train))

# 各Foldのtest予測確率を平均する
test_prob_sum = np.zeros(len(X_test))

# 各行が何番目のFoldだったか保存
fold_numbers = np.zeros(len(X_train), dtype=int)


for fold, (train_idx, valid_idx) in enumerate(
    skf.split(X_train, y_train),
    start=1
):
    X_tr = X_train.iloc[train_idx]
    y_tr = y_train.iloc[train_idx]

    X_val = X_train.iloc[valid_idx]
    y_val = y_train.iloc[valid_idx]

    # モデル学習
    model = train_model(X_tr, y_tr)

    # 検証データの予測確率
    val_prob, val_pred = predict(
        model,
        X_val,
        threshold=0.5
    )

    # testデータの予測確率
    fold_test_prob, _ = predict(
        model,
        X_test,
        threshold=0.5
    )

    oof_prob[valid_idx] = val_prob
    fold_numbers[valid_idx] = fold

    test_prob_sum += fold_test_prob

    score_05 = f1_score(y_val, val_pred)

    print(
        f"Fold {fold}: "
        f"F1(threshold=0.50)={score_05:.4f}"
    )


# 5モデルのtest予測確率を平均
test_prob = test_prob_sum / skf.n_splits


# ==========================
# OOF予測で閾値を最適化
# ==========================

best_threshold = 0.5
best_oof_score = 0.0

for threshold in np.arange(0.10, 0.901, 0.01):
    oof_pred = (oof_prob >= threshold).astype(int)

    score = f1_score(
        y_train,
        oof_pred
    )

    if score > best_oof_score:
        best_oof_score = score
        best_threshold = float(threshold)


print("=" * 40)
print(f"Best threshold : {best_threshold:.2f}")
print(f"OOF F1         : {best_oof_score:.4f}")
print("=" * 40)


# ==========================
# 最適閾値で各Foldを再評価
# ==========================

fold_scores = []

for fold in range(1, skf.n_splits + 1):
    mask = fold_numbers == fold

    fold_pred = (
        oof_prob[mask] >= best_threshold
    ).astype(int)

    fold_score = f1_score(
        y_train.iloc[np.where(mask)[0]],
        fold_pred
    )

    fold_scores.append(fold_score)

    print(
        f"Fold {fold}: "
        f"F1={fold_score:.4f}"
    )

print("-" * 40)
print(f"Mean Fold F1   : {np.mean(fold_scores):.4f}")
print(f"OOF F1         : {best_oof_score:.4f}")
print("-" * 40)


# ==========================
# 提出用予測
# ==========================

test_pred = (
    test_prob >= best_threshold
).astype(int)


# ==========================
# 提出ファイル作成
# ==========================

if len(sample_submit) != len(test_pred):
    raise ValueError(
        "sample_submit.csvとtest.csvの行数が一致しません。"
        f" sample_submit={len(sample_submit)},"
        f" test={len(test_pred)}"
    )

sample_submit[1] = test_pred

sample_submit.to_csv(
    "submission.csv",
    index=False,
    header=False
)

print("submission.csv を作成しました。")
print(f"購入フラグ1の予測数: {test_pred.sum()}")
print(f"購入フラグ0の予測数: {(test_pred == 0).sum()}")