from pathlib import Path
import pandas as pd

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

# ==========================
# 学習
# ==========================

model = train_model(X_train, y_train)

# ==========================
# 予測
# ==========================

# OOFで見つけた最適閾値
best_threshold = 0.34

test_prob, test_pred = predict(
    model,
    X_test,
    threshold=best_threshold
)

# ==========================
# 提出ファイル作成
# ==========================

sample_submit[1] = test_pred

sample_submit.to_csv(
    "submission.csv",
    index=False,
    header=False
)

print("submission.csv を作成しました。")
print(f"予測1 : {(test_pred==1).sum()}")
print(f"予測0 : {(test_pred==0).sum()}")