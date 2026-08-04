#trainとtestからX_train, y_train, X_testを作成
import pandas as pd
import statsmodels.api as sm

TARGET = "購入フラグ"
NUMERICAL_FEATURES = ["従業員数", "当期純利益", "アンケート４"]


def prepare_data(train, test):
    X_train = train[NUMERICAL_FEATURES]
    y_train = train[TARGET]

    X_test = test[NUMERICAL_FEATURES]

    # 切片を追加
    X_train = sm.add_constant(X_train)
    X_test = sm.add_constant(X_test)

    return X_train, y_train, X_test