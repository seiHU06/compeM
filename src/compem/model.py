#学習処理を以下に記述
import statsmodels.api as sm


def train_model(X_train, y_train):
    model = sm.Logit(y_train, X_train)
    result = model.fit(disp=0)

    return result