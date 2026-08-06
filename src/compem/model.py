from sklearn.linear_model import LogisticRegression


def train_model(X_train, y_train):
    """
    L2正則化付きロジスティック回帰を学習する。
    Cを小さくすることで、TF-IDFによる過学習を抑える。
    """

    model = LogisticRegression(
        C=0.05,
        solver="liblinear",
        max_iter=3000,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model