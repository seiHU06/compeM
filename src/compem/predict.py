#予測
def predict(model, X, threshold=0.5):
    prob = model.predict(X)
    pred = (prob > threshold).astype(int)

    return prob, pred