def predict(model, X, threshold=0.5):

    prob = model.predict_proba(X)[:, 1]

    pred = (prob >= threshold).astype(int)

    return prob, pred