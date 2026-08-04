#呼び出し
from compem.feature import prepare_data
from compem.model import train_model
from compem.predict import predict

X_train, y_train, X_test = prepare_data(train, test)

model = train_model(X_train, y_train)

train_prob, train_pred = predict(model, X_train)
test_prob, test_pred = predict(model, X_test)

from sklearn.metrics import f1_score

f1 = f1_score(y_train, train_pred)
print(f"F1 Score: {f1:.4f}")