import pandas as pd
import numpy as np
from preprocessor import Preprocessor
from statsmodels.discrete.discrete_model import Probit

pd.set_option('display.max_columns', 10)
df = pd.read_excel("Project 2 - Data.xls")

preprocessor = Preprocessor(df)
x_train, y_train, x_test, y_test = preprocessor.combine()

model = Probit(y_train, x_train)
probit_model = model.fit()
predict_proba = probit_model.predict(x_test)


def predict(predict_proba):
    prediction = []
    for probability in predict_proba:
        if probability > 0.5:
            prediction.append(1)
        else:
            prediction.append(0)
    return prediction


def model_score(prediction, target):
    score = 0
    for i in range(len(prediction)):
        if prediction[i] == target[i]:
            score += 1
    return score / len(prediction)


binary_prediction = predict(predict_proba)
score = model_score(binary_prediction, y_test)

total_test_defaults = np.sum(binary_prediction)

