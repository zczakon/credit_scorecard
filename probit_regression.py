import pandas as pd
import numpy as np
from preprocessor import Preprocessor
from statsmodels.discrete.discrete_model import Probit

pd.set_option('display.max_columns', 10)
df = pd.read_excel("Project 2 - Data.xls")
# print(df.head())

preprocessor = Preprocessor(df)
x_train, y_train, x_test, y_test = preprocessor.combine()

model = Probit(y_train, x_train)
probit_model = model.fit()
probit_preds = probit_model.predict(x_test)


# print(probit_preds)
# print(np.sum(probit_preds))


def predict(prob_default):
    prediction = []
    for probability in prob_default:
        if probability > 0.5:
            prediction.append(1)
        else:
            prediction.append(0)
    return prediction


def model_score(prediction, target):
    score = 0
    for i in range(len(prediction)):
        if prediction[i] != target[i]:  # in this model target variable is "NOT DEFAULT"
            score += 1
    return score / len(prediction)


binary_prediction = predict(probit_preds)
score = model_score(binary_prediction, y_test)
print(binary_prediction)
print('Predicted number of defaults', np.sum(binary_prediction))
print('Score:', score)
