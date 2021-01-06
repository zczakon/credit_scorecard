from preprocessor import Preprocessor
import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 10)
df = pd.read_excel("Project 2 - Data.xls")

# custom preprocessing
preprocessor = Preprocessor(df)
preprocessor.adjust_excel()
preprocessor.convert_numbers_to_numeric()
df.drop(columns=['ASSESSMENT_YEAR', 'GROUP_FLAG', 'TURNOVER', 'INDUSTRY'], axis=1, inplace=True)

print(df.columns)


class ImpliedModel:
    def __init__(self, data):
        self.data = data.to_numpy()
        self.x = data[:, :6]
        self.y = data[:, 6]

    weights = 0.01 * np.array([20, 10, 10, 15, 25, 20])

    def score(self):
        return np.sum(np.multiply(self.weights, self.x), axis=1) / len(self.weights)

    def pd(self):
        exp = np.exp(-0.1 * self.score())
        denominator = 1 + exp
        return 1 / denominator

    def predict(self):
        prob_default = self.pd()
        prediction = []
        for probability in prob_default:
            if probability > 0.5:
                prediction.append(1)
            else:
                prediction.append(0)
        return prediction

    def model_score(self, y_test):
        score = 0
        prediction = self.predict()
        for i in range(len(prediction)):
            if prediction[i] != y_test[i]:  # in this model target variable is "NOT DEFAULT"
                score += 1
        return score / len(prediction)
