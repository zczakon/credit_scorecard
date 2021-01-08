from preprocessor import Preprocessor
import pandas as pd
import numpy as np

# TODO ML score metrics should be moved elsewhere as they are general

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

    # simple prediction, but could be done using sigmoid etc.
    def predict(self):
        prob_default = self.pd()
        prediction = []
        for probability in prob_default:
            if probability > 0.5:
                prediction.append(1)
            else:
                prediction.append(0)
        return prediction

    # accuracy is (true_positive+true_negative)/total
    def accuracy(self, y_test):
        score = 0
        prediction = self.predict()
        for i in range(len(prediction)):
            if prediction[i] != y_test[i]:  # in this model target variable is "NOT DEFAULT"
                score += 1
        return (self.true_positive(y_test) + self.true_negative(y_test)) / len(prediction)

    def precision(self, y_test):
        true_positive = self.true_positive(y_test)
        false_positive = np.sum(self.predict()) - true_positive
        return np.divide(true_positive, true_positive + false_positive)

    def recall(self, y_test):
        true_positive = self.true_positive(y_test)
        negative = len(y_test) - np.sum(self.predict())
        false_negative = negative - self.true_negative(y_test)

        return np.divide(true_positive, true_positive + false_negative)

    def true_positive(self, y_test):
        true_pos = 0
        prediction = self.predict()
        for i in range(len(prediction)):
            if y_test == 1 & prediction[i] != y_test[i]:
                true_pos += 1
        return true_pos

    def true_negative(self, y_test):
        true_neg = 0
        prediction = self.predict()
        for i in range(len(prediction)):
            if y_test == 0 & prediction[i] != y_test[i]:
                true_neg += 1
        return true_neg

    def f1_model_score(self, y_test):
        return 2 * np.divide(self.precision(y_test) * self.recall(y_test), self.precision(y_test) + self.recall(y_test))
