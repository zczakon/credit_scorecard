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
y_test = df['DEFAULT_FLAG'].to_numpy()  # here test data is all data


# print(df.columns)

class ImpliedModel:
    def __init__(self, data):
        self.data = data
        self.x = data[:, :6]
        self.y = data[:, 6]

    weights = 0.01 * np.array([20, 10, 10, 15, 25, 20])

    def score(self):
        return (np.sum(np.multiply(self.weights, self.x), axis=1)) / len(self.weights)

    def pd(self):
        exp = np.exp(-0.1 * self.score())
        denominator = 1 + exp
        return 1 / denominator  # probability of non-default #everything evaluates to default ie >0.5 ie 1

    # simple prediction, but could be done using sigmoid etc.
    def predict(self):
        prob_default = self.pd()
        prediction = []
        for probability in prob_default:
            if probability > 0.5:
                prediction.append(1)
            else:
                prediction.append(0)
        # print('prediction:',prediction)
        return prediction  # prediction is all 1

    def true_positive(self, y_test):
        true_pos = 0
        prediction = self.predict()
        for i in range(len(prediction)):
            if y_test[i] == 1 & prediction[i] == y_test[i]:
                true_pos += 1
        # print('true positive:',true_pos)
        return true_pos  # should be around 10% of all OK, those are truly defaulted

    def true_negative(self, y_test):
        true_neg = 0
        prediction = self.predict()
        for i in range(len(prediction)):
            if prediction[i] == 0 & y_test[i] == 0:
                true_neg += 1
        print('true negative:', true_neg)
        return true_neg  # should be 0

    # accuracy is (true_positive+true_negative)/total
    def accuracy(self, y_test):
        score = 0
        prediction = self.predict()
        for i in range(len(prediction)):
            if prediction[i] == y_test[i]:
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
        print('negative', negative)
        return np.divide(true_positive, true_positive + false_negative)

    def f1_model_score(self, y_test):
        return 2 * np.divide(self.precision(y_test) * self.recall(y_test), self.precision(y_test) + self.recall(y_test))

    def model_score(self, y):
        score = 0
        prediction = self.predict()
        for i in range(len(prediction)):
            if prediction[i] == y[i]:
                score += 1
        return score / len(prediction)


implied_model = ImpliedModel(df.to_numpy())
score = implied_model.model_score(y_test)
precision = implied_model.precision(y_test)
recall = implied_model.recall(y_test)

predict_proba = implied_model.pd()
prediction = implied_model.predict()
number_of_predicted_defaults = np.sum(prediction)

print('recall:', recall)
print('precision:', precision)
print('F1 score:', implied_model.f1_model_score(y_test))
print('accuracy:', implied_model.accuracy(y_test))

print('Score:', score)
