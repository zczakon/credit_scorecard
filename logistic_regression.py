from sklearn.linear_model import LogisticRegression
import pandas as pd
from preprocessor import Preprocessor
import numpy as np

pd.set_option('display.max_columns', 10)
df = pd.read_excel("Project 2 - Data.xls")

preprocessor = Preprocessor(df)
x_train, y_train, x_test, y_test = preprocessor.combine()

total_test_defaults = np.sum(y_test)
print('Total defaults in test data: ', total_test_defaults)

logisticRegr = LogisticRegression()
logisticRegr.fit(x_train, y_train)
logisticRegr.predict(x_test)
predict_proba = logisticRegr.predict_proba(x_test)[:, 1]
score = logisticRegr.score(x_test, y_test)
print(score)

