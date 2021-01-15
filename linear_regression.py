from sklearn.linear_model import LinearRegression
import pandas as pd
from preprocessor import Preprocessor
import numpy as np

pd.set_option('display.max_columns', 10)
df = pd.read_excel("Project 2 - Data.xls")

preprocessor = Preprocessor(df)

x_train, y_train, x_test, y_test = preprocessor.combine()

total_test_defaults = np.sum(y_test)

regr = LinearRegression()
regr.fit(x_train, y_train)
prediction =regr.predict(x_test)
print(prediction)
score = regr.score(x_test, y_test)
print(score)
