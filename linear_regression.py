from sklearn.linear_model import LinearRegression
import pandas as pd
from preprocessor import Preprocessor

pd.set_option('display.max_columns', 10)
df = pd.read_excel("Project 2 - Data.xls")
# print(df.head())

preprocessor = Preprocessor(df)

x_train, y_train, x_test, y_test = preprocessor.combine()

regr = LinearRegression()
regr.fit(x_train, y_train)
regr.predict(x_test)
score = regr.score(x_test, y_test)
print(score)
