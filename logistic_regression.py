from sklearn.linear_model import LogisticRegression
import pandas as pd
from preprocessor import Preprocessor

pd.set_option('display.max_columns', 10)
df = pd.read_excel("Project 2 - Data.xls")
# print(df.head())

preprocessor = Preprocessor(df)
x_train, y_train, x_test, y_test = preprocessor.combine()

logisticRegr = LogisticRegression()
logisticRegr.fit(x_train, y_train)
logisticRegr.predict(x_test)
score = logisticRegr.score(x_test, y_test)
print(score)

# TODO add cross validation?