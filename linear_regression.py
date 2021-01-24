from sklearn.linear_model import LinearRegression
import pandas as pd
from preprocessor import Preprocessor
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

pd.set_option('display.max_columns', 10)
df = pd.read_excel("Project 2 - Data.xls")

preprocessor = Preprocessor(df)
# x_train, y_train, x_test, y_test = preprocessor.combine() # for woe

preprocessor.remove_duplicates()
preprocessor.adjust_excel()  # similar iv removed
preprocessor.convert_numbers_to_numeric()
train, test = preprocessor.split()
train = preprocessor.encode_categorical(train)
test = preprocessor.encode_categorical(test)
x_train, y_train, x_test, y_test = preprocessor.provide_x_y(train, test)

preprocessing.scale(x_train, axis=0)
preprocessing.scale(y_train, axis=0)
preprocessing.scale(x_test, axis=0)
preprocessing.scale(y_test, axis=0)

total_test_defaults = np.sum(y_test)

# data is normalized (removed mean) but not standardized
regr = LinearRegression(normalize=True)
regr.fit(x_train, y_train)
prediction = regr.predict(x_test)
score = regr.score(x_test, y_test)

print(plt.plot(prediction))

print('Score:', score)

residuals = y_test - prediction
plt.scatter(residuals, prediction)
plt.show()
