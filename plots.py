import matplotlib.pyplot as plt
from logistic_regression import score as log_regr_score
from implied_model import score as implied_score
from linear_regression import score as linear_score
from probit_regression import score as probit_score
from preprocessor import Preprocessor
import pandas as pd
from df_template import render_mpl_table
import scorecardpy as sc

scores = [log_regr_score, probit_score, linear_score, implied_score]
models = ['Logit', 'Probit', 'Linear', 'Implied']

plt.barh(y=models, width=scores, height=0.5)
plt.show()

df = pd.read_excel("Project 2 - Data.xls")
preprocessor = Preprocessor(df)
preprocessor.remove_duplicates()
preprocessor.adjust_excel()
iv_table = sc.iv(df, 'DEFAULT_FLAG')
fig, ax = render_mpl_table(iv_table, col_width=4.0)
fig.savefig("iv_table.png")
