import pandas as pd
import numpy as np
import scorecardpy as sc


class Preprocessor:
    def __init__(self, data):
        self.df = data

    def adjust_excel(self):
        # adjusting df
        self.df.drop('Unnamed: 0', axis=1, inplace=True)
        self.df.drop('Unnamed: 1', axis=1, inplace=True)
        self.rename_columns()
        self.df.drop(self.df.tail(3).index, inplace=True)
        return self.df

    def rename_columns(self):
        keys = ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5', 'Unnamed: 6', 'Unnamed: 7',
                'Unnamed: 8', 'Unnamed: 9', 'Unnamed: 10', 'Unnamed: 11', 'Unnamed: 12']
        values = ['ASSESSMENT_YEAR', 'PRODUCT_DEMAND', 'OWNERS_MANAGEMENT', 'ACCESS_CREDIT', 'PROFITABILITY',
                  'SHORT_TERM_LIQUIDITY', 'MEDIUM_TERM_LIQUIDITY', 'GROUP_FLAG', 'TURNOVER', 'INDUSTRY', 'DEFAULT_FLAG']
        d = {}
        for i in range(len(keys)):
            d[keys[i]] = values[i]
        self.df.rename(columns=d, inplace=True)
        self.df.drop(self.df.index[0], inplace=True)
        pass

    # preprocessing
    def convert_numbers_to_numeric(self):
        to_convert = []
        for column in self.df.columns:
            if column != 'INDUSTRY':
                to_convert.append(column)
        # print(to_convert)
        self.df[to_convert] = self.df[to_convert].apply(pd.to_numeric, errors='coerce')

    @staticmethod
    def encode_categorical(train_or_test):
        return sc.one_hot(train_or_test, cols_skip=['TURNOVER', 'PRODUCT_DEMAND', 'ACCESS_CREDIT', 'OWNERS_MANAGEMENT',
                                                    'SHORT_TERM_LIQUIDITY', 'MEDIUM_TERM_LIQUIDITY', 'PROFITABILITY',
                                                    'ASSESSMENT_YEAR'], cols_encode='INDUSTRY')

    def split(self):
        train, test = sc.split_df(self.df, y='DEFAULT_FLAG', ratio=0.8, seed=186).values()
        return train, test

    def woe_transform(self, train, test):
        # includes var filtering and one-hot encoding of 'INDUSTRY' column in all data
        train = sc.var_filter(train, 'DEFAULT_FLAG', var_kp='INDUSTRY')
        self.encode_categorical(train)
        bins = sc.woebin(train, 'DEFAULT_FLAG')
        train_woe = sc.woebin_ply(train, bins)
        train_columns = ['ACCESS_CREDIT', 'ASSESSMENT_YEAR', 'MEDIUM_TERM_LIQUIDITY', 'OWNERS_MANAGEMENT',
                         'PRODUCT_DEMAND',
                         'PROFITABILITY', 'SHORT_TERM_LIQUIDITY', 'TURNOVER', 'DEFAULT_FLAG', 'INDUSTRY']
        test_selected = test[train_columns]
        self.encode_categorical(test_selected)
        test_woe = sc.woebin_ply(test_selected, bins)

        return train_woe, test_woe

    @staticmethod
    def provide_x_y(train, test):
        # TODO it might change train test permanently to numpy?
        train = train.to_numpy()
        test = test.to_numpy()

        n = np.shape(train)[1]

        x_train = train[:, 1:n]
        y_train = train[:, 0]
        x_test = test[:, 1:n]
        y_test = test[:, 0]

        return x_train, y_train, x_test, y_test

    def combine(self):
        self.adjust_excel()
        self.convert_numbers_to_numeric()
        train, test = self.split()
        train_woe, test_woe = self.woe_transform(train, test)
        x_train, y_train, x_test, y_test = self.provide_x_y(train_woe, test_woe)

        return x_train, y_train, x_test, y_test
