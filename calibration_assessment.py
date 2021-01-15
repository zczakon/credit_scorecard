# import pandas as pd
import numpy as np
from scipy.stats import norm


# division to groups and all the test should be performed on... test data

class Utils:
    def __init__(self, data):
        self.data = data

    def divide_to_groups(self, predicted_probability, n=10):
        """
        :param predicted_probability: 1d array of PDs (nums between 0&1, 1=certain default)
        :param n: desired number of groups, defaults to 10
        :return: list od 2d arrays of PDs and default flagst
        """
        observations = self.data[:, np.shape(self.data)[1] - 1]
        num_of_obs = np.shape(observations)[0]

        obs_and_predictions = np.column_stack((observations, predicted_probability))
        group_size = num_of_obs // n

        groups = []
        for i in range(n):
            if i <= n - 2:
                groups.append(obs_and_predictions[i * group_size:(i + 1) * group_size, :])
            else:
                groups.append(obs_and_predictions[(n - 1) * group_size:, :])
        return groups

    @staticmethod
    def avg_group_proba(group):
        """
        :param group: 2d array of PDs and default flags
        :return: average PD in a group
        """
        pds = group[:, 1] #sum of probabilities of defaults
        return np.sum(pds) / len(pds)

    def sorted_groups(self, predicted_probability, n=10):
        """
        :param predicted_probability: 1d array of predicted default probabilities (1=certain default)
        :param n: number of groups
        :return: n groups of pairs in an array [realized default flag, PD](<-somehow in this order) sorted by avg groups PD in descending order
        """
        groups = self.divide_to_groups(predicted_probability, n)
        avg_probs = [self.avg_group_proba(group) for group in groups]

        # rows in array are pairs of group, avg_prob, I sort them by avg probability
        groups = np.column_stack((groups, avg_probs))

        sorted_groups = groups[groups[:, 1].argsort()]
        sorted_groups = np.flip(sorted_groups, axis=0)
        # print(sorted_groups[:,0])

        return sorted_groups[:, 0]


class CalibrationMetrics(Utils):
    def __init__(self, data):
        super().__init__(data)
        self.data = data

    # calibration accuracy
    def hosmer_lemeshow(self, groups):
        """
        :groups: optional, list of 2d arrays of PDs and default flags
        :return: list of hosmer lemeshow statistic values for given groups
        """
        hs = []
        for group in groups:
            total_number = np.shape(group)[0]
            number_of_defaults = np.sum(group[:, 0])
            realized_default_rate = number_of_defaults / total_number
            predicted_default_rate = self.avg_group_proba(group)
            hs_stat = np.divide((realized_default_rate - predicted_default_rate) ** 2,
                                predicted_default_rate * (1 - predicted_default_rate)) * total_number
            hs.append(hs_stat)
        return hs

    def brier_score(self, groups):
        total_borrowers = 0
        bs_group = []
        for group in groups:
            borrowers_in_group = np.shape(group)[0]
            total_borrowers += borrowers_in_group
            number_of_defaults = np.sum(group[:, 0])
            realized_default_rate = number_of_defaults / borrowers_in_group
            predicted_default_rate = self.avg_group_proba(group)
            summed = realized_default_rate * (1 - realized_default_rate) + (
                    predicted_default_rate - realized_default_rate) ** 2
            # print('summed type:',type(summed))
            # print('borrowers_in_group type:',type(borrowers_in_group))
            # print(borrowers_in_group * summed)

            bs_group.append(borrowers_in_group * summed)

        # print(bs_group)
        # print(np.sum(bs_group))

        bs = (1 / total_borrowers) * np.sum(bs_group)

        # print(bs)
        return bs

    def brier_skill_score(self, groups, total_defaults, num_of_obs=1161):
        pd_observed = total_defaults / num_of_obs
        return 1 - np.divide(self.brier_score(groups), pd_observed * (1 - pd_observed))

    def normal_approximation_bounds(self, group, q):
        borrowers_in_group = np.shape(group)[0]
        # defaulted_in_group = np.sum(group[:, 1])
        g_avg = self.avg_group_proba(group)
        upper_bound = norm.pps((q + 1) / 2) * np.sqrt(np.divide((g_avg * (1 - g_avg)), borrowers_in_group))
        lower_bound = g_avg - upper_bound
        return lower_bound, upper_bound

    def normal_approximation_test(self, group):
        # 3=red, 2=yellow, 1=green
        lower_bound_99, upper_bound_99 = self.normal_approximation_bounds(group, 0.99)
        lower_bound_95, upper_bound_95 = self.normal_approximation_bounds(group, 0.95)
        g_avg = self.avg_group_proba(group)
        if g_avg >= upper_bound_99:
            rating = 'Green'
        elif g_avg <= upper_bound_95:
            rating = 'Red'
        else:
            rating = 'Yellow'
        return rating

#from logistic_regression import predict_proba, x_test, y_test, total_test_defaults
from probit_regression import predict_proba, x_test, y_test, total_test_defaults

y_test = np.reshape(y_test, (1161, 1))
data = np.concatenate((x_test, y_test), axis=1)
utils = Utils(data)
groups = utils.sorted_groups(predict_proba, n=10)
print('lengths', [len(group) for group in groups])
defaults_in_groups = [np.sum(group[i][0] for i in range(len(group))) for group in groups]
print('defaults in groups:', defaults_in_groups)
print('PDs in groups', [utils.avg_group_proba(group) for group in groups])

calibration_metrics = CalibrationMetrics(utils)
hs_statistics = calibration_metrics.hosmer_lemeshow(groups)
brier_score = calibration_metrics.brier_score(groups)
brier_skill_score = calibration_metrics.brier_skill_score(groups, total_test_defaults)

print('H-S statistics for groups', hs_statistics)
print('H-S statistic in total:', np.sum(hs_statistics))
print('Brier score: ', brier_score)
print('Brier skill score: ', brier_skill_score)

from scipy.stats import chisquare

p = 1 - chisquare(hs_statistics, 8)[1]
p = "{:.50f}".format(float(p))
print('p value for H-S:', p)
