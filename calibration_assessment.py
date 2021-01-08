# import pandas as pd
import numpy as np


class Utils:

    def __init__(self, data):
        self.data = data

    def divide_to_groups(self, predicted_probability, n=10):
        """
        :param predicted_probability: 1d array of PDs (nums between 0&1, 1=certain default)
        :param n: desired number of groups, defaults to 10
        :return: list od 2d arrays of PDs and default flagst
        """
        observations = self.data['DEFAULT_FLAG']
        observations = observations.to_numpy()
        num_of_obs = len(observations)
        obs_and_predictions = np.concatenate(observations, predicted_probability, axis=1)

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
        pds = group[:, 0]
        return np.sum(pds) / len(pds)

    def sorted_groups(self, predicted_probability, n=10):
        """
        :param predicted_probability: 1d array of predicted default probabilities (1=certain default)
        :param n: number of groups
        :return: n groups of pairs [PD, realized default flag] sorted by avg groups PD
        """
        groups = self.divide_to_groups(predicted_probability, n)
        avg_probs = [self.avg_group_proba(group) for group in groups]

        # rows in array are pairs of group,avg_prob, I sort them by avg probability
        groups = np.concatenate(groups, avg_probs, axis=1)
        sorted_groups = sorted(groups, key=lambda x: x[1], reverse=False)
        sorted_groups = sorted_groups[:, 0]

        return sorted_groups


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
            number_of_defaults = np.sum(group[:, 1])
            realized_default_rate = number_of_defaults / total_number
            predicted_default_rate = self.avg_group_proba(group)
            hs_stat = np.divide((realized_default_rate - predicted_default_rate) ** 2,
                                predicted_default_rate(1 - predicted_default_rate)) * total_number
            hs.append(hs_stat)
        return hs

    def brier_score(self, groups):
        total_borrowers = 0
        bs_group = []
        for group in groups:
            borrowers_in_group = np.shape(group)[0]
            total_borrowers += borrowers_in_group
            number_of_defaults = np.sum(group[:, 1])
            realized_default_rate = number_of_defaults / borrowers_in_group
            predicted_default_rate = self.avg_group_proba(group)
            summed = realized_default_rate * (1 - realized_default_rate) + (
                    predicted_default_rate - realized_default_rate) ** 2
            bs_group.append(borrowers_in_group * summed)

        bs_group = np.ndarray(bs_group)
        bs = (1 / total_borrowers) * np.sum(bs_group)
        return bs

    def brier_skill_score(self, groups):
        pd_observed = np.sum(self.data['DEFAULT_FLAG']) / len(self.data['DEFAULT_FLAG'])
        return 1 - np.divide(self.brier_score(groups), pd_observed * (1 - pd_observed))

    def binomial_calibration_test(self):
        pass

    def normal_approximation(self):
        pass
