import numpy as np
from scipy.stats import norm


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
        pds = group[:, 1]  # sum of probabilities of defaults
        return np.sum(pds) / len(pds)

    def sorted_groups(self, predicted_probability, n=10):
        """
        :param predicted_probability: 1d array of predicted default probabilities (1=certain default)
        :param n: number of groups
        :return: n groups of pairs in an array [realized default flag, PD](<-somehow in this order) sorted by avg groups PD in descending order
        """
        groups = self.divide_to_groups(predicted_probability, n)
        avg_probs = [self.avg_group_proba(group) for group in groups]

        groups = np.column_stack((groups, avg_probs))

        sorted_groups = groups[groups[:, 1].argsort()]
        sorted_groups = np.flip(sorted_groups, axis=0)

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

            bs_group.append(borrowers_in_group * summed)
        bs = (1 / total_borrowers) * np.sum(bs_group)
        return bs

    def brier_skill_score(self, groups, total_defaults, num_of_obs=1161):
        pd_observed = total_defaults / num_of_obs
        return 1 - np.divide(self.brier_score(groups), pd_observed * (1 - pd_observed))

    def normal_approximation_bounds(self, group, q):
        borrowers_in_group = np.shape(group)[0]
        g_avg = self.avg_group_proba(group)
        upper_bound = norm.ppf((q + 1) / 2) * np.sqrt(np.divide((g_avg * (1 - g_avg)), borrowers_in_group))
        lower_bound = g_avg - upper_bound
        return lower_bound, upper_bound

    def normal_approximation_test(self, group):
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

