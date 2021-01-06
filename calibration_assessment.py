import pandas as pd
import numpy as np

class CalibrationMetrics(data):
    def __init__(self):
        self.data = data

    def divide_to_groups(self,predicted_proba , n=10):
        """

        :param predicted_proba:
        :param n:
        :return: list od 2d arrays of default flag and predicted probability of default
        """
        observations=self.data['DEFAULT_FLAG']
        observations = observations.to_numpy()
        num_of_obs = len(observations)
        obs_and_preds = np.concatenate(observations, predicted_proba ,axis=1)

        group_size = num_of_obs//n
        groups=[]

        for i in range(n):
            if i <= n-2:
                groups.append(obs_and_preds[i*group_size:(i+1)*group_size,:])
            else:
                groups.append(obs_and_preds[(n-1)*group_size:,:])
        return groups

    def sort_groups(self, predicted_proba):


    #calibration accuracy
    def hosmer_lemeshow(self):
        pass

    def brier_score(self):

    def brier_skill_score(self):
        pass

    def binomial_calibration_test(self):
        pass

    def normal_approximation(self):
        pass



    pass
