from calibration_assessment_tools import *
from scipy.stats import chisquare


def generate_assessment(predict_proba, x_test, y_test, total_test_defaults):
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

    p = 1 - chisquare(hs_statistics, 8)[1]
    p = "{:.50f}".format(float(p))
    print('p value for H-S:', p)

    traffic_lights = [calibration_metrics.normal_approximation_test(group) for group in groups]
    print('traffic lights:', traffic_lights)
    pass


from logistic_regression import predict_proba, x_test, y_test, total_test_defaults
# from probit_regression import predict_proba, x_test, y_test, total_test_defaults

generate_assessment(predict_proba, x_test, y_test, total_test_defaults)
