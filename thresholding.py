import numpy as np


def metric_components(y_test, threshold, prediction_matrix):
    # Initialisation
    TN = FP = FN = TP = 0

    # For each index in the prediction matrix
    for index in range(len(prediction_matrix)):

        # Classify splice sites based on the threshold at which recall = 0.95
        if y_test[index] == 0 and (prediction_matrix[index][1] < threshold):
            TN += 1
        if y_test[index] == 0 and (prediction_matrix[index][1] >= threshold):
            FP += 1
        if y_test[index] == 1 and (prediction_matrix[index][1] < threshold):
            FN += 1
        if y_test[index] == 1 and (prediction_matrix[index][1] >= threshold):
            TP += 1

    return TN, FP, FN, TP


def threshold95_searcher(y_test, prediction_matrix):

    '''
    :param prediction_matrix: y_prob
    :return: threshold for which recall >= 0.95
    '''

    for threshold in reversed(np.arange(0.01, 0.5, 0.001)):

        TN, FP, FN, TP = metric_components(y_test, float('{0:.2f}'.format(threshold)), prediction_matrix)
        recall = TP/(TP+FN)

        if recall >= 0.95:
            return float('{0:.2f}'.format(threshold))

    return '########## Try different threshold values!'
