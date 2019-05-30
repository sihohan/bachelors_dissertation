from heapq import nlargest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.svm import LinearSVC
from time import time
from xgboost import XGBClassifier

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as mt

import manual_feature_extraction as mfe
import thresholding as th
import time2str as t2s

# Choose the data type, feature type, number of times to repeat training, and k
model = input("Linear SVM or XGBoost (svm / xgb)?: ")
data_type = input("Input data type (donors / acceptors): ")
feature_type = input("Input feature type (positional / compositional / potential): ")
n_repeat = int(input("Repeat training for the following number of times (integer): "))
k = int(input("K-mer considered is (integer): "))
print()

# Define local context sizes depending on the data and feature type
if feature_type == 'positional':
    if data_type == 'donors':
        if k == 1:
            context_list = [(60, 120), (100, 100), (50, 50), (20, 20)]
        if k == 2:
            context_list = [(40, 120), (100, 100), (50, 50), (100, 20)]
        if k == 3:
            context_list = [(40, 80), (100, 100), (50, 50), (80, 40)]
    if data_type == 'acceptors':
        if k == 1:
            context_list = [(80, 80), (100, 100), (50, 50), (20, 100)]
        if k == 2:
            context_list = [(80, 60), (100, 100), (50, 50), (20, 80)]
        if k == 3:
            context_list = [(80, 60), (100, 100), (50, 50), (20, 100)]

if feature_type == 'compositional':
    if data_type == 'donors':
        if k == 3:
            context_list = [(20, 80), (100, 100), (50, 50), (100, 20)]
        if k == 4:
            context_list = [(80, 80), (100, 100), (50, 50), (60, 20)]
        if k == 5:
            context_list = [(80, 80), (100, 100), (50, 50), (20, 40)]
        if k == 6:
            context_list = [(80, 80), (100, 100), (50, 50), (20, 20)]
    if data_type == 'acceptors':
        if k == 3:
            context_list = [(20, 60), (100, 100), (50, 50), (100, 20)]
        if k == 4:
            context_list = [(20, 80), (100, 100), (50, 50), (100, 40)]
        if k == 5:
            context_list = [(40, 60), (100, 100), (50, 50), (100, 20)]
        if k == 6:
            context_list = [(80, 80), (100, 100), (50, 50), (100, 20)]

if feature_type == 'potential':
    if data_type == 'donors':
        context_list = [(20, 60), (100, 100), (50, 50), (100, 20)]
    if data_type == 'acceptors':
        context_list = [(20, 80), (100, 100), (50, 50), (100, 20)]

# Define the feature name as Pk for positional features or Ck for compositional features
if feature_type in ['positional', 'compositional']:
    feature_name = '{}'.format(feature_type[0].upper())+'{}'.format(k)

# Define the feature name as RF for coding potential features
if feature_type == 'potential':
    feature_name = 'RF'

# If dealing with linear SVM
if model == 'svm':

    # Preallocate numpy arrays filled with 'None'
    precision_final = np.full(len(context_list), fill_value=None)
    recall_final = np.full(len(context_list), fill_value=None)
    f1_final = np.full(len(context_list), fill_value=None)
    Pr_95_final = np.full(len(context_list), fill_value=None)
    gs_time_final = np.full(len(context_list), fill_value=None)
    threshold_95_final = np.full(len(context_list), fill_value=None)

    # Initialise a counter
    cnt = 0

    for p, q in context_list:

        print('LOCAL CONTEXT SIZE: ({}, {})'.format(p, q))

        # Extract feature sets
        X, y = mfe.train_test_extraction(data_type, p, q, feature_type, k, subsample=True)

        # Split the feature sets into train / test sets in a 2:1 ratio
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=42)

        # Initialise a linear SVM classifier
        svclassifier = LinearSVC(verbose=0, dual=False)

        # Tune the cost parameter C through grid search and a 10-fold cross validation
        # n_jobs: the number of jobs to run in parallel
        # pre_dispatch: controls the number of jobs that get dispatched during parallel execution;
        # memory consumption can be limited by reducing this number
        # scoring: the metric used to evaluate the predictions on the test test
        # refit: the metric used to refit the estimator, using the optimal set of parameters
        svclassifier = GridSearchCV(estimator=svclassifier, param_grid={'C': [2**k for k in range(-12, 4+1)]},
                                    cv=10, n_jobs=-1, pre_dispatch='2*n_jobs', scoring='f1', refit='f1')

        # Calibrate the classifier, because LinearSVC does not support probability prediction
        clf = CalibratedClassifierCV(svclassifier, cv=10)

        # Start recording the training time
        t1_gs_time = time()

        print('Start grid search and model training...')
        # Train the model with the optimal set of parameters
        # eval_metric: evaluation metric for the validation data (auc: area under the curve)
        # eval_set: evaluation sets
        # early_stopping_rounds: number after which the training stops if the validation score stops to improve
        clf.fit(X_train, y_train)
        print('Model training completed!')
        print()

        # Record the training time
        gs_time = time() - t1_gs_time

        # Convert recorded times (s) into strings of format h:mm:ss
        gs_time = t2s.time2str(gs_time)

        print("Classifying splice sites...")
        # Obtain prediction probabilities generated by the binary logistic regression algorithm
        y_prob = clf.predict_proba(X_test)

        # Classify a candidate splice site into a true or pseudo splice site based on a threshold value of 0.5
        y_pred = [np.argmax(pred_prob) for pred_prob in y_prob]

        # Precision, recall and F1 score at threshold = 0.5
        precision = mt.precision_score(y_test, y_pred)
        recall = mt.recall_score(y_test, y_pred)
        f1 = mt.f1_score(y_test, y_pred)

        # Search the threshold at which recall = 0.95
        threshold_95 = th.threshold95_searcher(y_test=y_test, prediction_matrix=y_prob)
        TN, FP, FN, TP = th.metric_components(y_test=y_test, threshold=threshold_95, prediction_matrix=y_prob)

        # Calculate the precision at recall = 0.96 (Pr.95)
        Pr_95 = TP/(TP+FP)

        # Store the final values
        precision_final[cnt] = precision
        recall_final[cnt] = recall
        f1_final[cnt] = f1
        Pr_95_final[cnt] = Pr_95
        gs_time_final[cnt] = gs_time
        threshold_95_final[cnt] = threshold_95

        # Proceed to next set of context sizes (p, q)
        cnt += 1

    # Frame a table for the display of results
    print(' {:^3} | {:^3} | {:^4} | {:^4} | {:^4} | {:^5} | {:^7} | {:^5} '
          .format('p', 'q', 'Pr', 'Re', 'f1', 'Pr.95', 't', 'th.95'))
    print('-{:-^4}|{:-^5}|{:-^6}|{:-^6}|{:-^6}|{:-^7}|{:-^9}|{:-^7}'
          .format('', '', '', '', '', '', '', ''))

    # For each set of local contexts (p, q)
    for i in range(len(context_list)):

        # Take p and q to print results
        p = context_list[i][0]
        q = context_list[i][1]

        # Print results in the defined table frame
        print(' {:^3} | {:^3} | {:0.2f} | {:0.2f} | {:0.2f} | {:0.2f}  | {:^7} | {:0.3f} '
              .format(p, q, precision_final[i], recall_final[i], f1_final[i], Pr_95_final[i], gs_time_final[i],
                      threshold_95_final[i]))

# If dealing with XGBoost
if model == 'xgb':

    # Preallocate numpy arrays filled with 'None'
    precision_arr = np.full(n_repeat, fill_value=None)
    recall_arr = np.full(n_repeat, fill_value=None)
    f1_arr = np.full(n_repeat, fill_value=None)
    Pr_95_arr = np.full(n_repeat, fill_value=None)
    rs_time_arr = np.full(n_repeat, fill_value=None)
    threshold_95_arr = np.full(n_repeat, fill_value=None)

    precision_final = np.full(len(context_list), fill_value=None)
    recall_final = np.full(len(context_list), fill_value=None)
    f1_final = np.full(len(context_list), fill_value=None)
    Pr_95_final = np.full(len(context_list), fill_value=None)
    rs_time_final = np.full(len(context_list), fill_value=None)
    threshold_95_final = np.full(len(context_list), fill_value=None)

    stdev_precision_final = np.full(len(context_list), fill_value=None)
    stdev_recall_final = np.full(len(context_list), fill_value=None)
    stdev_f1_final = np.full(len(context_list), fill_value=None)
    stdev_Pr_95_final = np.full(len(context_list), fill_value=None)
    stdev_rs_time_final = np.full(len(context_list), fill_value=None)
    stdev_threshold_95_final = np.full(len(context_list), fill_value=None)

    # Define a set of hyperparameters
    fixed_params = {
                'objective': 'binary:logistic',
                'n_estimators': 9999,
                'tree_method': 'gpu_hist',
                'scale_pos_weight': 10,
    }

    # Initialise an XGBoost classifier
    clf_0 = XGBClassifier(**fixed_params)

    # Initialise a counter
    cnt = 0

    # For all sets of context sizes (p, q)
    for p, q in context_list:

        print('LOCAL CONTEXT SIZE: ({}, {})'.format(p, q))

        # For n_repeat times
        for i in range(n_repeat):

            # Print trial number if training is repeated more than one time
            if n_repeat > 1:
                print('TRIAL #{}'.format(i+1))
                print()

            print('Start feature extraction...')
            # Extract feature sets
            X, y = mfe.train_test_extraction(data_type, p, q, feature_type, k, subsample=True)
            print('Feature extraction completed!')
            print()

            # Split the feature sets into train / validation / test sets in a 3:1:1 ratio
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

            # Define a set of parameters that you want to tune
            parameters = {
                'max_depth': [3, 4, 5, 6, 7, 8, 9], 'min_child_weight': [3, 4, 5, 6, 7, 8, 9],
                'gamma': [0, 0.1, 0.3], 'subsample': [0.7, 0.8, 0.9], 'colsample_bytree': [0.7, 0.8, 0.9],
                'eta': [0.01, 0.03, 0.1, 0.2, 0.3]
            }

            # Tune the parameters through randomise search and a 10-fold cross validation
            # n_jobs: the number of jobs to run in parallel
            # pre_dispatch: controls the number of jobs that get dispatched during parallel execution;
            # memory consumption can be limited by reducing this number
            # scoring: the metric used to evaluate the predictions on the test test
            # refit: the metric used to refit the estimator, using the optimal set of parameters
            clf = RandomizedSearchCV(estimator=clf_0, param_distributions=parameters,
                                     n_jobs=-1, pre_dispatch='2*n_jobs', cv=10,
                                     scoring='f1', refit='f1', error_score='raise')

            # Start recording the training time
            t1_rs_time = time()

            print('Start randomised search and model training...')
            # Train the model with the optimal set of parameters
            # eval_metric: evaluation metric for the validation data (auc: area under the curve)
            # eval_set: evaluation sets
            # early_stopping_rounds: number after which the training stops if the validation score stops to improve
            clf.fit(X=X_train, y=y_train,
                    eval_metric='auc', eval_set=[(X_val, y_val)],
                    early_stopping_rounds=50, verbose=False)
            print('Model training completed!')
            print()

            # Record the training time
            rs_time = time()-t1_rs_time

            # Feature important values are obtained as an array
            feature_importance_list = clf.best_estimator_.feature_importances_

            # Plot the feature importance values for each feature index and save the graph in a .png format
            plt.bar(range(len(feature_importance_list)), feature_importance_list)
            plt.xlabel("Feature Index")
            plt.ylabel("Feature Importance")
            plt.show()
            plt.savefig('{}_{}_{}_{}_feature_importance.png'.format(data_type, feature_name, p, q))
            print('Saved feature importance graph!')
            print()

            # Analyse feature importance values of positional feature sets for nucleotide pattern detection,
            if data_type == 'positional':

                # We will take the top 5 feature importance values for donors
                if data_type == "donors":
                    n = 5
                # We will take the top 10 feature importance values for acceptors
                if data_type == "acceptors":
                    n = 10

                # Store feature index and feature importance in a dictionary
                # in the form {'index': importance} for the top feature importance values
                top_feature_importance = {index: importance for index, importance in enumerate(feature_importance_list)
                                          if importance in nlargest(n, feature_importance_list)}

                # Preallocate a numpy array filled with 'None'
                nucleotide_index_list = np.full(n, fill_value=None)

                # Convert feature indices to nucleotide indices and store in an array
                for i, top_feature_index in enumerate(top_feature_importance):

                    feature_index = top_feature_index
                    nucleotide_index = feature_index//(4**k)
                    nucleotide_index_list[i] = nucleotide_index

                # Initialise a dictionary
                nucleotide_frequency_pos = {}

                # Load the previously saved positive sample sequence list
                pos_seq_list = np.load('{}_pos_seq_list_feat_imp.npy'.format(data_type))

                # For each position in the nucleotide index array
                for nucleotide_index in nucleotide_index_list:

                    # Initialise a nested dictionary
                    nucleotide_frequency_pos[nucleotide_index] = {}

                    # For all positive sample sequences, find the most frequently occurring
                    # nucleotides with the highest feature importance values and store in the nested dictionary
                    # in the form {nucleotide index:{nucleotide: occurrence}}
                    for pos_seq in pos_seq_list:

                        nucleotide = pos_seq[nucleotide_index]
                        try:
                            nucleotide_frequency_pos[nucleotide_index][nucleotide] += 1
                        except:
                            nucleotide_frequency_pos[nucleotide_index][nucleotide] = 1

            print("Classifying splice sites...")
            # Obtain prediction probabiliries generated by the binary logistic regression algorithm
            y_prob = clf.predict_proba(X_test)

            # Classify a candidate splice site into a true or pseudo splice site based on a threshold value of 0.5
            y_pred = [np.argmax(pred_prob) for pred_prob in y_prob]

            # Precision, recall and F1 score at threshold = 0.5
            precision = mt.precision_score(y_test, y_pred)
            recall = mt.recall_score(y_test, y_pred)
            f1 = mt.f1_score(y_test, y_pred)

            # Search the threshold at which recall = 0.95
            threshold_95 = th.threshold95_searcher(y_test=y_test, prediction_matrix=y_prob)
            TN, FP, FN, TP = th.metric_components(y_test=y_test, threshold=threshold_95, prediction_matrix=y_prob)

            # Calculate the precision at recall = 0.96 (Pr.95)
            Pr_95 = TP/(TP+FP)

            # Store all evaluation metrics in arrays
            precision_arr[i] = precision
            recall_arr[i] = recall
            f1_arr[i] = f1
            Pr_95_arr[i] = Pr_95
            rs_time_arr[i] = rs_time
            threshold_95_arr[i] = threshold_95

        # Calculate means
        precision = np.mean(precision_arr)
        recall = np.mean(recall_arr)
        f1 = np.mean(f1_arr)
        Pr_95 = np.mean(Pr_95_arr)
        rs_time = np.mean(rs_time_arr)
        threshold_95 = np.mean(threshold_95_arr)

        # Calculate standard deviations
        stdev_precision = np.std(precision_arr)
        stdev_recall = np.std(recall_arr)
        stdev_f1 = np.std(f1_arr)
        stdev_Pr_95 = np.std(Pr_95_arr)
        stdev_rs_time = np.std(rs_time_arr)
        stdev_threshold_95 = np.std(threshold_95_arr)

        # Convert recorded times (s) into strings of format h:mm:ss
        rs_time = t2s.time2str(rs_time)
        stdev_rs_time = t2s.time2str(stdev_rs_time)

        # Store the final values
        precision_final[cnt] = precision
        recall_final[cnt] = recall
        f1_final[cnt] = f1
        Pr_95_final[cnt] = Pr_95
        rs_time_final[cnt] = rs_time
        threshold_95_final[cnt] = threshold_95

        stdev_precision_final[cnt] = stdev_precision
        stdev_recall_final[cnt] = stdev_recall
        stdev_f1_final[cnt] = stdev_f1
        stdev_Pr_95_final[cnt] = stdev_Pr_95
        stdev_rs_time_final[cnt] = stdev_rs_time
        stdev_threshold_95_final[cnt] = stdev_threshold_95

        # Proceed to next set of context sizes (p, q)
        cnt += 1

    # Frame a table for the display of results
    print(' {:^3} | {:^3} | {:^11} | {:^11} | {:^11} | {:^11} | {:^17} | {:^11} '
          .format('p', 'q', 'Precision', 'Recall', 'F1 Score', 'Pr.95', 'Training Time', 'Threshold.95'))
    print('-{:-^4}|{:-^5}|{:-^13}|{:-^13}|{:-^13}|{:-^13}|{:-^19}|{:-^14}'
          .format('', '', '', '', '', '', '', ''))

    # For each set of local contexts (p, q)
    for i in range(len(context_list)):

        # Take p and q to print results
        p = context_list[i][0]
        q = context_list[i][1]

        # Print results in the defined table frame
        print(' {:^3} | {:^3} | {:0.2f} ({:0.2f}) | {:0.2f} ({:0.2f}) | {:0.2f} ({:0.2f}) | {:0.2f} ({:0.2f}) | {:^5} ({:^5}) | {:0.3f} ({:0.3f}) '
              .format(p, q, precision_final[i], stdev_precision_final[i], recall_final[i], stdev_recall_final[i],
                      f1_final[i], stdev_f1_final[i], Pr_95_final[i], stdev_Pr_95_final[i],
                      rs_time_final[i], stdev_rs_time_final[i], threshold_95_final[i], stdev_threshold_95_final[i]))
