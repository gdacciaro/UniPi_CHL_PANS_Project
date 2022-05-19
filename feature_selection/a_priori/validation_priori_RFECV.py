import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

import Constants
from dataset_managers.DatasetManagerFS import DatasetManager_FeatureSelection
from scaler_selector import all_scaler
from utilities import write_csv_validation

warnings.filterwarnings("ignore")

min_features_to_select = 100


def applyRFECV(estimator, X, y, k):
    clf = RFECV(estimator, step=1, min_features_to_select=min_features_to_select, cv=StratifiedKFold(k, shuffle=True))
    clf.fit(X, y)

    plt.figure(figsize=(10, 10))
    plt.plot(range(1, len(clf.grid_scores_) + 1), clf.grid_scores_)
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score")
    plt.show()

    return clf


def estimator_selection(X, y, k):
    estimators = []
    params_list = []
    estimators_selected = []

    estimators.append(SVC())
    estimators.append(LogisticRegression())

    params_list.append({'estimator__C': [0.1, 1, 10],
                        'estimator__kernel': ['linear'],
                        'estimator__class_weight': ['balanced']})
    params_list.append({'estimator__penalty': ['l1', 'l2'],
                        'estimator__C': [0.1, 1, 10],
                        'estimator__solver': ['liblinear']})

    for i in range(len(estimators)):
        selector = RFECV(estimators[i], step=1, min_features_to_select=min_features_to_select,
                         cv=StratifiedKFold(k, shuffle=True), scoring='accuracy')
        clf = GridSearchCV(selector, params_list[i], cv=3, verbose=2)
        clf.fit(X, y)
        print("================== Best estimator ==========================")
        print(clf.best_estimator_.estimator_)
        estimators_selected.append(clf.best_estimator_.estimator_)

    estimator_best = None
    max_accuracy = 0
    for estimator in estimators_selected:
        rfecv = RFECV(
            estimator=estimator,
            step=1,
            min_features_to_select=min_features_to_select,
            cv=StratifiedKFold(k, shuffle=True),
            scoring="accuracy"
        )
        rfecv.fit(X, y)

        print("Estimator: " + str(estimator))

        # Plot number of features VS. cross-validation scores
        plt.figure(figsize=(10, 10))
        plt.xlabel("Number of features selected")
        plt.ylabel("Cross validation score (accuracy)")
        plt.plot(
            range(1, len(rfecv.grid_scores_) + 1),
            rfecv.grid_scores_,
        )
        plt.show()
        if max_accuracy < np.amax(rfecv.grid_scores_):
            max_accuracy = np.amax(rfecv.grid_scores_)
            estimator_best = estimator

    return estimator_best


def feature_selection(estimator, X, y, k):
    rfe = applyRFECV(estimator, X, y, k)
    accuracy = np.amax(rfe.grid_scores_)
    indexes_selected = np.where(rfe.support_ == True)

    print("Optimal number of features: %d" % rfe.n_features_)

    return indexes_selected, accuracy


def eliminate_features(X, indexes_delete):
    X_copy = np.copy(X)

    indexes = np.ndarray.tolist(indexes_delete[0])
    new_X = list()
    for row in X_copy:
        transformed_row = list()
        for i in range(len(indexes)):
            transformed_row.append(row[indexes[i]])
        new_X.append(transformed_row)

    return new_X


def RFECV_feature_selection2(training_data, validation_data, test_data, scaler_title, index):
    data_apriori = pd.read_csv(Constants.ROOT + '/data/a_priori/a_priori_dataset.csv', sep=',')
    data_apriori.pop("Patient id")
    data_apriori.pop("TIME POINT")

    id_train, X_train, y_train = training_data
    id_val, X_val, y_val = validation_data
    id_test, X_test, y_test = test_data

    estimator = estimator_selection(X_train, y_train, 4)
    indexes, accuracy = feature_selection(estimator, X_train, y_train, k=4)
    print("[A_PRIORI]", "ACCURACY:", accuracy)
    print("[", scaler_title, "]", indexes)
    X_train_transformed = eliminate_features(X_train, indexes)
    X_val_transformed = eliminate_features(X_val, indexes)
    X_test_transformed = eliminate_features(X_test, indexes)

    cols_name = []
    priori_features = data_apriori.iloc[:, indexes[0]]
    cols_name = priori_features.columns

    write_csv_validation(X_train_transformed, X_val_transformed, X_test_transformed, y_train, y_val, y_test, id_train,
                         id_val, id_test, "rfe", scaler_title, tag="apriori", column_names=cols_name, index=index)



if __name__ == '__main__':
    Constants.folder = "a_priori"
    db = DatasetManager_FeatureSelection()
    np.random.seed(42)

    types = [Constants.folder]
    for type in types:
        for index, ((id_train,X_train, y_train), (id_val,X_val, y_val), (id_test,X_test, y_test)) in enumerate(db.get_data2(type)):
            all_possible_scaled_X_train, all_possible_scaled_X_val, all_possible_scaled_X_test = all_scaler(X_train, X_val, X_test)

            for i in range(len(all_possible_scaled_X_train)):
                scaler_title, train_set = all_possible_scaled_X_train[i]
                _, val_set = all_possible_scaled_X_val[i]
                _, test_set = all_possible_scaled_X_test[i]
                RFECV_feature_selection2((id_train, train_set, y_train), (id_val, val_set, y_val),
                                         (id_test, test_set, y_test), scaler_title, index)