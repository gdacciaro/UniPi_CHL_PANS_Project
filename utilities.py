import csv

import numpy as np
from sklearn.model_selection import KFold

import Constants
from models.NaiveBayes import NaiveBayes
from models.DecisionTree import DecisionTree
from models.KNN import KNN
from models.LRegression import LRegression
from models.MLP import MLP
from models.RandomForest import RandomForest
from models.SVM import SVM


def clean_empty_data(X,y):
    assert len(X) == len(y), str(len(X)) +"!="+ str(len(y))
    clean_X = list()
    clean_y = list()

    for i, feature in enumerate(X):
        if len(feature) > 0:
            clean_X.append(feature)
            clean_y.append(y[i])

    return clean_X,clean_y

def write_csv(feature_dev, feature_test, target_dev,target_test,sample_ids_dev, sample_ids_test, fe_method, scaler_method, tag = "meta", index=None):
    if index is None:
        index_part = ""
    else:
        index_part = "_"+str(index)
    name_file = Constants.ROOT + '/data/' +Constants.folder+"/"+ "poste_" + tag + "_" + fe_method + "_" + scaler_method + index_part + ".csv"
    with open(name_file, mode='w') as the_csv:
        csv_writer = csv.writer(the_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        for id, f_dev, t_dev in zip(sample_ids_dev, feature_dev, target_dev):
            f_dev = list(f_dev)
            csv_writer.writerow([id, *f_dev, t_dev])

        for id, f_test, t_test in zip(sample_ids_test, feature_test, target_test):
            f_test = list(f_test)
            csv_writer.writerow([id, *f_test, t_test])


def write_csv_validation(feature_train, feature_val, feature_test, target_train, target_val, target_test,
                         sample_ids_train, sample_ids_val, sample_ids_test, fe_method, scaler_method, tag="meta",
                         column_names=[], index=None):
    if index is None:
        index_part = ""
    else:
        index_part = "_" + str(index)
    name_file = Constants.ROOT + '/data/' + Constants.folder + "/"+ Constants.folder+"_" + tag + "_" + fe_method + "_" + scaler_method + index_part + ".csv"
    with open(name_file, mode='w') as the_csv:
        csv_writer = csv.writer(the_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        c_names = list(column_names)
        if len(c_names)>0:
            csv_writer.writerow(["Sample Id", *c_names, "Target"])

        for id, f_train, t_train in zip(sample_ids_train, feature_train, target_train):
            f_train = list(f_train)
            csv_writer.writerow([id, *f_train, t_train])

        for id, f_val, t_val in zip(sample_ids_val, feature_val, target_val):
            f_val = list(f_val)
            csv_writer.writerow([id, *f_val, t_val])

        for id, f_test, t_test in zip(sample_ids_test, feature_test, target_test):
            f_test = list(f_test)
            csv_writer.writerow([id, *f_test, t_test])

def get_all_models():
    all_SVM = SVM.get_all_combinations()
    all_MLP = MLP.get_all_combinations()
    all_RF = RandomForest.get_all_combinations()
    all_KNN = KNN.get_all_combinations()
    all_BNs = NaiveBayes.get_all_combinations()
    all_LReg = LRegression.get_all_combinations()
    all_Ctree = DecisionTree.get_all_combinations()

    all_models = [*all_BNs, *all_LReg, *all_Ctree, *all_SVM, *all_KNN, *all_RF, *all_MLP]
    return all_models

def eliminate_features(X, feature_selected_mask):
    X_copy = np.copy(X)

    result = list()
    for row in X_copy:
        transformed_row = list()
        for i, column in enumerate(row):
            if feature_selected_mask[i]:
                transformed_row.append(column)
        result.append(transformed_row)
    return result

def dataframe_to_list(df):
    res = list()
    for i in range(df.shape[0]):
        row = df.loc[[i]]
        res.append(row.values.ravel())

    return res

def k_fold(features_dev, target_dev, k):
    assert len(features_dev) == len(target_dev), str(len(features_dev)) +"!="+ str(len(target_dev))

    kf = KFold(n_splits=k)

    results = list()

    for train_index, validation_index in kf.split(features_dev):
        all_features_train = list()
        all_targets_train  = list()

        for item in train_index:
            single_features_train = features_dev[item]
            single_target_train   = target_dev[item]
            all_features_train.append(single_features_train)
            all_targets_train.append(single_target_train)

        all_features_validation = list()
        all_targets_validation = list()

        for item in validation_index:
            single_features_validation = features_dev[item]
            single_target_validation = target_dev[item]
            all_features_validation.append(single_features_validation)
            all_targets_validation.append(single_target_validation)

        results.append({
            "train":all_features_train,
            "target_train":all_targets_train,
            "validation":all_features_validation,
            "target_validation":all_targets_validation
        })

    return results