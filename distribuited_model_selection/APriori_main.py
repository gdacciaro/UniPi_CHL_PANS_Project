import platform
import threading

import numpy as np

import Constants
from dataset_managers.DatasetManager import DatasetManager
from distribuited_model_selection.DatabaseManager import get_trial_information
from distribuited_model_selection.DistribuitedModelSelection import execute_trials

#TelegramAPI.writeAMessage("36985739","change")

if __name__ == '__main__':
    Constants.who =  str(platform.node())+"_"+ str(threading.get_ident())
    Constants.folder = "a_priori"
    np.random.seed(42)
    dataset_manager = DatasetManager()

    table_to_try = [
        ("a_priori_isomap_min_max_scaling_0", "a_priori_isomap_min_max_scaling_0.csv"),
        ("a_priori_isomap_min_max_scaling_1", "a_priori_isomap_min_max_scaling_1.csv"),
        ("a_priori_isomap_min_max_scaling_2", "a_priori_isomap_min_max_scaling_2.csv"),
        ("a_priori_isomap_standard_scaling_0", "a_priori_isomap_standard_scaling_0.csv"),
        ("a_priori_isomap_standard_scaling_1", "a_priori_isomap_standard_scaling_1.csv"),
        ("a_priori_isomap_standard_scaling_2", "a_priori_isomap_standard_scaling_2.csv"),
        ("a_priori_isomap_unscaled_0", "a_priori_isomap_unscaled_0.csv"),
        ("a_priori_isomap_unscaled_1", "a_priori_isomap_unscaled_1.csv"),
        ("a_priori_isomap_unscaled_2", "a_priori_isomap_unscaled_2.csv"),
        ("a_priori_isomap_yeo_johnson_transformation_0", "a_priori_isomap_yeo_johnson_transformation_0.csv"),
        ("a_priori_isomap_yeo_johnson_transformation_1", "a_priori_isomap_yeo_johnson_transformation_1.csv"),
        ("a_priori_isomap_yeo_johnson_transformation_2", "a_priori_isomap_yeo_johnson_transformation_2.csv"),
        ("a_priori_pca_min_max_scaling_0", "a_priori_pca_min_max_scaling_0.csv"),
        ("a_priori_pca_min_max_scaling_1", "a_priori_pca_min_max_scaling_1.csv"),
        ("a_priori_pca_min_max_scaling_2", "a_priori_pca_min_max_scaling_2.csv"),
        ("a_priori_pca_standard_scaling_0", "a_priori_pca_standard_scaling_0.csv"),
        ("a_priori_pca_standard_scaling_1", "a_priori_pca_standard_scaling_1.csv"),
        ("a_priori_pca_standard_scaling_2", "a_priori_pca_standard_scaling_2.csv"),
        ("a_priori_pca_unscaled_0", "a_priori_pca_unscaled_0.csv"),
        ("a_priori_pca_unscaled_1", "a_priori_pca_unscaled_1.csv"),
        ("a_priori_pca_unscaled_2", "a_priori_pca_unscaled_2.csv"),
        ("a_priori_pca_yeo_johnson_transformation_0", "a_priori_pca_yeo_johnson_transformation_0.csv"),
        ("a_priori_pca_yeo_johnson_transformation_1", "a_priori_pca_yeo_johnson_transformation_1.csv"),
        ("a_priori_pca_yeo_johnson_transformation_2", "a_priori_pca_yeo_johnson_transformation_2.csv"),
        ("a_priori_rfe_min_max_scaling_0", "a_priori_rfe_min_max_scaling_0.csv"),
        ("a_priori_rfe_min_max_scaling_1", "a_priori_rfe_min_max_scaling_1.csv"),
        ("a_priori_rfe_min_max_scaling_2", "a_priori_rfe_min_max_scaling_2.csv"),
        ("a_priori_rfe_standard_scaling_0", "a_priori_rfe_standard_scaling_0.csv"),
        ("a_priori_rfe_standard_scaling_1", "a_priori_rfe_standard_scaling_1.csv"),
        ("a_priori_rfe_standard_scaling_2", "a_priori_rfe_standard_scaling_2.csv"),
        ("a_priori_rfe_unscaled_0", "a_priori_rfe_unscaled_0.csv"),
        ("a_priori_rfe_unscaled_1", "a_priori_rfe_unscaled_1.csv"),
        ("a_priori_rfe_unscaled_2", "a_priori_rfe_unscaled_2.csv"),
        ("a_priori_rfe_yeo_johnson_transformation_0", "a_priori_rfe_yeo_johnson_transformation_0.csv"),
        ("a_priori_rfe_yeo_johnson_transformation_1", "a_priori_rfe_yeo_johnson_transformation_1.csv"),
        ("a_priori_rfe_yeo_johnson_transformation_2", "a_priori_rfe_yeo_johnson_transformation_2.csv"),
    ]

    counter = 0
    size = len(table_to_try)

    while True:
        try:
            counter, table, file_name, count = get_trial_information(table_to_try, counter)
            print("Remaining models:", count, table)
            Constants.MySQL_name_table = table
            X_train, y_train, X_test, y_test, _, _ = dataset_manager.read_dataset_with_feature_selection(file_name)

            value = min(count+1, 250)
            for _ in range(1,value):
                try:
                    execute_trials(X_train, y_train, k=4)
                except Exception as e:
                    print("Exception:", e)
                    raise e

            if counter == size:
                break
        except Exception as e:
            print("Exception:", e)
