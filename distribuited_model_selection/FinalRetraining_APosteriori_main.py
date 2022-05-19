import platform
import threading

import numpy as np

import Constants
from dataset_managers.DatasetManager import DatasetManager
from distribuited_model_selection.DatabaseManager import getBestModel

#TelegramAPI.writeAMessage("36985739","change")

if __name__ == '__main__':
    Constants.who =  str(platform.node())+"_"+ str(threading.get_ident())
    np.random.seed(42)

    dataset_manager = DatasetManager()
    table_to_try = [
        ("a_posteriori_meta_isomap_min_max_scaling_0", "a_posteriori_meta_isomap_min_max_scaling_0.csv"),
        ("a_posteriori_meta_isomap_min_max_scaling_1", "a_posteriori_meta_isomap_min_max_scaling_1.csv"),
        ("a_posteriori_meta_isomap_min_max_scaling_2", "a_posteriori_meta_isomap_min_max_scaling_2.csv"),
        ("a_posteriori_meta_isomap_standard_scaling_0", "a_posteriori_meta_isomap_standard_scaling_0.csv"),
        ("a_posteriori_meta_isomap_standard_scaling_1", "a_posteriori_meta_isomap_standard_scaling_1.csv"),
        ("a_posteriori_meta_isomap_standard_scaling_2", "a_posteriori_meta_isomap_standard_scaling_2.csv"),
        ("a_posteriori_meta_isomap_unscaled_0", "a_posteriori_meta_isomap_unscaled_0.csv"),
        ("a_posteriori_meta_isomap_unscaled_1", "a_posteriori_meta_isomap_unscaled_1.csv"),
        ("a_posteriori_meta_isomap_unscaled_2", "a_posteriori_meta_isomap_unscaled_2.csv"),
        ("a_posteriori_meta_isomap_yeo_johnson_transformation_0",
         "a_posteriori_meta_isomap_yeo_johnson_transformation_0.csv"),
        ("a_posteriori_meta_isomap_yeo_johnson_transformation_1",
         "a_posteriori_meta_isomap_yeo_johnson_transformation_1.csv"),
        ("a_posteriori_meta_isomap_yeo_johnson_transformation_2",
         "a_posteriori_meta_isomap_yeo_johnson_transformation_2.csv"),
        ("a_posteriori_meta_pca_min_max_scaling_0", "a_posteriori_meta_pca_min_max_scaling_0.csv"),
        ("a_posteriori_meta_pca_min_max_scaling_1", "a_posteriori_meta_pca_min_max_scaling_1.csv"),
        ("a_posteriori_meta_pca_min_max_scaling_2", "a_posteriori_meta_pca_min_max_scaling_2.csv"),
        ("a_posteriori_meta_pca_standard_scaling_0", "a_posteriori_meta_pca_standard_scaling_0.csv"),
        ("a_posteriori_meta_pca_standard_scaling_1", "a_posteriori_meta_pca_standard_scaling_1.csv"),
        ("a_posteriori_meta_pca_standard_scaling_2", "a_posteriori_meta_pca_standard_scaling_2.csv"),
        ("a_posteriori_meta_pca_unscaled_0", "a_posteriori_meta_pca_unscaled_0.csv"),
        ("a_posteriori_meta_pca_unscaled_1", "a_posteriori_meta_pca_unscaled_1.csv"),
        ("a_posteriori_meta_pca_unscaled_2", "a_posteriori_meta_pca_unscaled_2.csv"),
        (
        "a_posteriori_meta_pca_yeo_johnson_transformation_0", "a_posteriori_meta_pca_yeo_johnson_transformation_0.csv"),
        (
        "a_posteriori_meta_pca_yeo_johnson_transformation_1", "a_posteriori_meta_pca_yeo_johnson_transformation_1.csv"),
        (
        "a_posteriori_meta_pca_yeo_johnson_transformation_2", "a_posteriori_meta_pca_yeo_johnson_transformation_2.csv"),
        ("a_posteriori_meta_rfe_min_max_scaling_0", "a_posteriori_meta_rfe_min_max_scaling_0.csv"),
        ("a_posteriori_meta_rfe_min_max_scaling_1", "a_posteriori_meta_rfe_min_max_scaling_1.csv"),
        ("a_posteriori_meta_rfe_min_max_scaling_2", "a_posteriori_meta_rfe_min_max_scaling_2.csv"),
        ("a_posteriori_meta_rfe_standard_scaling_0", "a_posteriori_meta_rfe_standard_scaling_0.csv"),
        ("a_posteriori_meta_rfe_standard_scaling_1", "a_posteriori_meta_rfe_standard_scaling_1.csv"),
        ("a_posteriori_meta_rfe_standard_scaling_2", "a_posteriori_meta_rfe_standard_scaling_2.csv"),
        ("a_posteriori_meta_rfe_unscaled_0", "a_posteriori_meta_rfe_unscaled_0.csv"),
        ("a_posteriori_meta_rfe_unscaled_1", "a_posteriori_meta_rfe_unscaled_1.csv"),
        ("a_posteriori_meta_rfe_unscaled_2", "a_posteriori_meta_rfe_unscaled_2.csv"),
        (
        "a_posteriori_meta_rfe_yeo_johnson_transformation_0", "a_posteriori_meta_rfe_yeo_johnson_transformation_0.csv"),
        (
        "a_posteriori_meta_rfe_yeo_johnson_transformation_1", "a_posteriori_meta_rfe_yeo_johnson_transformation_1.csv"),
        (
        "a_posteriori_meta_rfe_yeo_johnson_transformation_2", "a_posteriori_meta_rfe_yeo_johnson_transformation_2.csv"),
        ("a_posteriori_prote_isomap_min_max_scaling_0", "a_posteriori_prote_isomap_min_max_scaling_0.csv"),
        ("a_posteriori_prote_isomap_min_max_scaling_1", "a_posteriori_prote_isomap_min_max_scaling_1.csv"),
        ("a_posteriori_prote_isomap_min_max_scaling_2", "a_posteriori_prote_isomap_min_max_scaling_2.csv"),
        ("a_posteriori_prote_isomap_standard_scaling_0", "a_posteriori_prote_isomap_standard_scaling_0.csv"),
        ("a_posteriori_prote_isomap_standard_scaling_1", "a_posteriori_prote_isomap_standard_scaling_1.csv"),
        ("a_posteriori_prote_isomap_standard_scaling_2", "a_posteriori_prote_isomap_standard_scaling_2.csv"),
        ("a_posteriori_prote_isomap_unscaled_0", "a_posteriori_prote_isomap_unscaled_0.csv"),
        ("a_posteriori_prote_isomap_unscaled_1", "a_posteriori_prote_isomap_unscaled_1.csv"),
        ("a_posteriori_prote_isomap_unscaled_2", "a_posteriori_prote_isomap_unscaled_2.csv"),
        ("a_posteriori_prote_isomap_yeo_johnson_transformation_0",
         "a_posteriori_prote_isomap_yeo_johnson_transformation_0.csv"),
        ("a_posteriori_prote_isomap_yeo_johnson_transformation_1",
         "a_posteriori_prote_isomap_yeo_johnson_transformation_1.csv"),
        ("a_posteriori_prote_isomap_yeo_johnson_transformation_2",
         "a_posteriori_prote_isomap_yeo_johnson_transformation_2.csv"),
        ("a_posteriori_prote_pca_min_max_scaling_0", "a_posteriori_prote_pca_min_max_scaling_0.csv"),
        ("a_posteriori_prote_pca_min_max_scaling_1", "a_posteriori_prote_pca_min_max_scaling_1.csv"),
        ("a_posteriori_prote_pca_min_max_scaling_2", "a_posteriori_prote_pca_min_max_scaling_2.csv"),
        ("a_posteriori_prote_pca_standard_scaling_0", "a_posteriori_prote_pca_standard_scaling_0.csv"),
        ("a_posteriori_prote_pca_standard_scaling_1", "a_posteriori_prote_pca_standard_scaling_1.csv"),
        ("a_posteriori_prote_pca_standard_scaling_2", "a_posteriori_prote_pca_standard_scaling_2.csv"),
        ("a_posteriori_prote_pca_unscaled_0", "a_posteriori_prote_pca_unscaled_0.csv"),
        ("a_posteriori_prote_pca_unscaled_1", "a_posteriori_prote_pca_unscaled_1.csv"),
        ("a_posteriori_prote_pca_unscaled_2", "a_posteriori_prote_pca_unscaled_2.csv"),
        ("a_posteriori_prote_pca_yeo_johnson_transformation_0",
         "a_posteriori_prote_pca_yeo_johnson_transformation_0.csv"),
        ("a_posteriori_prote_pca_yeo_johnson_transformation_1",
         "a_posteriori_prote_pca_yeo_johnson_transformation_1.csv"),
        ("a_posteriori_prote_pca_yeo_johnson_transformation_2",
         "a_posteriori_prote_pca_yeo_johnson_transformation_2.csv"),
        ("a_posteriori_prote_rfe_min_max_scaling_0", "a_posteriori_prote_rfe_min_max_scaling_0.csv"),
        ("a_posteriori_prote_rfe_min_max_scaling_1", "a_posteriori_prote_rfe_min_max_scaling_1.csv"),
        ("a_posteriori_prote_rfe_min_max_scaling_2", "a_posteriori_prote_rfe_min_max_scaling_2.csv"),
        ("a_posteriori_prote_rfe_standard_scaling_0", "a_posteriori_prote_rfe_standard_scaling_0.csv"),
        ("a_posteriori_prote_rfe_standard_scaling_1", "a_posteriori_prote_rfe_standard_scaling_1.csv"),
        ("a_posteriori_prote_rfe_standard_scaling_2", "a_posteriori_prote_rfe_standard_scaling_2.csv"),
        ("a_posteriori_prote_rfe_unscaled_0", "a_posteriori_prote_rfe_unscaled_0.csv"),
        ("a_posteriori_prote_rfe_unscaled_1", "a_posteriori_prote_rfe_unscaled_1.csv"),
        ("a_posteriori_prote_rfe_unscaled_2", "a_posteriori_prote_rfe_unscaled_2.csv"),
        ("a_posteriori_prote_rfe_yeo_johnson_transformation_0",
         "a_posteriori_prote_rfe_yeo_johnson_transformation_0.csv"),
        ("a_posteriori_prote_rfe_yeo_johnson_transformation_1",
         "a_posteriori_prote_rfe_yeo_johnson_transformation_1.csv"),
        ("a_posteriori_prote_rfe_yeo_johnson_transformation_2",
         "a_posteriori_prote_rfe_yeo_johnson_transformation_2.csv"),
    ]


    import csv
    i = 0
    name_file = Constants.ROOT + '/data/' + "results_a_posteriori.csv"
    with open(name_file, mode='w') as the_csv:
        csv_writer = csv.writer(the_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        csv_writer.writerow(["i", "type", "db", "fe", "scaler", "result", "acc", "f1", str("model"), "file", "index"])
        for table_name, file_name in table_to_try:
            Constants.MySQL_name_table = table_name
            id, model,result = getBestModel()

            pieces = table_name.split("_")
            type = "a_posteriori"
            db = pieces[2]
            fe = pieces[3]
            scaler = table_name
            scaler = scaler[:len(scaler) - 2]
            scaler = scaler.replace("a_posteriori_","").replace("meta_","").replace("prote_","") \
                                  .replace("pca_", "").replace("rfe_", "").replace("isomap_", "") \

            print("==============")
            print(">>>table_name ", table_name)
            print(">>>model [",id,"]",model)
            print(">>>file_name ", file_name)
            print(">>>type ", type, " ", db," ",fe," ",scaler)

            print("=== Results ===")

            X_train, y_train, X_test, y_test, _, _ = dataset_manager.read_dataset_with_feature_selection(file_name)
            metrics = model.fit_and_predict(X_train, y_train, X_test, y_test)
            print(">>>\t",metrics)
            acc = metrics["accuracy"]
            f1 = metrics["f1"]

            index = pieces[-1]

            csv_writer.writerow([i, type,db,fe,scaler, result, acc,f1,str(model),file_name,index])

            print("==============")
            print("\n\n")
            i+=1
