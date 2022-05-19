import math
import warnings

import numpy as np
from sklearn.manifold import Isomap

import Constants
from dataset_managers.DatasetManagerFS import DatasetManager_FeatureSelection
from scaler_selector import all_scaler
from utilities import write_csv_validation

warnings.filterwarnings("ignore")
import time

def isomap_feature_selection(training_data, validation_data, test_data, scaler_title, type, index):
    id_train, X_train, y_train = training_data
    id_val, X_val, y_val = validation_data
    id_test,X_test, y_test = test_data

    min = math.inf
    embedding_winner = None
    i = 0

    print("===================")
    print(""" type:""" + str(type) + """
        index:""" + str(index) + """
        scaler_title:""" + str(scaler_title) + """""")
    print("===================")
    for components in range(1, 65):
        for neighbors in range(1, 65):
            if neighbors >= components:
                continue
            for path_method in ["D", "FW"]:
                for eigen_solver in ["arpack", "dense"]:
                    for neighbors_algorithm in ["brute", "kd_tree", "ball_tree"]:
                        for p in [1, 2]:
                            start_time = time.time()
                            try:
                                embedding = Isomap(n_components=components, n_neighbors=neighbors,
                                                   eigen_solver=eigen_solver, neighbors_algorithm=neighbors_algorithm,
                                                   path_method=path_method, p=p)

                                training_data = X_train.copy()
                                embedding.fit(training_data)
                                error = embedding.reconstruction_error()

                                if math.isnan(error):
                                    continue

                                print(i, " | ", embedding, "|", error, "|", round(time.time() - start_time, 6), "sec ")

                                if embedding.reconstruction_error() < min:
                                    min = embedding.reconstruction_error()
                                    embedding_winner = embedding

                            except Exception as e:
                                model_str = "Isomap(n_components=" + str(components) + ", n_neighbors=" + str(
                                    neighbors) + "," + \
                                            "eigen_solver=" + str(eigen_solver) + ",neighbors_algorithm=" + str(
                                    neighbors_algorithm) + "," + \
                                            "path_method=" + str(path_method) + ", p=" + str(p)
                                print(i, " | ", model_str, "|", "error", "|", round(time.time() - start_time, 6),
                                      "sec ", " -> ", e)

                            i += 1

    text = "Before transformation: (rows,columns)" + str(np.shape(X_train))

    embedding_winner.fit(X_train)
    X_train_transformed = embedding_winner.transform(X_train)
    X_val_transformed = embedding_winner.transform(X_val)
    X_test_transformed = embedding_winner.transform(X_test)
    text += "\nAfter transformation: (rows,columns)" + str(np.shape(X_train_transformed))

    write_csv_validation(X_train_transformed, X_val_transformed, X_test_transformed, y_train, y_val, y_test, id_train, id_val, id_test, "isomap", scaler_title, tag=type, index=index)

if __name__ == '__main__':

    Constants.folder = "a_priori"
    db = DatasetManager_FeatureSelection()
    np.random.seed(42)
    j = 0
    types = [Constants.folder]
    for type in types:
        for index, ((id_train,X_train, y_train), (id_val,X_val, y_val), (id_test,X_test, y_test)) in enumerate(db.get_data2(type)):
            all_possible_scaled_X_train, all_possible_scaled_X_val, all_possible_scaled_X_test = all_scaler(X_train, X_val, X_test)

            for i in range(len(all_possible_scaled_X_train)):
                scaler_title, train_set = all_possible_scaled_X_train[i]
                _, val_set = all_possible_scaled_X_val[i]
                _, test_set = all_possible_scaled_X_test[i]
                print("J =",j)
                isomap_feature_selection((id_train, train_set, y_train), (id_val, val_set, y_val), (id_test, test_set, y_test), scaler_title, type,index)
                j+=1
