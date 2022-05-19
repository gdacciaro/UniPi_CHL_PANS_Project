import warnings

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA  # to perform PCA to plot data
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from dataset_managers.DatasetManagerFS import DatasetManager_FeatureSelection
from scaler_selector import all_scaler
from utilities import write_csv_validation

warnings.filterwarnings("ignore")

def pca_gridsearch(X, y):
    pipe = Pipeline(
        [
            # the reduce_dim stage is populated by the param_grid
            ("reduce_dim", "passthrough"),
            ("classify", LinearSVC(dual=False, max_iter=10000)),
        ]
    )

    N_FEATURES_OPTIONS = []
    C_OPTIONS = [1, 10, 100, 1000]

    for i in range(2, len(X) + 1):
        N_FEATURES_OPTIONS.append(i)


    param_grid = [
        {
            "reduce_dim": [PCA()],
            "reduce_dim__n_components": N_FEATURES_OPTIONS,
            "classify__C": C_OPTIONS,
        }
    ]
    reducer_labels = ["PCA"]

    grid = GridSearchCV(pipe, n_jobs=1, param_grid=param_grid)
    grid.fit(X, y)

    mean_scores = grid.cv_results_["mean_test_score"]#np.array(grid.cv_results_["mean_test_score"])
    # scores are in the order of param_grid iteration, which is alphabetical
    mean_scores = mean_scores.reshape(len(C_OPTIONS), -1, len(N_FEATURES_OPTIONS))
    # select score for best C
    mean_scores = mean_scores.max(axis=0)
    list_scores = mean_scores.flatten().tolist()
    # select the max accuracy index
    max_value = max(list_scores)
    index_max = list_scores.index(max_value)
    bar_offsets = np.arange(len(N_FEATURES_OPTIONS)) * (len(reducer_labels) + 1) + 0.5

    plt.figure(figsize=(10,10))
    COLORS = "bgrcmyk"
    for i, (label, reducer_scores) in enumerate(zip(reducer_labels, mean_scores)):
        plt.bar(bar_offsets + i, reducer_scores, label=label, color=COLORS[i])

    plt.title("Comparing feature reduction techniques")
    plt.xlabel("Reduced number of features")
    plt.xticks(bar_offsets + len(reducer_labels) / 2, N_FEATURES_OPTIONS)
    plt.ylabel("Omics classification accuracy")
    plt.ylim((0, 1))
    plt.legend(loc="upper left")

    plt.show()
    
    return N_FEATURES_OPTIONS[index_max]

def PCA_feature_selection2(training_data, validation_data, test_data, scaler_title, type, index):
    id_train, X_train, y_train = training_data
    id_val, X_val, y_val = validation_data
    id_test, X_test, y_test = test_data
    
    n_features = pca_gridsearch(X_train, y_train)
    pca = PCA(n_components=n_features)

    X_train_transformed = pca.fit_transform(X_train)
    X_val_transformed = pca.transform(X_val)
    X_test_transformed = pca.transform(X_test)

    write_csv_validation(X_train_transformed, X_val_transformed, X_test_transformed, y_train, y_val, y_test, id_train, id_val, id_test, "pca", scaler_title, tag=type,index=index)

if __name__ == '__main__':
    
    db = DatasetManager_FeatureSelection()

    types = ["meta", "prote"]
    for type in types:
        for index, ((id_train,X_train, y_train), (id_val,X_val, y_val), (id_test,X_test, y_test)) in enumerate(db.get_data2(type)):
            all_possible_scaled_X_train, all_possible_scaled_X_val, all_possible_scaled_X_test = all_scaler(X_train, X_val, X_test)

            for i in range(len(all_possible_scaled_X_train)):
                scaler_title, train_set = all_possible_scaled_X_train[i]
                _, val_set = all_possible_scaled_X_val[i]
                _, test_set = all_possible_scaled_X_test[i]

                PCA_feature_selection2((id_train, train_set, y_train), (id_val, val_set, y_val), (id_test, test_set, y_test), scaler_title, type,index)
