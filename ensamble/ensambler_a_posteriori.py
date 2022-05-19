import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.exceptions import ConvergenceWarning

from Constants import ROOT
from Metrics import Metrics
from dataset_managers.DatasetManager import DatasetManager
from models.NaiveBayes import NaiveBayes
from models.DecisionTree import DecisionTree
from models.KNN import KNN
from models.LRegression import LRegression
from models.MLP import MLP
import pandas as pd

from models.RandomForest import RandomForest
from models.SVM import SVM

import warnings
warnings.filterwarnings("error")
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

np.random.seed(42)

df = pd.read_csv(ROOT + '/data/results_a_posteriori.csv', sep=',')
all_data = {}
for possible_db in ["meta","prote"]:
    all_data[possible_db] = {}
    for possible_fe in ["isomap","pca","rfe"]:
        all_data[possible_db][possible_fe] = {}
        for possible_scaler in ["min_max_scaling", "standard_scaling", "yeo_johnson_transformation", "unscaled"]:
            all_data[possible_db][possible_fe][possible_scaler] = list()


for i in range(len(df)) :
    db, fe, scaler, index, file_name = df.loc[i, "db"], df.loc[i, "fe"], df.loc[i, "scaler"], df.loc[i, "index"], df.loc[i, "file"]

    model_str = df.loc[i, "model"]
    pieces = model_str.split(" | ")
    model_type = pieces[0]
    model_param = pieces[1]
    model_param = str(model_param).replace("\'", "\"") \
                        .replace("[", "\"(").replace("]", ")\"") \
                        .replace("True", "true") \
                        .replace("False", "false")

    if model_type == "KNN":
        model = KNN.deserialize(model_param)
    elif model_type == "Naive Bayes":
        model = NaiveBayes.deserialize(model_param)
    elif model_type == "MLP":
        model = MLP.deserialize(model_param)
    elif model_type == "DecisionTree":
        model = DecisionTree.deserialize(model_param)
    elif model_type == "SVM":
        model = SVM.deserialize(model_param)
    elif model_type == "RandomForest":
        model = RandomForest.deserialize(model_param)
    elif model_type == "LRegression":
        model = LRegression.deserialize(model_param)

    key = file_name.replace(".csv","")
    all_data[db][fe][scaler].append((model, db, fe, scaler, index, file_name))

dataset_manager = DatasetManager()
for db in all_data:
    print(db)
    for fe in all_data[db]:
        for scaler in all_data[db][fe]:
            list_of_trial = all_data[db][fe][scaler]
            (model_1, db_1, fe_1, scaler_1, index_1, file_name_1) = list_of_trial[0]
            (model_2, db_2, fe_2, scaler_2, index_2, file_name_2) = list_of_trial[1]
            (model_3, db_3, fe_3, scaler_3, index_3, file_name_3) = list_of_trial[2]

            models = [("model_1", model_1.model),
                      ("model_2", model_2.model),
                      ("model_3", model_3.model)]
            ensemble = VotingClassifier(estimators=models)

            datasets = [
                ("db_1", file_name_1),
                ("db_2", file_name_2),
                ("db_3", file_name_3),
            ]

            avg_acc = 0
            avg_f1 = 0

            for table_name, file_name in datasets:
                import warnings
                warnings.filterwarnings('ignore')
                X_train, y_train, X_test, y_test, _, _ = dataset_manager.read_dataset_with_feature_selection(file_name)
                ensemble.fit(X_train, y_train)
                y_prec = ensemble.predict(X_test)
                acc = (Metrics.get_accuracy(y_test, y_prec))
                f1 = (Metrics.get_f1_score(y_test, y_prec))
                avg_acc += acc
                avg_f1 += f1

            print(fe_1.capitalize()+" & "+ scaler_1.replace("_","-").capitalize() +" & "+ str(round((avg_f1 / 3),6)) +" & "+ str(round((avg_acc / 3),6))+" \\\\ \hline")
