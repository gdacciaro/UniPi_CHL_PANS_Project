import time

import numpy as np

import Constants
from Metrics import Metrics
from distribuited_model_selection.DatabaseManager import getUnsolvedAttempt, postResult
from utilities import  k_fold


def execute_trials(features_dev, target_dev, k):
    start_time = time.time()
    id, model = getUnsolvedAttempt()

    try:
        result_kfold = k_fold(features_dev, target_dev, k=k)
        accuracies = list()
        for fold in result_kfold:
            X_train, y_train = fold["train"], fold["target_train"]
            X_val, y_val = fold["validation"], fold["target_validation"]
            model.fit(X_train, y_train)
            predicted = model.predict(X_val)
            accuracy = Metrics.get_accuracy(predicted, y_val) if predicted is not None else 0
            accuracies.append(accuracy)
        final_accuracy = np.mean(accuracies)
        exec_time = time.time()-start_time
        postResult(id,final_accuracy,exec_time)

        print("Id:", id, " Result:", final_accuracy, " Time:",exec_time)
        print("model:", model)

        return True
    except Exception as e:
        exec_time = time.time() - start_time
        postResult(id, Constants.ERROR_VALUE ,exec_time)
        print("Id:", id, " Result:", "ERROR", " Time:",exec_time)
        print("Exception:" ,e)
        print("model:", model)
        return False