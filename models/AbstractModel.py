from abc import ABC, abstractmethod

from Metrics import Metrics


class AbstractModel(ABC):

    @abstractmethod
    def __init__(self, **kwargs):
        self.parameters = kwargs
        pass

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, train):
        pass

    def fit_and_predict(self,X_train, y_train, X_test, y_test):
        from warnings import simplefilter
        simplefilter(action='ignore')

        self.fit(X_train, y_train)
        out = self.predict(X_test)
        if out is None:
            result = {
                "f1": 0,
                "accuracy": 0
            }
        else:
            result = {
                "f1":Metrics.get_f1_score(out, y_test),
                "accuracy":Metrics.get_accuracy(out, y_test)
            }
        return result

    # @abstractmethod
    # def get_important_features(self):
    #     pass
    #
    # def plot_important_features(self):
    #     important_features = self.get_important_features()
    #     import matplotlib.pyplot as plt
    #     plt.bar([x for x in range(len(important_features))], important_features)
    #     plt.show()

    @staticmethod
    def get_all_combinations():
        pass

    def get_param(self, name, can_be_none=False):
        if not can_be_none:
            try:
                assert self.parameters[name] is not None
                return self.parameters[name]
            except Exception as e:
                raise KeyError("The hyperparamer "+str(name)+" is not valid :",e)