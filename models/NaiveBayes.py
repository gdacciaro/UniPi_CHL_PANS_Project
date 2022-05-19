from sklearn.naive_bayes import GaussianNB
import json
from models.AbstractModel import AbstractModel


class NaiveBayes(AbstractModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        var_smoothing = self.get_param("var_smoothing")

        self.model = GaussianNB(priors=None, var_smoothing=var_smoothing)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def __str__(self):
        return "Naive Bayes | "+str(self.parameters)+""

    @staticmethod
    def get_all_combinations():
        var_smoothings = [1, 1.e-01,1.e-02,1.e-03,1.e-04,1.e-05,
                          1.e-06,1.e-07,1.e-08]

        result = list()

        for var_smoothing in var_smoothings:
            result.append(NaiveBayes(var_smoothing=var_smoothing))
        return result
    
    def serialize(self):
        serialized_model = {"model":"Naive Bayes", "params": self.parameters}
        return json.dumps(serialized_model)

    @staticmethod
    def deserialize(str):
        params = json.loads(str)
        model = NaiveBayes(**params)
        return model
