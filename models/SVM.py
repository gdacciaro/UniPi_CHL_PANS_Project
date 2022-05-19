from sklearn import svm
import json
from models.AbstractModel import AbstractModel


class SVM(AbstractModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        C = self.get_param("C")
        kernel = self.get_param("kernel")

        degree = self.get_param("degree", can_be_none=True)
        if degree is None: #Default value
            degree=3

        gamma = self.get_param("gamma", can_be_none=True)
        if gamma is None: #Default value
            gamma="scale"

        self.model = svm.SVC(random_state=42, gamma=gamma, kernel=kernel, C=C, degree=degree)

    def fit(self, X, y):
        self.fitted = self.model.fit(X, y)

    def predict(self, X):
        assert self.fitted is not None
        return self.fitted.predict(X)

    def __str__(self):
        return "SVM | "+str(self.parameters)+""

    @staticmethod
    def get_all_combinations():
        Cs = [1, 0.1, 0.001, 0.0001,0.0000001]
        kernels = ["linear", "rbf", "poly"]
        degrees = [2,3,4,8]
        gammas = ['scale','auto', 0.1, 0.001,0.000001]

        result = list()
        for C in Cs:
            for kernel in kernels:

                if kernel == "linear":
                    result.append(SVM(C=C, kernel=kernel))
                    continue

                for gamma in gammas:
                    if kernel != "poly":
                        result.append(SVM(C=C, kernel=kernel, gamma=gamma))
                        continue
                    else:
                        for degree in degrees:
                            result.append(SVM(C=C, kernel=kernel, gamma=gamma, degree=degree))
                        continue

        return result

    def serialize(self):
        serialized_model = {"model":"SVM", "params": self.parameters}
        return json.dumps(serialized_model)

    @staticmethod
    def deserialize(str):
        params = json.loads(str)
        model = SVM(**params)
        return model
