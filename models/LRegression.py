from sklearn.linear_model import LogisticRegression
import json
from models.AbstractModel import AbstractModel


class LRegression(AbstractModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        penalty = self.get_param("penalty")
        C = self.get_param("C")
        class_weight = self.get_param("class_weight",can_be_none=True)
        solver = self.get_param("solver")
        max_iter = self.get_param("max_iter")

        self.model = LogisticRegression(random_state=42,
                                        penalty=penalty, C=C, class_weight=class_weight, solver=solver, max_iter = max_iter)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def __str__(self):
        return "LRegression | "+str(self.parameters)+""

    @staticmethod
    def get_all_combinations():
        penalties = ["l1", "l2", "none"] 
        Cs = [1.0, 0.5, 0.25, 0.1]
        class_weights = ['balanced']
        solvers = ["newton-cg", "lbfgs", "liblinear"]
        max_iters = [150,350]

        result = list()
        for solver in solvers:
            for penalty in penalties:
                for max_iter in max_iters:
                    for class_weight in class_weights:
                        for C in Cs:
                            if solver == "newton-cg" and penalty == "l1":
                                continue
                            if solver == "lbfgs" and penalty == "l1":
                                continue
                            if solver == "liblinear" and penalty == "none":
                                continue
                            result.append(LRegression(penalty=penalty, C=C, class_weight=class_weight, solver=solver, max_iter=max_iter))
        return result
    
    def serialize(self):
        serialized_model = {"model":"LRegression", "params": self.parameters}
        return json.dumps(serialized_model)

    @staticmethod
    def deserialize(str):
        params = json.loads(str)
        model = LRegression(**params)
        return model

if __name__ == '__main__':
    l = LRegression.get_all_combinations()
    for i in l:
        print(i)
