from sklearn.neighbors import KNeighborsClassifier
import json
from models.AbstractModel import AbstractModel


class KNN(AbstractModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        n_neighbors = self.get_param("n_neighbors")
        weights = self.get_param("weights")
        algorithm = self.get_param("algorithm")

        self.model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def __str__(self):
        return "KNN | "+str(self.parameters)+""

    @staticmethod
    def get_all_combinations():
        n_neighbors = [1,2,3,5,8,10]
        weights = ["uniform", "distance"]
        algorithms = ["ball_tree","kd_tree", "brute"]

        result = list()
        for n_neighbor in n_neighbors:
            for weight in weights:
                for algorithm in algorithms:
                    result.append(KNN(n_neighbors=n_neighbor,weights=weight,algorithm=algorithm))
        return result

    def serialize(self):
        serialized_model = {"model":"KNN", "params": self.parameters}
        return json.dumps(serialized_model)

    @staticmethod
    def deserialize(str):
        params = json.loads(str)
        model = KNN(**params)
        return model

    def get_important_features(self):
        pass