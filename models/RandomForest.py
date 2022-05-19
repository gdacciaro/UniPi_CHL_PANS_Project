import json

from sklearn.ensemble import RandomForestClassifier

from models.AbstractModel import AbstractModel


class RandomForest(AbstractModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        n_estimators = self.get_param("n_estimators")
        criterion = self.get_param("criterion")
        max_depth = self.get_param("max_depth")
        min_samples_split = self.get_param("min_samples_split")
        min_samples_leaf = self.get_param("min_samples_leaf")

        self.model = RandomForestClassifier(random_state=42, n_estimators=n_estimators,
                                            criterion=criterion,
                                            max_depth=max_depth,min_samples_split=min_samples_split,
                                            min_samples_leaf=min_samples_leaf

                                            )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def get_important_features(self):
        return self.model.feature_importances_

    @staticmethod
    def get_all_combinations():
        n_estimatorss = [1,10,50]
        criterions = ["gini", "entropy"]
        max_depths = [2, 3, 6, 9]
        min_samples_splits = [0.5, 2, 3, 4]
        min_samples_leafs = [1, 2, 3]

        result = list()
        for criterion in criterions:
            for n_estimators in n_estimatorss:
                for max_depth in max_depths:
                    for min_samples_split in min_samples_splits:
                        for min_samples_leaf in min_samples_leafs:
                                result.append(RandomForest(criterion=criterion, n_estimators=n_estimators, max_depth=max_depth,
                                                           min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                                                ))
        return result
    
    def serialize(self):
        serialized_model = {"model":"RandomForest", "params": self.parameters}
        return json.dumps(serialized_model)

    @staticmethod
    def deserialize(str):
        params = json.loads(str)
        model = RandomForest(**params)
        return model

    def __str__(self):
        return "RandomForest | "+str(self.parameters)+""