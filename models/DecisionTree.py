import json

from sklearn.tree import DecisionTreeClassifier
from models.AbstractModel import AbstractModel

class DecisionTree(AbstractModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        criterion = self.get_param("criterion")
        splitter = self.get_param("splitter")
        max_depth = self.get_param("max_depth")
        min_samples_split = self.get_param("min_samples_split")
        min_samples_leaf = self.get_param("min_samples_leaf")
        ccp_alpha = self.get_param("ccp_alpha")

        self.model = DecisionTreeClassifier(random_state=42, criterion=criterion, splitter=splitter, max_depth=max_depth,
                                            min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf
                                            , ccp_alpha = ccp_alpha)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def get_important_features(self):
        return self.model.feature_importances_

    @staticmethod
    def get_all_combinations():
        criterions = ["gini", "entropy"]
        splitters = ["best", "random"]
        max_depths = [2,3,6]
        min_samples_splits = [0.5,2,4,5]
        min_samples_leafs = [1,3,5]
        ccp_alphas = [0,0.001, 0.5, 1]

        result = list()
        for criterion in criterions:
            for splitter in splitters:
                for max_depth in max_depths:
                    for min_samples_split in min_samples_splits:
                        for min_samples_leaf in min_samples_leafs:
                            for ccp_alpha in ccp_alphas:
                                result.append(DecisionTree(criterion=criterion, splitter=splitter, max_depth=max_depth,
                                                           min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                                           ccp_alpha = ccp_alpha
                                                           ))
        return list()
    
    def serialize(self):
        serialized_model = {"model":"DecisionTree", "params": self.parameters}
        return json.dumps(serialized_model)

    @staticmethod
    def deserialize(str):
        params = json.loads(str)
        model = DecisionTree(**params)
        return model

    def __str__(self):
        return "DecisionTree | "+str(self.parameters)+""