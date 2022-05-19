from sklearn.neural_network import MLPClassifier
import json
from models.AbstractModel import AbstractModel


class MLP(AbstractModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.converged = True

        batch_size = self.get_param("batch_size")
        learning_rate_init = self.get_param("learning_rate_init")
        activation = self.get_param("activation")
        solver = self.get_param("solver")
        max_iter = self.get_param("max_iter")
        alpha = self.get_param("alpha")
        early_stopping = self.get_param("early_stopping")
        hidden_layer_sizes = self.parse_topology(self.get_param("hidden_layer_sizes"))

        self.model = MLPClassifier(random_state=42, max_iter=max_iter,
                                   solver=solver,
                                   alpha=alpha,
                                   early_stopping = early_stopping,
                                   activation=activation,
                                   hidden_layer_sizes=hidden_layer_sizes,
                                   batch_size=batch_size, learning_rate_init=learning_rate_init)

        import warnings
        warnings.filterwarnings("error")

    def fit(self, X, y):
        import sklearn
        try:
            self.model.fit(X, y)
        except sklearn.exceptions.ConvergenceWarning:
            self.converged=False

    def predict(self, X):
        if not self.converged:
            return None
        return self.model.predict(X)

    @staticmethod
    def get_all_combinations():
        hidden_layer_sizes = [(8,),(16,),(64,),
                              (8,16,), (32,64),
                              (8,32,), (64,32,),
                              (64,64,32,),(32,32,8,),
                              (16,16,16,8,),
                              (32,16,)
                              ]
        batch_sizes = [1,4,16]
        activations = ["tanh", "relu"]
        early_stoppings = [True]
        solvers = ["adam"]
        max_iters = [150,350]
        alphas = [0.1, 0.001, 0.00001, 0.000001,]
        learning_rate_inits = [0.1, 0.0001, 0.0000001]

        result = list()
        for batch_size in batch_sizes:
            for solver in solvers:
                for max_iter in max_iters:
                    for alpha in alphas:
                        for early_stopping in early_stoppings:
                            for hidden_layer_size in hidden_layer_sizes:
                                for activation in activations:
                                    for learning_rate_init in learning_rate_inits:
                                        result.append(MLP(batch_size=batch_size,
                                                          activation=activation,
                                                          alpha=alpha,
                                                          early_stopping = early_stopping,
                                                          hidden_layer_sizes=hidden_layer_size,
                                                          max_iter=max_iter,
                                                          learning_rate_init=learning_rate_init,
                                                          solver=solver))
        return result

    def __str__(self):
        return "MLP | "+str(self.parameters)+""
    
    def serialize(self):
        serialized_model = {"model":"MLP", "params": self.parameters}
        return json.dumps(serialized_model)

    @staticmethod
    def deserialize(str):
        params = json.loads(str)
        model = MLP(**params)
        return model

    def parse_topology(self, topology_str):
        if type(topology_str) == tuple or type(topology_str) == list:
            return topology_str
        topology_str = topology_str.replace(" ","")
        results = list()
        buffer = ""
        for char in topology_str:
            if (char != ")") and (char != "("):
                if char != ",":
                    buffer = buffer + str(char)
                else:
                    results.append(int(buffer))
                    buffer = ""
        results.append(int(buffer))
        return tuple(results)


if __name__ == '__main__':
    print(MLP.deserialize("""
    {"batch_size": 1,
 "activation": "logistic",
 "alpha": 0.1,
 "early_stopping": true,
 "hidden_layer_sizes": "(8, 16)",
 "max_iter": 50,
 "learning_rate_init": 0.1,
 "solver": "adam"}
 """).parameters)