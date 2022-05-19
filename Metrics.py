from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


class Metrics:

    @staticmethod
    def get_accuracy(list_a, list_b):
        Metrics.__check_preconditions(list_a, list_b)
        return accuracy_score(list_a, list_b)

    @staticmethod
    def get_f1_score(list_a, list_b):
        Metrics.__check_preconditions(list_a, list_b)
        return f1_score(list_a, list_b, average='macro')

    @staticmethod
    def __check_preconditions(list_a, list_b):
        assert list_a is not None
        assert list_b is not None
        assert len(list_a) == len(list_b)
