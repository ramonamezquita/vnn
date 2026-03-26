from ._weigths_initializers import get_
from ._regualizers import get_regularizer

class PriorStrategy:
    def __init__(self, weights_initializer, regularizer):
        self.weights_initializer = weights_initializer
        self.regularizer = regularizer


    def make_weights_initializer(self) -> None:
        pass


    def make_regualizer(self) -> None:
        pass



class CauchyPrior:
    pass



class GaussianPrior:
    