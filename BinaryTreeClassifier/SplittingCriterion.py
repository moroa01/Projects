import numpy as np
import math

class SplittingCriterion:

    function: callable

    def _binary_gini_function(self, input):
        p = np.mean(input)
        return 2 * p * (1 - p)

    def _binary_scaled_entropy(self, input):
        p = np.mean(input)
        return -(p * np.log2(p)) - ((1-p) * np.log2(1-p))

    def _square_root_impurity(self, input):
        p = np.mean(input)
        return math.sqrt(p*(1-p))

    _functions = {
        'gini': _binary_gini_function,
        'entropy': _binary_scaled_entropy,
        'square root': _square_root_impurity
    }

    def __init__(self, name = "gini"):
        self.function = self._functions[name]

    def __call__(self, input):
        return self.function(self, input)