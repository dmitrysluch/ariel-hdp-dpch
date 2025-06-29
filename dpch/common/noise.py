import numpy as np
from dpch.common.interface import NoiseMixin


# Please make LaplacianNoise class. It should satisfy following interface. The constructor should recieve a single parameter epsilon.
# When evaluating LaplacianNoise is applied using numpy with zero mean and lambda set to sensitivity/eps
class LaplacianNoise(NoiseMixin):
    def __init__(
        self,
        total_epsilon: float,
        queries_total: int,
        l1_sensitivity: float,
    ):
        eps_per_query = total_epsilon / queries_total
        self.scale = l1_sensitivity / eps_per_query

    def evaluate(self, args: np.ndarray) -> np.ndarray:
        return args + np.random.laplace(loc=0, scale=self.scale, size=args.shape)


class GaussianNoise(NoiseMixin):
    def __init__(
        self,
        total_epsilon: float,
        total_delta: float,
        queries_total: int,
        l2_sensitivity: float,
    ):
        delta_prime = total_delta / 2  # common choice for delta'
        eps_per_query = total_epsilon / np.sqrt(queries_total * np.log(1 / delta_prime))
        delta_per_query = (total_delta - delta_prime) / queries_total
        self.std = (
            l2_sensitivity * np.sqrt(2 * np.log(1.25 / delta_per_query)) / eps_per_query
        )

    def evaluate(self, args: np.ndarray) -> np.ndarray:
        return args + np.random.normal(loc=0, scale=self.std, size=args.shape)
