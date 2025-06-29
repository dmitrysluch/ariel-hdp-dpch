from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any
import numpy as np


# Please make abstract class which has method compute_sensitivity taking parameter norm of type string,
# and two properties l1_sensitivity and l2_sensitivity,
# which are equal to the method compute_sensitivity called on strings "l1" and "l2". Use abc
class SensitivityMixin(ABC):
    @abstractmethod
    def compute_sensitivity(self, norm: str) -> float:
        """
        Compute sensitivity for a given norm

        Args:
            norm: String specifying the norm type ('l1' or 'l2')

        Returns:
            float: Computed sensitivity value
        """
        pass

    @abstractmethod
    def get_size(self) -> int:
        pass

    @property
    def l1_sensitivity(self) -> float:
        return self.compute_sensitivity("l1")

    @property
    def l2_sensitivity(self) -> float:
        return self.compute_sensitivity("l2")


# Implement SensitivityMixin if a function is lipschitz. The class has abstract method get_lipschitz_parameter(norm: str) -> float, and abstract method get_arguments(),
# which returns objects satisfying sensitivity SensitivityMixin.
# Method compute_sensitivity is implemented as following, for each argument sensitivity is calculated, then the total sensitivity of arguments is calculated
# (as a first or a second norm). Finally it is multiplied by corresponding get_lipschitz_parameter value and returned
class FunctionSensitivityMixin(SensitivityMixin, ABC):
    @abstractmethod
    def get_lipschitz_parameter(self, norm: str) -> float:
        pass

    @abstractmethod
    def get_arguments(self) -> SensitivityMixin:
        pass

    def compute_sensitivity(self, norm: str) -> float:
        args = self.get_arguments()
        sensitivities = [arg.compute_sensitivity(norm) for arg in args]

        if norm == "l1":
            total_sensitivity = sum(sensitivities)
        elif norm == "l2":
            total_sensitivity = (sum(s**2 for s in sensitivities)) ** 0.5
        else:
            raise ValueError(f"Unsupported norm: {norm}")

        return self.get_lipschitz_parameter(norm) * total_sensitivity


# Implement bounded mixin. It is a class which has a single abstract method max_norm,
# which recieves a norm parameter of type string and returns float
class BoundedMixin(ABC):
    @abstractmethod
    def max_norm(self, norm: str) -> float:
        """
        Returns maximum norm value
        Args:
            norm: String specifying the norm type ('l1' or 'l2')
        Returns:
            float: Maximum norm value
        """
        pass


# Please implement NoiseMixin abstact class
# Noise should provide evaluate() abstract method, which recieves args array of type np.ndarray and sensitivity of type float.
class NoiseMixin(ABC):
    @abstractmethod
    def evaluate(self, args: np.ndarray, sensitivity: float) -> np.ndarray:
        """
        Add noise to the input array
        Args:
            args: Input array to add noise to
            sensitivity: Sensitivity value for noise calculation
        Returns:
            np.ndarray: Array with added noise
        """
        pass


# Make an SQLQuery dataclass which consists of a query as str and list of parameters as dict
@dataclass
class SQLQuery:
    query: str
    parameters: Dict[str, Any]
    rv_name: str


class SQLMixin(ABC):
    @abstractmethod
    def get_sql(self) -> SQLQuery:
        """
        Returns SQL query computing this column

        Returns:
            str: SQL query
        """
        pass

    @abstractmethod
    def validate_columns(self, columns: list["Column"]) -> None:
        """
        Validate columns used in the SQL query

        Args:
            columns: List of Column objects to validate
        """
        pass
