from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Protocol

import numpy as np
from pydantic import BaseModel

from dpch.common.schema import SchemaDataset


class QueryProto(Protocol):
    tp: str

    def get_arguments() -> list["QueryProto"]: ...


class DPValueError(ValueError):
    pass


class DPQueryMixin(BaseModel, ABC):
    @abstractmethod
    def shape(self, ds: SchemaDataset) -> tuple[int, int]:
        pass

    def len(self, ds: SchemaDataset) -> int:
        return self.shape(ds)[0]

    def n_cols(self, ds: SchemaDataset) -> int:
        return self.shape(ds)[1]

    @abstractmethod
    def max_changed_rows(self, ds: SchemaDataset) -> list[int]:
        pass

    @abstractmethod
    def max_over_columns(self, ds: SchemaDataset) -> list[float]:
        pass

    @abstractmethod
    def min_over_columns(self, ds: SchemaDataset) -> list[float]:
        pass

    # Please implement max_norm_along_row method computing the upper bound on corresponding norm.
    # Get maximum (absolute) value for each row using max_along_row/min_along_row method.
    @lru_cache
    def max_norm_over_columns(self, ds: SchemaDataset, norm: str) -> list[float]:
        """
        Returns maximum norm value over columns
        Args:
            norm: String specifying the norm type ('l1' or 'l2')
        Returns:
            float: Maximum norm value
        """
        max_vals = self.max_over_columns(ds)
        min_vals = self.min_over_columns(ds)
        abs_vals = np.maximum(np.abs(max_vals), np.abs(min_vals))

        if norm == "l1":
            return [self.len(ds) * x for x in abs_vals]
        elif norm == "l2":
            return [self.len(ds) ** 0.5 * x for x in abs_vals]
        else:
            raise DPValueError(f"Unsupported norm: {norm}")

    # Please implement method computing bound on corresponding total norm of a dataframe.
    # Use get_len and max_norm_over_rows methods
    @lru_cache
    def max_total_norm(self, ds: SchemaDataset, norm: str) -> float:
        """
        Returns maximum norm value of whole dataframe
        Args:
            norm: String specifying the norm type ('l1' or 'l2')
        Returns:
            float: Maximum norm value
        """
        if norm == "l1":
            return sum(self.max_norm_over_columns(ds, norm))
        elif norm == "l2":
            return sum(self.max_norm_over_columns(ds, norm) ** 2) ** 0.5
        else:
            raise DPValueError(f"Unsupported norm: {norm}")

    @abstractmethod
    def sensitivity_over_columns(self, ds: SchemaDataset, norm: str) -> list[float]:
        """
        Compute sensitivity for a given norm over columns

        Args:
            norm: String specifying the norm type ('l1' or 'l2')

        Returns:
            float: Computed sensitivity value
        """
        pass

    @lru_cache
    def sensitivity(self, ds: SchemaDataset, norm: str) -> float:
        """
        Compute sensitivity for a given norm for a whole dataframe

        Args:
            norm: String specifying the norm type ('l1' or 'l2')

        Returns:
            float: Computed sensitivity value
        """
        if norm == "l1":
            return sum(self.sensitivity_over_columns(ds, norm))
        elif norm == "l2":
            return sum(self.sensitivity_over_columns(ds, norm) ** 2) ** 0.5
        else:
            raise DPValueError(f"Unsupported norm: {norm}")

    @abstractmethod
    def validate_against_schema(self, ds: SchemaDataset) -> None:
        """
        Validate query provided schema

        Args:
            schema: schema of dataset to be queried
        """
        pass

    # Lets make repr_columns abstract method for DPQueryMixin.
    # It takes SchemaDataset ds and returns a list of strings
    @abstractmethod
    def repr_columns(self, ds: SchemaDataset) -> list[str]:
        """
        Returns short representation for each of resulting columns

        Args:
            ds: schema dataset to query column information from

        Returns:
            list[str]: List of column names
        """
        pass

    class Config:
        frozen = True


# Please fix the following mixin. It must implement compute_sensitivity_over_columns method.
# It is expected that all arguments have exactly same shape (validate this),
# and the output has same number of columns as arguments.
# The sensitivity is computed for each column separately, using the same lipschitz parameter.
class BroadcastLipschitzMixin(DPQueryMixin, ABC):
    @abstractmethod
    def get_lipschitz_parameter(self, ds: SchemaDataset, norm: str) -> float:
        pass

    @abstractmethod
    def get_arguments(self) -> list[DPQueryMixin]:
        pass

    def sensitivity_over_columns(self, ds: SchemaDataset, norm: str) -> list[float]:
        args = self.get_arguments()
        if not args:
            return []

        # Validate shapes
        shape = args[0].shape(ds)
        for arg in args[1:]:
            if arg.shape(ds) != shape:
                raise DPValueError("All arguments must have the same shape")

        # Compute per-column sensitivities
        sensitivities_per_col = [arg.sensitivity_over_columns(ds, norm) for arg in args]
        lipschitz = self.get_lipschitz_parameter(ds, norm)

        if norm == "l1":
            return [
                lipschitz * sum(s[i] for s in sensitivities_per_col)
                for i in range(shape[1])
            ]
        elif norm == "l2":
            return [
                lipschitz * (sum(s[i] ** 2 for s in sensitivities_per_col)) ** 0.5
                for i in range(shape[1])
            ]
        else:
            raise DPValueError(f"Unsupported norm: {norm}")
