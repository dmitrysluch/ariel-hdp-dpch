from functools import lru_cache
from typing import Literal, Optional

from pydantic import Field

from dpch.common.queries.interface import (
    BroadcastLipschitzMixin,
    DPQueryMixin,
    DPValueError,
)
from dpch.common.schema import SchemaColumn, SchemaDataFrame, SchemaDataset

# I done some refactoring to dpch.common.queries.interface, could you please fix the query implementations.

# Please implement sensitivity_over_columns, max_over_columns and max_norm_over_columns.

# You can use BroadcastLipschitzMixin for simplifying implementation for lipshwitz column-wise functions (sum and mean)

# get_size method was removed and replaced with .shape, please implement it (you have to check the length of the first column in dataset and the number of columns)

# for max_over_columns and min_over_columns you must presume order of columns (norm values must be the same as column names values)

# you have to extract dataframe from dataset to get columns, see common.schema

# for column query max_norm_over_columns must return minimum of max_norm specified in schema (l1 oor l2 correspondingly) or norm computed using max value (you can use super())


# For ColumnsQuery please extract looking up the dataframe in dataset to a new method
class ColumnsQuery(DPQueryMixin):
    tp: Literal["columns"]
    columns: tuple[str, ...]
    dataframe: str

    def get_arguments(self) -> list[DPQueryMixin]:
        return []

    @lru_cache
    def _get_dataframe(self, ds: SchemaDataset) -> SchemaDataFrame:
        """Get the dataframe that matches self.dataframe from the dataset."""
        dataframe = next(
            (df for df in ds.dataframes if df.name == self.dataframe), None
        )
        if dataframe is None:
            raise DPValueError(f"Dataframe {self.dataframe} not found in dataset")
        return dataframe

    @lru_cache
    def _get_column(self, dataframe: SchemaDataFrame, col_name: str) -> SchemaColumn:
        """Get the column object by name from the dataframe."""
        col = next((col for col in dataframe.columns if col.name == col_name), None)
        if col is None:
            raise DPValueError(f"Column {col_name} not found")
        return col

    @lru_cache
    def shape(self, ds) -> tuple[int, int]:
        dataframe = self._get_dataframe(ds)

        return (dataframe.n_rows, len(self.columns))

    @lru_cache
    def max_changed_rows(self, ds) -> int:
        return self._get_dataframe(ds).max_changed_rows

    @lru_cache
    def max_over_columns(self, ds) -> list[float]:
        dataframe = self._get_dataframe(ds)
        return [
            self._get_column(dataframe, col_name).max_val for col_name in self.columns
        ]

    @lru_cache
    def min_over_columns(self, ds) -> list[float]:
        dataframe = self._get_dataframe(ds)
        return [
            self._get_column(dataframe, col_name).min_val for col_name in self.columns
        ]

    @lru_cache
    def max_norm_over_columns(self, ds, norm: str) -> list[float]:
        dataframe = self._get_dataframe(ds)
        # Use super() to get computed_norm
        computed_norm = super().max_norm_over_columns(ds, norm)
        result = []
        for col_name in self.columns:
            col = self._get_column(dataframe, col_name)
            computed_col_norm = computed_norm[self.columns.index(col_name)]
            if norm == "l1" and col.max_l1_norm is not None:
                result.append(min(col.max_l1_norm, computed_col_norm))
            elif norm == "l2" and col.max_l2_norm is not None:
                result.append(min(col.max_l2_norm, computed_col_norm))
            elif norm in ["l1", "l2"]:
                result.append(computed_col_norm)
            else:
                DPValueError("Unsupported norm")
        return result

    @lru_cache
    def sensitivity_over_columns(self, ds, norm: str) -> list[float]:
        dataframe = self._get_dataframe(ds)
        return [
            abs(
                self._get_column(dataframe, col_name).max_val
                - self._get_column(dataframe, col_name).min_val
            )
            * dataframe.max_changed_rows
            for col_name in self.columns
        ]

    @lru_cache
    def validate_against_schema(self, ds):
        dataframe = self._get_dataframe(ds)
        query_cols = set(self.columns)
        df_cols = set(col.name for col in dataframe.columns)
        if len(query_cols & df_cols) != len(query_cols):
            raise DPValueError("Non existent columns queried")


# Please fix SumQuery. The resulting shape is 1 x columns.shape()[1] (as all rows are summed).
# Max over columns, min_over_columns, max_norm_over_columns change accordingly
# (that is max and min, are multiplied by size, l1 norm is left unchanged, l2 norm is assigned to l1 norm)
class SumQuery(BroadcastLipschitzMixin):
    tp: Literal["sum"]
    columns: "OneOfQueries" = Field(discriminator="tp")

    def get_arguments(self) -> list[DPQueryMixin]:
        return [self.columns]

    @lru_cache
    def shape(self, ds) -> tuple[int, int]:
        # All rows are summed, so shape is (1, number of columns)
        return (1, self.columns.n_cols(ds))

    def max_changed_rows(self, ds) -> int:
        return 1

    @lru_cache
    def get_lipschitz_parameter(self, ds, norm: str) -> float:
        # Summation is 1-Lipschitz under both L1 and L2 norms
        if norm in ("l1", "l2"):
            return 1.0
        raise DPValueError(f"Unsupported norm: {norm}")

    @lru_cache
    def max_over_columns(self, ds):
        # Multiply max value by number of rows
        size = self.columns.n_cols(ds)
        return [v * size for v in self.columns.max_over_columns(ds)]

    @lru_cache
    def min_over_columns(self, ds):
        # Multiply min value by number of rows
        size = self.columns.n_cols(ds)
        return [v * size for v in self.columns.min_over_columns(ds)]

    @lru_cache
    def max_norm_over_columns(self, ds, norm: str) -> list[float]:
        norm_via_min_max = super().max_norm_over_columns(ds, norm)
        # l1 norm is unchanged, l2 norm is assigned to l1 norm
        l1_norms = self.columns.max_norm_over_columns(ds, "l1")
        if norm in ["l1", "l2"]:
            return [min(x, y) for x, y in zip(l1_norms, norm_via_min_max)]
        else:
            raise DPValueError(f"Unsupported norm: {norm}")

    def validate_against_schema(self, ds):
        self.columns.validate_against_schema(ds)


# Please fix MeanQuery similarly to sum query.
# For lipschitz parameter it is 1 / columns/len(ds).
# For max_over_columns and min_over_columns no multiplication is needed.
# l1 norm is divided by len, l2 is l1 divided by len
class MeanQuery(BroadcastLipschitzMixin):
    tp: Literal["mean"]
    columns: "OneOfQueries" = Field(discriminator="tp")

    def get_arguments(self) -> list[DPQueryMixin]:
        return [self.columns]

    @lru_cache
    def shape(self, ds) -> tuple[int, int]:
        # All rows are summed, so shape is (1, number of columns)
        return (1, self.columns.n_cols(ds))

    def max_changed_rows(self, ds) -> int:
        return 1

    @lru_cache
    def max_over_columns(self, ds):
        # No multiplication needed for mean
        return self.columns.max_over_columns(ds)

    @lru_cache
    def min_over_columns(self, ds):
        # No multiplication needed for mean
        return self.columns.min_over_columns(ds)

    @lru_cache
    def max_norm_over_columns(self, ds, norm: str) -> list[float]:
        norm_via_min_max = super().max_norm_over_columns(ds, norm)
        l1_norms = self.columns.max_norm_over_columns(ds, "l1")
        size = self.columns.shape(ds)[0]
        if norm in ["l1", "l2"]:
            return [min(x / size, y) for x, y in zip(l1_norms, norm_via_min_max)]
        else:
            raise DPValueError(f"Unsupported norm: {norm}")

    @lru_cache
    def get_lipschitz_parameter(self, ds, norm: str) -> float:
        size = self.columns.shape(ds)[0]
        if norm in ("l1", "l2"):
            return 1.0 / size
        raise DPValueError(f"Unsupported norm: {norm}")

    def validate_against_schema(self, ds):
        self.columns.validate_against_schema(ds)


# Please fix Covariance query.
# If columns2 is not set, then self-covariance is computed,
# so you may assume it to be equal to columns1. In both cases result
# is a single row with columns1.n_cols * columns2.n_cols
# columns that is a flattened covariance matrix.

# Considering max and min values, calculate all combinations of max and min,
# and take max and min of them correspondingly
# (because two negative minimums may yield positive maximum)

# max_norm_over_columns for both second and first norms is calculated using second norm by Cauchy-Schwarz (multiply norms, divide by len and take sqrt)

# formula for sensitivity is correct, it is
# d1 * m2 + d2 * m1 + d1 * d2 but note that you have to iterate over all pairs of columns


class CovarianceQuery(DPQueryMixin):
    tp: Literal["covariance"]
    columns1: "OneOfQueries" = Field(discriminator="tp")
    columns2: Optional["OneOfQueries"] = Field(discriminator="tp", default=None)

    def get_arguments(self) -> list[DPQueryMixin]:
        return (
            [self.columns1] if self.columns2 is None else [self.columns1, self.columns2]
        )

    def _get_columns2(self):
        """Get columns2, using columns1 for self-covariance if columns2 is None."""
        return self.columns2 if self.columns2 is not None else self.columns1

    def shape(self, ds) -> tuple[int, int]:
        # Result is a single row with columns1.n_cols * columns2.n_cols columns (covariance matrix)
        n_cols1 = self.columns1.n_cols(ds)
        n_cols2 = self._get_columns2().n_cols(ds)
        return (1, n_cols1 * n_cols2)

    def max_changed_rows(self, ds) -> int:
        return 1

    def max_over_columns(self, ds):
        # Calculate all combinations of max and min, take max of them
        columns2 = self._get_columns2()
        max1 = self.columns1.max_over_columns(ds)
        min1 = self.columns1.min_over_columns(ds)
        max2 = columns2.max_over_columns(ds)
        min2 = columns2.min_over_columns(ds)

        result = []
        for i in range(len(max1)):
            for j in range(len(max2)):
                # All combinations: max1*max2, max1*min2, min1*max2, min1*min2
                combinations = [
                    max1[i] * max2[j],
                    max1[i] * min2[j],
                    min1[i] * max2[j],
                    min1[i] * min2[j],
                ]
                result.append(max(combinations))
        return result

    def min_over_columns(self, ds):
        # Calculate all combinations of max and min, take min of them
        columns2 = self._get_columns2()
        max1 = self.columns1.max_over_columns(ds)
        min1 = self.columns1.min_over_columns(ds)
        max2 = columns2.max_over_columns(ds)
        min2 = columns2.min_over_columns(ds)

        result = []
        for i in range(len(max1)):
            for j in range(len(max2)):
                # All combinations: max1*max2, max1*min2, min1*max2, min1*min2
                combinations = [
                    max1[i] * max2[j],
                    max1[i] * min2[j],
                    min1[i] * max2[j],
                    min1[i] * min2[j],
                ]
                result.append(min(combinations))
        return result

    def max_norm_over_columns(self, ds, norm: str) -> list[float]:
        # Calculate using l2 norm by Cauchy-Schwarz: multiply norms, divide by len and take sqrt
        columns2 = self._get_columns2()
        norms1 = self.columns1.max_norm_over_columns(ds, "l2")
        norms2 = columns2.max_norm_over_columns(ds, "l2")
        size = self.columns1.len(ds)
        norm_via_min_max = super().max_norm_over_columns(ds, norm)

        result = []
        for i in range(len(norms1)):
            for j in range(len(norms2)):
                if norm in ["l1", "l2"]:
                    # Cauchy-Schwartz bound: multiply norms, divide by len and take sqrt
                    result.append(
                        min((norms1[i] * norms2[j] / size) ** 0.5),
                        norm_via_min_max[i * len(norms1) + j],
                    )
                else:
                    raise DPValueError(f"Unsupported norm: {norm}")
        return result

    def sensitivity_over_columns(self, ds, norm: str) -> list[float]:
        # Sensitivity for covariance using Cauchy-Schwarz bound: d1 * m2 + d2 * m1 + d1 * d2
        # Iterate over all pairs of columns
        columns2 = self._get_columns2()
        d1 = self.columns1.sensitivity_over_columns(ds, "l2")
        d2 = columns2.sensitivity_over_columns(ds, "l2")
        m1 = self.columns1.max_norm_over_columns(ds, "l2")
        m2 = columns2.max_norm_over_columns(ds, "l2")

        result = []
        for i in range(len(d1)):
            for j in range(len(d2)):
                result.append(d1[i] * m2[j] + d2[j] * m1[i] + d1[i] * d2[j])
        return result

    def validate_against_schema(self, ds):
        self.columns1.validate_against_schema(ds)
        if self.columns2 is not None:
            self.columns2.validate_against_schema(ds)


class HistogramQuery(DPQueryMixin):
    tp: Literal["histogram"]
    columns: ColumnsQuery = Field(discriminator="tp")
    bins: int

    def get_arguments(self) -> list[DPQueryMixin]:
        return [self.columns]

    def shape(self, ds) -> tuple[int, int]:
        return (self.bins, 1)

    def max_changed_rows(self, ds) -> int:
        return [self.columns.max_changed_rows(ds) * 2]

    def max_over_columns(self, ds):
        return [self.columns.len(ds)]

    def min_over_columns(self, ds):
        return [0]

    def max_norm_over_columns(self, ds, norm: str) -> list[float]:
        return [self.columns.len()]

    def sensitivity_over_columns(self, ds, norm: str) -> list[float]:
        max_changed = self.columns.max_changed_rows(ds)
        if norm == "l1":
            return [2 * max_changed]
        elif norm == "l2":
            return [(2 * max_changed) ** 0.5]
        else:
            raise DPValueError(f"Unsupported norm: {norm}")

    def validate_against_schema(self, ds):
        self.columns.validate_against_schema(ds)
        if self.columns.n_cols(ds) != 1:
            raise DPValueError(
                "Histograms are supported only for single columned queries"
            )


OneOfQueries = ColumnsQuery | SumQuery | MeanQuery | HistogramQuery | CovarianceQuery
