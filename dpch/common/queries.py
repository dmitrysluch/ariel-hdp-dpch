from secrets import token_hex
from pydantic import BaseModel, field_validator
from typing import List, Literal
from dpch.common.interface import (
    SQLQuery,
    FunctionSensitivityMixin,
    SQLMixin,
    SensitivityMixin,
    BoundedMixin,
)
from dpch.common.column import Column

# Please write an implementation of sum query (it is a plain sum over all rows).
# Use summation lipschitzness to implement FunctionSensitivityMixin
# The class has a single field which is a column and it implements both SQLMixin and SensitivityMixin. It is used both in get_arguments and get_sql


class SumQuery(BaseModel, SQLMixin, FunctionSensitivityMixin):
    tp: Literal["sum"]
    column: "OneOfQueries"

    def get_size(self) -> int:
        return 1

    def get_arguments(self) -> List["OneOfQueries"]:
        return [self.column]

    def get_lipschitz_parameter(self, norm: str) -> float:
        # Summation is 1-Lipschitz under both L1 and L2 norms
        if norm in ("l1", "l2"):
            return 1.0
        raise ValueError(f"Unsupported norm: {norm}")

    def get_sql(self) -> SQLQuery:
        inner_sql = self.column.get_sql()
        rv = f"rv_{token_hex(5)}"
        query = f"SELECT 1 as id, SUM ({inner_sql.rv_name}) AS {rv} FROM ({inner_sql.query})"
        return SQLQuery(query=query, parameters=inner_sql.parameters, rv_name=rv)

    def validate_columns(self, columns: list["Column"]) -> bool:
        return self.column.validate_columns(columns)


class MeanQuery(BaseModel, SQLMixin, FunctionSensitivityMixin):
    tp: Literal["mean"]
    column: "OneOfQueries"

    def get_size(self) -> int:
        return 1

    def get_arguments(self) -> List["OneOfQueries"]:
        return [self.column]

    def get_lipschitz_parameter(self, norm: str) -> float:
        # Summation is 1-Lipschitz under both L1 and L2 norms
        if norm in ("l1", "l2"):
            return 1 / self.column.get_size()
        raise ValueError(f"Unsupported norm: {norm}")

    def get_sql(self) -> SQLQuery:
        inner_sql = self.column.get_sql()
        p_size = f"size_{token_hex(5)}"
        rv = f"rv_{token_hex(5)}"
        query = f"SELECT 1 as id, SUM ({inner_sql.rv_name}) / %({p_size})s AS {rv} FROM ({inner_sql.query})"
        return SQLQuery(
            query=query,
            parameters=inner_sql.parameters | {p_size: self.column.get_size()},
            rv_name=rv,
        )

    def validate_columns(self, columns: list["Column"]) -> bool:
        return self.column.validate_columns(columns)


# Please write Covariance class. It has column1 and column2 as arguments. Add pydantic validator that checks that these columns satisfy BoundedMixin.
# Covariance class must satisfy BaseModel, SQLMixin and SensitivityMixin.
# For sensitivity notice that if l2 sensitivity of first column is d1, second d2, l2 norms of maximum values are m1 and m2 then resulting sensitivity is d1m2 + d2m1 + d1d2 by Cauchy-Schwartz
class Covariance(BaseModel, SQLMixin, SensitivityMixin):
    tp: Literal["covariance"]
    column1: "OneOfQueries"
    column2: "OneOfQueries"

    @field_validator("column1", "column2")
    @classmethod
    def check_bounded(cls, v):
        if not isinstance(v, BoundedMixin):
            raise ValueError("Columns must satisfy BoundedMixin")
        return v

    def get_size(self) -> int:
        return 1

    def compute_sensitivity(self, norm: str) -> float:
        if norm not in ["l1", "l2"]:
            raise ValueError("Norm must be either 'l1' or 'l2'")

        # We always use L2 norms because of Cauchy-Schwartz. We could have used L1 norms as well, but them would simply yield worser bound.
        d1 = self.column1.compute_sensitivity("l2")
        d2 = self.column2.compute_sensitivity("l2")
        m1 = self.column1.max_norm("l2")
        m2 = self.column1.max_norm("l2")

        return d1 * m2 + d2 * m1 + d1 * d2

    def get_sql(self) -> SQLQuery:
        sql1 = self.column1.get_sql()
        sql2 = self.column2.get_sql()
        rv = f"rv_{token_hex(5)}"
        query = f"""
        SELECT 1 as id, AVG({sql1.rv_name} * {sql2.rv_name}) AS {rv}
        FROM ({sql1.query}) AS t1
        INNER JOIN ({sql2.query}) AS t2
        USING id
        """
        return SQLQuery(
            query=query, parameters=sql1.parameters | sql2.parameters, rv_name=rv
        )

    def validate_columns(self, columns: list["Column"]) -> bool:
        return self.column1.validate_columns(columns) and self.column2.validate_columns(
            columns
        )


# Please write a Histagram class. It takes Column as input (not arbitrary query). Implement compute_sensitivity and get_sql methods.
# For l1 norm sensitivity change is 2 * abs(max_val - min_val). For l2 norm the multiplier is sqrt(2)


class Histogram(BaseModel, SQLMixin, SensitivityMixin):
    tp: Literal["histogram"]
    column: Column
    bins: int

    def get_size(self) -> int:
        return self.bins

    def compute_sensitivity(self, norm: str) -> float:
        if norm not in ["l1", "l2"]:
            raise ValueError("Norm must be either 'l1' or 'l2'")
        return 2 if norm == "l1" else (2**0.5)

    def get_sql(self):
        col_query = self.column.get_sql()
        p_bins = f"bins_{token_hex(5)}"
        p_interval = f"interval_{token_hex(5)}"
        p_min = f"min_{token_hex(5)}"
        p_max = f"max_{token_hex(5)}"
        rv = f"rv_{token_hex(5)}"
        return SQLQuery(
            query=f"""
SELECT
    id, 
    count(*) AS {rv}
FROM (
    SELECT
        toInt32(({col_query.rv_name} - %({p_min})s) / %({p_interval})s) AS id
    FROM ({col_query.query}) WHERE ({col_query.rv_name} >= %({p_min})s AND {col_query.rv_name} <= %({p_max})s)
)
GROUP BY id
ORDER BY id WITH FILL FROM 0 TO %({p_bins})s
""",
            parameters=col_query.parameters
            | {
                p_interval: (self.column.max_val - self.column.min_val) / self.bins,
                p_min: self.column.min_val,
                p_max: self.column.max_val,
                p_bins: self.bins,
            },
            rv_name=rv,
        )


OneOfQueries = SumQuery | MeanQuery | Column | Histogram | Covariance
