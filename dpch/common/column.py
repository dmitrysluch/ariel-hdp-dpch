from pydantic import BaseModel
from typing import Literal
from dpch.common.interface import SQLMixin, SQLQuery, SensitivityMixin, BoundedMixin
from secrets import token_hex


# Please write column pydantic model. It should contain name, minimum and maximum value (of each item), size,
# and have method compute sensitivity which takes norm as string (l1 or l2) and returns the maximum difference in l1 or l2 norm corrispondingly,
# if a single item of the column changes
class Column(BaseModel, SQLMixin, SensitivityMixin, BoundedMixin):
    tp: Literal["column"]
    name: str
    min_val: float
    max_val: float
    size: int

    def get_size(self) -> int:
        return self.size

    def max_norm(self, norm: str) -> float:
        return max(self.min_val, self.max_val)

    def compute_sensitivity(self, norm: str) -> float:
        if norm not in ["l1", "l2"]:
            raise ValueError("Norm must be either 'l1' or 'l2'")
        return abs(self.max_val - self.min_val)

    def get_sql(self):
        p_name = f"name_{token_hex(5)}"
        rv = f"rv_{token_hex(5)}"
        # TODO: (think about SQL injection here)
        return SQLQuery(
            query=f"SELECT id, {self.name} AS {rv} FROM %(table)s",
            parameters={p_name: self.name},
            rv_name=rv,
        )

    def validate_columns(self, columns: list["Column"]) -> bool:
        return self in columns
