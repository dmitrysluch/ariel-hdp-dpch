from pydantic import BaseModel, Field

from dpch.common.session import Session

"""
Defines the schema for differential privacy data handling.

This module provides schema definitions for managing data in a differential privacy context.
The schema ensures that sensitive data properties are properly bounded without exposing actual
data characteristics that could compromise privacy.

Note on min/max values and norms:
For differential privacy purposes, the min_val, max_val, and norm values in SchemaColumn
do not represent actual statistical properties of the data. Instead:
- min_val represents a reasonable lower bound that could contain the actual minimum
- max_val represents a reasonable upper bound that could contain the actual maximum  
- max_l1_norm/max_l2_norm represent reasonable upper bounds for the corresponding norms

This approach prevents leaking actual data characteristics while still providing
necessary bounds for differential privacy mechanisms.
"""


# Defines properties and bounds for a single data column.
class SchemaColumn(BaseModel):
    name: str
    min_val: float
    max_val: float
    max_l1_norm: float | None = Field(default=None)
    max_l2_norm: float | None = Field(default=None)

    class Config:
        frozen = True


# Represents a data table with its columns and privacy settings.
class SchemaDataFrame(BaseModel):
    name: str
    columns: tuple[SchemaColumn, ...]
    n_rows: int
    # The maximum amount of rows, changes in which must be invisible to the adversary.
    # Set to 0 if the dataframe contains no secret rows.
    max_changed_rows: int

    class Config:
        frozen = True


# Defines differential privacy parameters (epsilon, delta).
class DPConstraints(BaseModel):
    max_eps: float
    max_delta: float

    class Config:
        frozen = True


# Groups related dataframes under common privacy constraints.
class SchemaDataset(BaseModel):
    name: str
    constraints: DPConstraints
    dataframes: tuple[SchemaDataFrame, ...]

    class Config:
        frozen = True


# Maps user roles to dataset access permissions.
# Note that roles are managed and evaluated by auth plugin only,
# and plugin may ignore role bindings provided via schema.
class DatasetRoleBinding(BaseModel):
    name: str
    dataset: str
    role: str

    class Config:
        frozen = True


# Top-level container for datasets and access control.
class Schema(BaseModel):
    datasets: tuple[SchemaDataset, ...]
    role_bindings: tuple[DatasetRoleBinding, ...]

    def dataset_from_session(self, s: Session):
        ds = next((ds for ds in self.datasets if s.dataset == ds.name), None)
        if ds is None:
            raise ValueError("Session dataset not present in schema")
        return ds

    class Config:
        frozen = True
