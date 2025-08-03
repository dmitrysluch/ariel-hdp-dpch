import pandas as pd

from dpch.client.client.interface import ClientMixin
from dpch.client.dataframe import DataFrame
from dpch.client.schema import SchemaCache


class Dataset:
    def __init__(self, client: ClientMixin, dataset_name: str, cache_ttl_seconds: int):
        self.client = client
        self.schema_cache = SchemaCache(client, cache_ttl_seconds=cache_ttl_seconds)
        self.dataset_name = dataset_name

    @property
    def schema(self):
        return self.schema_cache.get_schema()

    @property
    def dataset(self):
        # Use the dataset with the specified name
        ds = next(
            (ds for ds in self.schema.datasets if ds.name == self.dataset_name), None
        )
        if ds is None:
            raise ValueError(f"Dataset '{self.dataset_name}' not found in schema")
        return ds

    def preview(self) -> pd.DataFrame:
        ds = self.dataset
        rows = []
        for df in ds.dataframes:
            rows.append(
                {
                    "dataframe": df.name,
                    "n_rows": df.n_rows,
                    "n_columns": len(df.columns),
                    "max_changed_rows": df.max_changed_rows,
                    "columns": ", ".join(col.name for col in df.columns),
                    "min_vals": [col.min_val for col in df.columns],
                    "max_vals": [col.max_val for col in df.columns],
                    "max_l1_norms": [col.max_l1_norm for col in df.columns],
                    "max_l2_norms": [col.max_l2_norm for col in df.columns],
                }
            )
        return pd.DataFrame(rows)

    def __getitem__(self, dataframe_name: str) -> DataFrame:
        # Validate dataframe exists
        df = next(
            (df for df in self.dataset.dataframes if df.name == dataframe_name), None
        )
        if df is None:
            raise KeyError(
                f"Dataframe '{dataframe_name}' not found in dataset '{self.dataset.name}'"
            )
        return DataFrame(self.client, self.schema_cache, dataframe_name)

    def __repr__(self) -> str:
        return f"Dataset: {self.dataset.name}\n{self.preview().to_string()}"

    def _repr_html_(self) -> str:
        return f"<b>Dataset: {self.dataset.name}</b>" + self.preview()._repr_html_()
