import pandas as pd

from dpch.client.client.interface import ClientMixin
from dpch.client.dataframe import DataFrame
from dpch.client.schema import SchemaCache


class Dataset:
    def __init__(self, client: ClientMixin, dataset_name: str, cache_ttl_seconds: int):
        self.client = client
        self.schema_cache = SchemaCache(client, ttl_seconds=cache_ttl_seconds)
        self.dataset_name = dataset_name

    @property
    def dataset(self):
        return self.schema_cache.get_schema()

    def preview(self) -> pd.DataFrame:
        ds = self.dataset
        rows = []
        for df in ds.dataframes:
            col_str = ", ".join(col.name for col in df.columns)
            if len(col_str) > 50:
                col_str = col_str[:47] + "..."
            rows.append(
                {
                    "dataframe": df.name,
                    "n_rows": df.n_rows,
                    "n_columns": len(df.columns),
                    "max_changed_rows": df.max_changed_rows,
                    "columns": col_str,
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
        return DataFrame(
            client=self.client,
            schema_cache=self.schema_cache,
            dataframe=df.name,
            query=None,
        )

    def __repr__(self) -> str:
        return f"Dataset: {self.dataset.name}\n{self.preview().to_string()}"

    def _repr_html_(self) -> str:
        return f"<b>Dataset: {self.dataset.name}</b>" + self.preview()._repr_html_()
