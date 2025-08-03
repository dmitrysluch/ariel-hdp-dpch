from functools import lru_cache
from typing import Self

import pandas as pd
from common.queries.queries import (
    ColumnsQuery,
    CovarianceQuery,
    HistogramQuery,
    MeanQuery,
    OneOfQueries,
    SumQuery,
)

from dpch.client.client.interface import ClientMixin
from dpch.client.schema import SchemaCache
from dpch.common.api import DebugQueryResponse, QueryRequest, RunQueryResponse

# Please implement BaseDataframe. Its basically a builder.
# It has str field dataframe and OneOfQueries query. On creation either may be set.
# Indexing operator is implemented which sets query to be columns query with specified dataset and columns.
# Index may be either string or list of strings. If Query is already set throw NonImplementedError.
# Also add methods for each query except ColumnsQuery which updates query and returns self.
# Please update BaseDataframe inplace

# Please add typings where they are missing,
# and in covariance query take dataframe as a second argument, not query.

# Here exists an error if no base query exists either in self or columns2 in covariance.
# Lets fix this by adding property which returns query if it is set
# and query selecting whole dataframe (columns=":") otherwise

# Lets use effective_query is all methods


class BaseDataframe:
    dataframe: str | None
    query: OneOfQueries | None

    def __init__(self, dataframe: str | None = None, query: OneOfQueries | None = None):
        self.dataframe = dataframe
        self.query = query

    def __getitem__(self, index: str | list[str] | tuple | slice) -> Self:
        if self.query is not None:
            raise NotImplementedError("Query is already set")
        if isinstance(index, str):
            if ":" in index:
                raise ValueError(
                    "String index must not contain ':' (use a slice object instead)"
                )
            columns = (index,)
        elif isinstance(index, slice):
            # Convert slice to string representation: start:end:step
            start = index.start or ""
            end = index.stop or ""
            step = "" if index.step is None else str(index.step)
            if step:
                columns = (f"{start}:{end}:{step}",)
            else:
                columns = (f"{start}:{end}",)
        elif isinstance(index, (list, tuple)):
            columns = tuple(index)
        else:
            raise TypeError("Index must be a string, slice, or list/tuple of strings")
        self.query = ColumnsQuery(
            tp="columns", columns=columns, dataframe=self.dataframe
        )
        return self

    @property
    @lru_cache
    def effective_query(self) -> OneOfQueries:
        """
        Returns the current query if set, otherwise a ColumnsQuery selecting all columns (":") for the dataframe.
        """
        if self.query is not None:
            return self.query
        if self.dataframe is None:
            raise ValueError("No dataframe specified for default query")
        return ColumnsQuery(tp="columns", columns=":", dataframe=self.dataframe)

    def sum(self) -> Self:
        self.query = SumQuery(tp="sum", columns=self.effective_query)
        return self

    def mean(self) -> Self:
        self.query = MeanQuery(tp="mean", columns=self.effective_query)
        return self

    def histogram(self, bins: int) -> Self:
        self.query = HistogramQuery(
            tp="histogram", columns=self.effective_query, bins=bins
        )
        return self

    def covariance(self, columns2: Self | None = None) -> Self:
        columns2_query = columns2.effective_query if columns2 is not None else None
        self.query = CovarianceQuery(
            tp="covariance", columns1=self.effective_query, columns2=columns2_query
        )
        return self


class DataFrame(BaseDataframe):
    def __init__(
        self,
        client: ClientMixin,
        schema_cache: SchemaCache,
        dataframe: str,
        query: OneOfQueries | None = None,
    ):
        super().__init__(dataframe=dataframe, query=query)
        self.client = client
        self.schema_cache = schema_cache

    def preview(self) -> pd.DataFrame:
        schema = self.schema_cache.get_schema()
        if self.query is None:
            raise ValueError("No query set for preview")
        ds = schema.dataset
        q = self.query
        min_vals = q.min_over_columns(ds)
        max_vals = q.max_over_columns(ds)
        sens = q.sensitivity_over_columns(ds, "l1")
        col_names = q.parse_columns(ds) if hasattr(q, "parse_columns") else []
        last_query = type(q).__name__
        df = pd.DataFrame(
            {
                "column": col_names,
                "min": min_vals,
                "max": max_vals,
                "sensitivity": sens,
                "query": [last_query] * len(col_names),
            }
        )
        return df

    def run(self, debug: bool = False) -> "QueryResultWrapper":
        schema = self.schema_cache.get_schema()
        if self.query is None:
            raise ValueError("No query set for run")
        req = QueryRequest(query=self.query, schema=schema)
        resp = self.client.query(req)
        return QueryResultWrapper(resp)

    def __repr__(self) -> str:
        try:
            preview_df = self.preview()
            return f"DataFrame preview:\n{preview_df.to_string()}"
        except Exception as e:
            return f"DataFrame (error in preview): {e}"

    def _repr_html_(self) -> str:
        try:
            preview_df = self.preview()
            return preview_df._repr_html_()
        except Exception as e:
            return f"<pre>DataFrame (error in preview): {e}</pre>"


class QueryResultWrapper:
    def __init__(self, resp: RunQueryResponse | DebugQueryResponse):
        self.resp = resp

    @property
    def result(self) -> pd.DataFrame:
        return pd.DataFrame(self.resp.result)

    @property
    def session(self) -> pd.DataFrame:
        return pd.DataFrame(self.resp.session)

    @property
    def raw_result(self) -> pd.DataFrame:
        if hasattr(self.resp, "raw_result") and self.resp.raw_result is not None:
            return pd.DataFrame(self.resp.raw_result)
        raise AttributeError("No raw_result in response")

    @property
    def sensitivity(self) -> pd.DataFrame:
        if hasattr(self.resp, "sensitivity") and self.resp.sensitivity is not None:
            return pd.DataFrame(self.resp.sensitivity)
        raise AttributeError("No sensitivity in response")

    def __repr__(self) -> str:
        s = f"Result:\n{self.result.to_string()}\nSession:\n{self.session.to_string()}"
        if hasattr(self.resp, "raw_result") and self.resp.raw_result is not None:
            s += f"\nRaw Result:\n{self.raw_result.to_string()}"
        if hasattr(self.resp, "sensitivity") and self.resp.sensitivity is not None:
            s += f"\nSensitivity:\n{self.sensitivity.to_string()}"
        return s

    def _repr_html_(self) -> str:
        html = self.result._repr_html_() + self.session._repr_html_()
        if hasattr(self.resp, "raw_result") and self.resp.raw_result is not None:
            html += self.raw_result._repr_html_()
        if hasattr(self.resp, "sensitivity") and self.resp.sensitivity is not None:
            html += self.sensitivity._repr_html_()
        return html
