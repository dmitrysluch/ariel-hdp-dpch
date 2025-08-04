import dataclasses
from functools import lru_cache
from typing import Self

import pandas as pd

from dpch.client.cache import Cache
from dpch.client.client.interface import ClientMixin
from dpch.common.api import DebugQueryResponse, QueryRequest, RunQueryResponse
from dpch.common.queries.interface import DPValueError
from dpch.common.queries.queries import (
    ColumnsQuery,
    CovarianceQuery,
    HistogramQuery,
    ListSelector,
    MeanQuery,
    OneOfQueries,
    SliceSelector,
    SumQuery,
)

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


@dataclasses.dataclass(repr=False, frozen=True, kw_only=True)
class BaseDataframe:
    dataframe: str | None
    query: OneOfQueries | None

    def __getitem__(self, index: str | list[str] | tuple | slice) -> Self:
        if self.query is not None:
            raise NotImplementedError("Query is already set")
        if isinstance(index, str):
            columns = ListSelector(tp="list", items=(index,))
        elif isinstance(index, slice):
            columns = SliceSelector(
                tp="slice", start=index.start, stop=index.stop, step=index.step
            )
        elif isinstance(index, (list, tuple)):
            columns = ListSelector(tp="list", items=tuple(index))
        else:
            raise TypeError("Index must be a string, slice, or list/tuple of strings")
        query = ColumnsQuery(tp="columns", columns=columns, dataframe=self.dataframe)
        return dataclasses.replace(self, query=query)

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
        return ColumnsQuery(
            tp="columns", columns=SliceSelector(tp="slice"), dataframe=self.dataframe
        )

    def sum(self) -> Self:
        query = SumQuery(tp="sum", columns=self.effective_query)
        return dataclasses.replace(self, query=query)

    def mean(self) -> Self:
        query = MeanQuery(tp="mean", columns=self.effective_query)
        return dataclasses.replace(self, query=query)

    def histogram(self, bins: int) -> Self:
        query = HistogramQuery(tp="histogram", columns=self.effective_query, bins=bins)
        return dataclasses.replace(self, query=query)

    def covariance(self, columns2: Self | None = None) -> Self:
        columns2_query = columns2.effective_query if columns2 is not None else None
        query = CovarianceQuery(
            tp="covariance", columns1=self.effective_query, columns2=columns2_query
        )
        return dataclasses.replace(self, query=query)


@dataclasses.dataclass(repr=False, frozen=True, kw_only=True)
class DataFrame(BaseDataframe):
    client: ClientMixin
    cache: Cache

    # Lets fix preview method implementation.
    # It has to return dataframe with index
    # equal to effective query representation.
    # Return all the fields relevant to the column:
    # l1, l2 sensitivity and max_norms, min, max, maximum_rows_changed, etc
    def preview(self) -> pd.DataFrame:
        ds = self.cache.schema()
        q = self.effective_query

        q.validate_against_schema(ds)

        # Get all relevant column information
        col_names = q.repr_columns(ds)
        min_vals = q.min_over_columns(ds)
        max_vals = q.max_over_columns(ds)
        l1_sens = q.sensitivity_over_columns(ds, "l1")
        l2_sens = q.sensitivity_over_columns(ds, "l2")
        l1_max_norms = q.max_norm_over_columns(ds, "l1")
        l2_max_norms = q.max_norm_over_columns(ds, "l2")
        max_changed_rows = q.max_changed_rows(ds)

        df = pd.DataFrame(
            {
                "min": min_vals,
                "max": max_vals,
                "l1_sensitivity": l1_sens,
                "l2_sensitivity": l2_sens,
                "l1_max_norm": l1_max_norms,
                "l2_max_norm": l2_max_norms,
                "max_changed_rows": max_changed_rows,
            },
            index=col_names,
        )
        return df

    def run(self) -> "QueryResultWrapper":
        ds = self.cache.schema()
        if self.query is None:
            raise ValueError("No query set for run")
        self.effective_query.validate_against_schema(ds)
        req = QueryRequest(query=self.effective_query)
        resp = self.client.query(req)
        return QueryResultWrapper(resp)

    def __repr__(self) -> str:
        try:
            return self.preview().__repr__()
        except DPValueError as e:
            return f"Invalid query: {e}"

    def _repr_html_(self) -> str:
        try:
            return self.preview()._repr_html_()
        except DPValueError as e:
            return f'<h4 style="color: #d53">Invalid query: {e}</h4>'


class QueryResultWrapper:
    def __init__(self, resp: RunQueryResponse | DebugQueryResponse):
        self.resp = resp

    @property
    def result(self) -> pd.DataFrame:
        return pd.DataFrame(self.resp.result, columns=self.resp.column_names)

    @property
    def new_session(self) -> pd.DataFrame:
        return pd.DataFrame(self.resp.new_session)

    @property
    def raw_result(self) -> pd.DataFrame:
        if hasattr(self.resp, "raw_result") and self.resp.raw_result is not None:
            return pd.DataFrame(self.resp.raw_result, columns=self.resp.column_names)
        raise AttributeError("No raw_result in response")

    @property
    def sensitivity(self) -> pd.DataFrame:
        if hasattr(self.resp, "sensitivity") and self.resp.sensitivity is not None:
            return pd.DataFrame(self.resp.sensitivity, index=self.resp.column_names)
        raise AttributeError("No sensitivity in response")

    def __repr__(self) -> str:
        s = ""
        s += f"Result:\n{self.result}"
        # New session
        s += f"\nNew session:\n{self.new_session}"
        # Raw result
        if hasattr(self.resp, "raw_result") and self.resp.raw_result is not None:
            s += f"\nRaw Result:\n{self.raw_result}"
        # Sensitivity
        if hasattr(self.resp, "sensitivity") and self.resp.sensitivity is not None:
            s += f"\nSensitivity:\n{self.sensitivity}"
        return s

    def _repr_html_(self) -> str:
        html = ""
        html += "<h3>Result</h3>" + self.result._repr_html_()
        if hasattr(self.resp, "raw_result") and self.resp.raw_result is not None:
            html += "<h3>Raw Result</h3>" + self.raw_result._repr_html_()
        if hasattr(self.resp, "sensitivity") and self.resp.sensitivity is not None:
            html += "<h3>Sensitivity</h3>" + self.sensitivity.transpose()._repr_html_()
        html += "<h3>New session</h3>" + self.new_session._repr_html_()
        return html
