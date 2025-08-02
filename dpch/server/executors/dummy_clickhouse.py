# This is dummy (that is unoptimized) executor for ClickHouse provided just as an example implementation.
# It's possible to add executors for other OLAP databases e.g. YTSaurus, Apache Hive etc,
# as well as optimized executor for ClickHouse.

import re
from dataclasses import asdict
from functools import singledispatch
from secrets import token_hex
from typing import Any

from numpydantic import NDArray, Shape
import numpy as np
from clickhouse_connect.driver.asyncclient import AsyncClient

from dpch.common.queries.interface import QueryProto
from dpch.common.queries.queries import (
    ColumnsQuery,
    CovarianceQuery,
    HistogramQuery,
    MeanQuery,
    SumQuery,
)
from dpch.common.schema import Schema, SchemaDataset
from dpch.server.auth.interface import ServerSession
from dpch.server.clickhouse import get_clickhouse_client as get_clickhouse_client_impl
from dpch.server.config import ClickHouseConfig
from dpch.server.executors.interface import ExecutorMixin, ExecutorValueError
from dpch.server.executors.sql import COLUMN_NAME_REGEX, SQLQuery, SQLQueryMixin
from dpch.server.utils import asynconce


@singledispatch
def processor_factory(query: QueryProto, ds: SchemaDataset) -> SQLQueryMixin:
    raise ExecutorValueError("Query not supported by Dummy ClickHouse executor")


# Please write the evaluate method. It should:
# 1. Get query arguments which are also queries.
# 2. Evaluate runs query against real clickhouse client.
# You have to validate and call get_sql for query arguments recursively,
# but don't make evaluate recursive rather add separate functions for validating and getting sql which are recursive.
class DummyClickHouseExecutor(ExecutorMixin):
    def __init__(self, clickhouse: Any):
        self.clickhouse_config = ClickHouseConfig.model_validate(clickhouse)

    @asynconce
    async def get_clickhouse_client(self) -> AsyncClient:
        return await get_clickhouse_client_impl(self.clickhouse_config)

    async def execute(
        self, query: QueryProto, schema: Schema, session: ServerSession
    ) -> tuple[NDArray[Shape["*, *"], np.float64], Any]: # noqa: F722
        def get_sql_recursive(q: QueryProto) -> SQLQuery:
            processor = processor_factory(q, schema.dataset_from_session(session))
            computed_args = [get_sql_recursive(arg) for arg in q.get_arguments()]
            return processor.get_sql(computed_args)

        # Then get SQL for query and all arguments recursively
        sql = get_sql_recursive(query)
        try:
            # Execute against ClickHouse client
            client = await self.get_clickhouse_client()
            result = await client.query(sql.query, parameters=sql.parameters)
        except Exception as e:
            raise ExecutorValueError("Failed executing query against ClickHouse") from e
        data = result.result_rows
        data = np.array(data)[:, 1:].astype(np.float64)  # drop id column
        return data, {"sql": asdict(sql)}


class ColumnsQueryProcessor(SQLQueryMixin):
    # Please make this query support selection from several columns. Args is not used. Get column names from query, validate them against re, if invalid throw ExecutorError.
    # Then generate names for return values and format sql query.
    # Please generate parameter name for table, use this parameter in query.
    # Set query.dataframe as parameter value
    def get_sql(self, args: list[SQLQuery]) -> SQLQuery:
        # Generate return value names for each column
        rv_names = {col: f"rv_{token_hex(5)}" for col in self.query.columns}

        # Validate column names against regex to prevent SQL injection
        for column in self.query.columns:
            if re.match(COLUMN_NAME_REGEX, column) is None:
                raise ExecutorValueError(
                    f"Invalid SQL column name: {column}, possible SQL injection attempt"
                )

        # Build the SELECT clause with all columns
        select_parts = [f"{col} AS {rv}" for col, rv in rv_names.items()]
        select_clause = ", ".join(["id"] + select_parts)

        # Create a parameter name for the table and use query.dataframe as its value
        table_param = "table_" + token_hex(5)

        return SQLQuery(
            query=f"SELECT {select_clause} FROM %({table_param})s",
            parameters={table_param: self.query.dataframe},
            rv_names=list(rv_names.values()),
        )


@processor_factory.register
def _(query: ColumnsQuery, ds: SchemaDataset) -> ColumnsQueryProcessor:
    return ColumnsQueryProcessor(query, ds)


# Please move query execution logic into dummy_clickhouse file
# (see how ColumnsQueryProcessor is implemented).


class SumQueryProcessor(SQLQueryMixin):
    def get_sql(self, args: list[SQLQuery]) -> SQLQuery:
        inner_sql = args[0]
        rv_names = [f"rv_{token_hex(5)}" for _ in inner_sql.rv_names]
        select_parts = [
            f"SUM({col}) AS {rv}" for col, rv in zip(inner_sql.rv_names, rv_names)
        ]
        select_clause = ", ".join(["1 as id"] + select_parts)
        query_str = f"SELECT {select_clause} FROM ({inner_sql.query})"
        return SQLQuery(
            query=query_str, parameters=inner_sql.parameters, rv_names=rv_names
        )


@processor_factory.register
def _(query: SumQuery, ds: SchemaDataset) -> SumQueryProcessor:
    return SumQueryProcessor(query, ds)


class MeanQueryProcessor(SQLQueryMixin):
    def get_sql(self, args: list[SQLQuery]) -> SQLQuery:
        inner_sql = args[0]
        p_size = f"size_{token_hex(5)}"
        rv_names = [f"rv_{token_hex(5)}" for _ in inner_sql.rv_names]
        select_parts = [
            f"SUM({col}) / %({p_size})s AS {rv}"
            for col, rv in zip(inner_sql.rv_names, rv_names)
        ]
        select_clause = ", ".join(["1 as id"] + select_parts)
        query_str = f"SELECT {select_clause} FROM ({inner_sql.query})"
        return SQLQuery(
            query=query_str,
            parameters=inner_sql.parameters | {p_size: self.query.columns.len(self.ds)},
            rv_names=rv_names,
        )


@processor_factory.register
def _(query: MeanQuery, ds: SchemaDataset) -> MeanQueryProcessor:
    return MeanQueryProcessor(query, ds)


class CovarianceQueryProcessor(SQLQueryMixin):
    def get_sql(self, args: list[SQLQuery]) -> SQLQuery:
        sql1 = args[0]
        # If column2 is not set, use column1 for self-covariance
        if self.query.columns2 is None:
            rv_names = [
                f"rv_{token_hex(5)}"
                for _ in range(len(sql1.rv_names) * len(sql1.rv_names))
            ]

            # Build self-covariance calculations
            covar_parts = []
            for i, col1 in enumerate(sql1.rv_names):
                for j, col2 in enumerate(sql1.rv_names):
                    rv = rv_names[i * len(sql1.rv_names) + j]
                    covar_parts.append(f"AVG({col1} * {col2}) AS {rv}")

            covar_clause = ", ".join(["1 as id"] + covar_parts)
            query_str = f"SELECT {covar_clause} FROM ({sql1.query})"

            return SQLQuery(
                query=query_str, parameters=sql1.parameters, rv_names=rv_names
            )
        else:
            sql2 = args[1]
            rv_names = [
                f"rv_{token_hex(5)}"
                for _ in range(len(sql1.rv_names) * len(sql2.rv_names))
            ]

            # Build covariance calculations for each pair of columns
            covar_parts = []
            for i, col1 in enumerate(sql1.rv_names):
                for j, col2 in enumerate(sql2.rv_names):
                    rv = rv_names[i * len(sql2.rv_names) + j]
                    covar_parts.append(f"AVG({col1} * {col2}) AS {rv}")

            covar_clause = ", ".join(["1 as id"] + covar_parts)
            query_str = f"""
            SELECT {covar_clause}
            FROM ({sql1.query}) AS t1
            INNER JOIN ({sql2.query}) AS t2
            USING id
            """

            return SQLQuery(
                query=query_str,
                parameters=sql1.parameters | sql2.parameters,
                rv_names=rv_names,
            )


@processor_factory.register
def _(query: CovarianceQuery, ds: SchemaDataset) -> CovarianceQueryProcessor:
    return CovarianceQueryProcessor(query, ds)


class HistogramQueryProcessor(SQLQueryMixin):
    def get_sql(self, args: list[SQLQuery]) -> SQLQuery:
        col_query = args[0]
        p_bins = f"bins_{token_hex(5)}"
        p_interval = f"interval_{token_hex(5)}"
        p_min = f"min_{token_hex(5)}"
        p_max = f"max_{token_hex(5)}"
        rv = f"rv_{token_hex(5)}"
        query_str = f"""
        SELECT
            id, 
            count(*) AS {rv}
        FROM (
            SELECT
                toInt32(({col_query.rv_names[0]} - %({p_min})s) / %({p_interval})s) AS id
            FROM ({col_query.query}) WHERE ({col_query.rv_names[0]} >= %({p_min})s AND {col_query.rv_names[0]} <= %({p_max})s)
        )
        GROUP BY id
        ORDER BY id WITH FILL FROM 0 TO %({p_bins})s
        """
        min_val = self.query.min_over_columns(self.ds)[0]
        max_val = self.query.max_over_columns(self.ds)[0]
        parameters = col_query.parameters | {
            p_interval: (max_val - min_val)
            / self.query.bins,
            p_min: min_val,
            p_max: max_val,
            p_bins: self.query.bins,
        }
        return SQLQuery(
            query=query_str,
            parameters=parameters,
            rv_names=[rv],
        )


@processor_factory.register
def _(query: HistogramQuery, ds: SchemaDataset) -> HistogramQueryProcessor:
    return HistogramQueryProcessor(query, ds)
