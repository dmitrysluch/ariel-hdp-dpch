# This is dummy (that is unoptimized) executor for ClickHouse provided just as an example implementation.
# It's possible to add executors for other OLAP databases e.g. YTSaurus, Apache Hive etc,
# as well as optimized executor for ClickHouse.

import re
from dataclasses import asdict
from secrets import token_hex
from typing import Any, Literal

import numpy as np
from clickhouse_connect.driver.asyncclient import AsyncClient

from dpch.common.queries.interface import QueryProto
from dpch.common.queries.queries import (
    ColumnsQuery,
    Covariance,
    Histogram,
    MeanQuery,
    SumQuery,
)
from dpch.common.schema import Schema, SchemaDataset
from dpch.server.auth.interface import ServerSession
from dpch.server.clickhouse import get_clickhouse_client as get_clickhouse_client_impl
from dpch.server.config import ClickHouseConfig
from dpch.server.executors.interface import ExecutorMixin, ExecutorValueError
from dpch.server.executors.registry import QueryProcessorRegistry
from dpch.server.executors.sql import COLUMN_NAME_REGEX, SQLQuery, SQLQueryMixin
from dpch.server.utils import once

reg = QueryProcessorRegistry[SQLQueryMixin]()


# Please write the evaluate method. It should:
# 1. Get query arguments which are also queries.
# 2. Recursively validate, then transform argument queries to sql.
# 3. Then evaluate runs query against real clickhouse client.
# You have to validate and call get_sql for query arguments recursively,
# but don't make evaluate recursive rather add separate functions for validating and getting sql which are recursive.
class DummyClickhouseExecutor(ExecutorMixin):
    def __init__(self, clickhouse: Any):
        self.clickhouse_config = ClickHouseConfig.model_validate(clickhouse)

    @once
    def get_clickhouse_client(self) -> AsyncClient:
        return get_clickhouse_client_impl(self.clickhouse_config)

    async def execute(self, query: QueryProto, schema: Schema, session: ServerSession):
        def get_sql_recursive(q: QueryProto) -> SQLQuery:
            processor = reg.parse_query_processor(q)
            computed_args = [get_sql_recursive(arg) for arg in q.get_arguments()]
            return processor.get_sql(q, computed_args)

        # Then get SQL for query and all arguments recursively
        sql = get_sql_recursive(query)
        # Execute against ClickHouse client
        client = self.get_clickhouse_client()
        try:
            result = await client.query(sql.query, parameters=sql.parameters)
        except Exception:
            pass
        data = result.result_rows
        data = np.array(data)[:, 1:]  # drop id column
        return data, {"sql": asdict(sql)}


@reg.register_query_processor
class ColumnsQueryProcessor(SQLQueryMixin):
    tp: Literal["columns"]

    # Please make this query support selection from several columns. Args is not used. Get column names from query, validate them against re, if invalid throw ExecutorError.
    # Then generate names for return values and format sql query.
    # Please generate parameter name for table, use this parameter in query.
    # Set query.dataframe as parameter value
    def get_sql(self, query: ColumnsQuery, args: list[SQLQuery]) -> SQLQuery:
        # Generate return value names for each column
        rv_names = {col: f"rv_{token_hex(5)}" for col in query.columns}

        # Validate column names against regex to prevent SQL injection
        for column in query.columns:
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
            parameters={table_param: query.dataframe},
            rv_names=list(rv_names.values()),
        )

    def validate_against_schema(
        self, query: ColumnsQuery, schema: SchemaDataset
    ) -> bool:
        if query.dataset != schema.name:
            return False
        for df in schema.dataframes:
            if df == query.dataframe:
                query_cols = set(query.columns)
                all_cols = set(col.name for col in df.columns)
                return len(query_cols & all_cols) == len(query_cols)
        return False


# Please move query execution logic into dummy_clickhouse file
# (see how ColumnsQueryProcessor is implemented).
# For validate_against_schema all queries (except ColumnsQueryProcessor)
# must always return True.
# For get_sql move implementation from dpch.common.queries


@reg.register_query_processor
class SumQueryProcessor(SQLQueryMixin):
    tp: Literal["sum"]

    # Please make the query to support single argument returning several columns.
    # args[0] rv_names will be names of the columns returned
    def get_sql(self, query: SumQuery, args: list[SQLQuery]) -> SQLQuery:
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

    def validate_against_schema(self, query: SumQuery, schema: SchemaDataset) -> bool:
        return True


@reg.register_query_processor
class MeanQueryProcessor(SQLQueryMixin):
    tp: Literal["mean"]

    # Please make query to support argument returning several columns as in sum query above
    def get_sql(self, query: MeanQuery, args: list[SQLQuery]) -> SQLQuery:
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
            parameters=inner_sql.parameters | {p_size: query.columns.get_size()},
            rv_names=rv_names,
        )

    def validate_against_schema(self, query: MeanQuery, schema: SchemaDataset) -> bool:
        return True


@reg.register_query_processor
class CovarianceQueryProcessor(SQLQueryMixin):
    tp: Literal["covariance"]

    # Please make query to support argument returning several columns.
    # Also please make it possible that query has only column1 set,
    # in that case self covariance is computed
    def get_sql(self, query: Covariance, args: list[SQLQuery]) -> SQLQuery:
        sql1 = args[0]
        # If column2 is not set, use column1 for self-covariance
        if query.columns2 is None:
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

    def validate_against_schema(self, query: Covariance, schema: SchemaDataset) -> bool:
        return True


@reg.register_query_processor
class HistogramQueryProcessor(SQLQueryMixin):
    tp: Literal["histogram"]

    def get_sql(self, query: Histogram, args: list[SQLQuery]) -> SQLQuery:
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
        parameters = col_query.parameters | {
            p_interval: (query.columns.max_val - query.columns.min_val) / query.bins,
            p_min: query.columns.min_val,
            p_max: query.columns.max_val,
            p_bins: query.bins,
        }
        return SQLQuery(
            query=query_str,
            parameters=parameters,
            rv_names=[rv],
        )

    def validate_against_schema(self, query: Histogram, schema: SchemaDataset) -> bool:
        return True
