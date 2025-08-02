#!/usr/bin/python3

import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).parent.parent))

import pandas as pd
from clickhouse_connect import get_client

from dpch.common.schema import (
    DatasetRoleBinding,
    DPConstraints,
    Schema,
    SchemaColumn,
    SchemaDataFrame,
    SchemaDataset,
)

columns: list[SchemaColumn] = []
df = pd.read_csv(pathlib.Path(__file__).parent / "dataset/vius_2021_puf.csv")
df.columns = pd.Series(df.columns).apply(lambda x: x.lower())


def add_schema_column(name):
    # Here min and max values leak, and in production applications the correct approach is
    # to provide reasonable but not tight bounds here.
    columns.append(
        SchemaColumn(
            name=name,
            min_val=df[name].min(),
            max_val=df[name].max(),
        )
    )


add_schema_column("tabweight")

# Please make OHE columns for regstate column
# Create one-hot encoded columns for regstate
regstate_dummies = pd.get_dummies(df["regstate"], prefix="regstate").astype("int")
columns.extend(
    [
        SchemaColumn(name=x, min_val=0, max_val=1, size=df.shape[0])
        for x in regstate_dummies.columns
    ]
)
df = pd.concat([df, regstate_dummies], axis=1)


def process_year(x):
    if x == "P99":
        return 1998
    elif x == "99":
        return 1999
    elif x == "21P":
        return 2021
    elif x == "Z":
        return None
    else:
        return int(f"20{x}")


df["acquireyear"] = df["acquireyear"].apply(process_year)
# Please implement filling of na with mean
df["acquireyear"] = df["acquireyear"].fillna(df["acquireyear"].mean())
add_schema_column("acquireyear")

df["acquisition"] = df["acquisition"].apply(lambda x: {1: 0, 2: 1, 3: None}[x])
df["acquisition"] = df["acquisition"].fillna(df["acquisition"].mean())
add_schema_column("acquisition")

# Please translate data in avgweight column to mean of the range via pandas apply. For X input value set it to NA


def process_avgweight(x):
    ranges = {
        "1": 3000,  # midpoint of 0-6000
        "2": 7250.5,  # midpoint of 6001-8500
        "3": 9250.5,  # midpoint of 8501-10000
        "4": 12000.5,  # midpoint of 10001-14000
        "5": 15000.5,  # midpoint of 14001-16000
        "6": 17750.5,  # midpoint of 16001-19500
        "7": 22750.5,  # midpoint of 19501-26000
        "8": 29500.5,  # midpoint of 26001-33000
        "9": 36500.5,  # midpoint of 33001-40000
        "10": 45000.5,  # midpoint of 40001-50000
        "11": 55000.5,  # midpoint of 50001-60000
        "12": 70000.5,  # midpoint of 60001-80000
        "13": 90000.5,  # midpoint of 80001-100000
        "14": 115000.5,  # midpoint of 100001-130000
        "15": 130001,  # minimum of 130001+
    }
    return ranges.get(x, None)


df["avgweight"] = df["avgweight"].apply(process_avgweight)
df["avgweight"] = df["avgweight"].fillna(df["avgweight"].mean())
add_schema_column("avgweight")

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].astype("str")

# Please insert resulting dataset in clickhouse table vius
# Could you please parse column dynamically from df

# Create ClickHouse client
client = get_client(
    host="localhost",
    port=8123,
    username="default",
    password="s3cur3",
    database="default",
)

# Generate column definitions dynamically
column_defs = []
for col in df.columns:
    # Check column type and assign appropriate ClickHouse type
    if df[col].dtype == "float64":
        col_type = "Float64"
    elif df[col].dtype == "float32":
        col_type = "Float32"
    elif df[col].dtype == "int64":
        col_type = "Int64"
    elif df[col].dtype == "uint8":
        col_type = "UInt8"
    else:
        col_type = "String"

    column_defs.append(f"{col} {col_type}")

# Generate two datasets differing in single step
df1 = df.drop(index=42)
df2 = df.drop(index=10424)

# Please transform code to create two tables vius1 and vius2 containing df1 and df2 correspondingly. Drop tables
# if they previously existed.
# Please use for loop
# Create dictionary to store dataframes
dfs = {"vius1": df1, "vius2": df2}

# Iterate through tables
for table_name, df_data in dfs.items():
    # Drop table if exists
    client.query(f"DROP TABLE IF EXISTS {table_name}")

    # Create table
    create_table_query = f"""
        CREATE TABLE {table_name} (
            {",".join(column_defs)}
        ) ENGINE = MergeTree() ORDER BY tabweight
    """
    client.query(create_table_query)

    # Insert data
    client.insert(
        table=table_name,
        data=df_data.to_numpy().tolist(),
        column_names=df.columns.to_list(),
    )

print("Data inserted")

ds1 = SchemaDataset(
    name="vius1",
    constraints=DPConstraints(
        max_eps=0.5,
        max_delta=1e-5,
    ),
    dataframes=[
        SchemaDataFrame(
            name="vius1",
            columns=tuple(columns),
            n_rows=df.shape[0],
            max_changed_rows=1,
        )
    ],
)

ds2 = SchemaDataset(
    name="vius2",
    constraints=DPConstraints(
        max_eps=0.5,
        max_delta=1e-5,
    ),
    dataframes=[
        SchemaDataFrame(
            name="vius2",
            columns=tuple(columns),
            n_rows=df.shape[0],
            max_changed_rows=1,
        )
    ],
)

schema = Schema(
    datasets=[ds1, ds2],
    role_bindings=[
        DatasetRoleBinding(
            name="analysts_can_read_vius1", dataset="vius1", role="data_analyst"
        ),
        DatasetRoleBinding(
            name="analysts_can_read_vius2", dataset="vius2", role="data_analyst"
        ),
    ],
)

with (pathlib.Path(__file__).parent.parent / "example_configs" / "schema.json").open(
    mode="w"
) as f:
    f.write(schema.model_dump_json())
