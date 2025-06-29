import traceback
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field

from dpch.common.queries import OneOfQueries
from dpch.common.column import Column
from dpch.common.interface import SQLQuery
from dpch.common.noise import LaplacianNoise, GaussianNoise
from dpch.server.config import AppConfig, load_config
from dpch.server.dependencies import get_clickhouse_client
from dpch.server.session import Session
from clickhouse_connect.driver.client import Client
from dataclasses import asdict
import numpy as np

app = FastAPI()


class QueryRequest(BaseModel):
    query: OneOfQueries = Field(discriminator="tp")


class RunQueryResponse(BaseModel):
    result: list


# Please make debug query response based on run query response but also containing data without noise, sensitivity, session, and result without noise
class DebugQueryResponse(RunQueryResponse):
    raw_result: list
    sensitivity: dict
    session: Session
    sql: dict


class SchemaResponse(BaseModel):
    columns_schema: list[Column]


# Please implement get_schema method. Method just loads config and returns response containing config.schema
# (which is pydantic model) to get config use Depends(load_config)
# Please write pydantic model for response
@app.get("/schema", response_model=SchemaResponse)
def get_schema(session: Session, conf: AppConfig = Depends(load_config)):
    return SchemaResponse(columns_schema=conf.columns_schema[session.table])


class QueryRequestWithSession(QueryRequest):
    session: Session


# Please write run_query method. Method receives QueryRequest. Add Session model and for now (for debugging purposes) lets provide it inside QueryRequest. Session contains following fields user_id, queries_total, queries_left which are int, eps, delta, eps_left, delta_left which are float, noise_type which is either "laplacian" or "gaussian".

# Method computes sensitivity via query parameter, instantiates noise based on type in session, passes eps, delta, queries_total, sensitivity to noise. Then it evaluates sql query, drops id column, applies noise to the rest and returns computed value to the client


@app.post("/query")
def run_query(
    request: QueryRequestWithSession,
    conf: AppConfig = Depends(load_config),
    client: Client = Depends(get_clickhouse_client),
) -> RunQueryResponse | DebugQueryResponse:
    if not request.query.validate_columns(conf.columns_schema[request.session.table]):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid columns",
        )

    sql_query: SQLQuery = request.query.get_sql()

    if request.session.noise_type == "laplacian":
        noise = LaplacianNoise(
            total_epsilon=request.session.max_eps,
            queries_total=request.session.queries_total,
            l1_sensitivity=request.query.l1_sensitivity,
        )
    else:
        noise = GaussianNoise(
            total_epsilon=request.session.max_eps,
            total_delta=request.session.max_delta,
            queries_total=request.session.queries_total,
            l2_sensitivity=request.query.l2_sensitivity,
        )

    try:
        result = client.query(
            sql_query.query,
            parameters=sql_query.parameters | {"table": request.session.table},
        )
    except Exception:
        if conf.debug:
            raise HTTPException(
                status_code=500,
                detail={"sql": asdict(sql_query), "traceback": traceback.format_exc()},
            )
        raise HTTPException(
            status_code=500,
            detail="Internal error",
        )

    data = result.result_rows
    noisy_data = noise.evaluate(np.array(data)[:, 1:])  # drop id column
    if conf.debug:
        return DebugQueryResponse(
            result=noisy_data.tolist(),
            raw_result=data,
            sensitivity={
                "l1": request.query.l1_sensitivity,
                "l2": request.query.l2_sensitivity,
            },
            session=request.session,
            sql=asdict(sql_query),
        )

    return RunQueryResponse(result=noisy_data.tolist())
