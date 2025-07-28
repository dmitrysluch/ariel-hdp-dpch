import traceback
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from dpch.common.noise import GaussianNoise, LaplacianNoise, NotEnoughPrivacyBudget
from dpch.common.queries.interface import DPValueError
from dpch.common.queries.queries import OneOfQueries
from dpch.common.schema import Schema, SchemaDataset
from dpch.common.session import Session
from dpch.server.auth.factory import get_auth_http_handler
from dpch.server.auth.interface import ServerSession
from dpch.server.config import AppConfig
from dpch.server.dependencies import (
    get_executor,
    get_schema,
    get_schema_provider,
    get_session,
    load_config,
)
from dpch.server.executors.interface import ExecutorMixin, ExecutorValueError


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = load_config()
    app.state.config = config

    schema_provider = get_schema_provider()
    auth_http_handler = get_auth_http_handler(config, schema_provider)
    app.mount("/auth", auth_http_handler, name="auth")

    yield


app = FastAPI(lifespan=lifespan)


class QueryRequest(BaseModel):
    query: OneOfQueries = Field(discriminator="tp")


class RunQueryResponse(BaseModel):
    result: list
    session: Session


# Please make debug query response based on run query response but also containing data without noise, sensitivity, session, and result without noise
class DebugQueryResponse(RunQueryResponse):
    raw_result: list
    sensitivity: dict
    sql: dict


class SchemaResponse(BaseModel):
    schema: SchemaDataset


# Please implement get_schema method. Method just loads config and returns response containing config.schema
# (which is pydantic model) to get config use Depends(get_session)
# Please write pydantic model for response
@app.get("/schema", response_model=SchemaResponse)
def get_schema(
    schema: Schema = Depends(get_schema),
    session: ServerSession = Depends(get_session),
):
    try:
        dataset = schema.dataset_from_session(session)
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail="Failed getting dataset from schema",
        ) from e
    return SchemaResponse(schema=dataset)


# Please fix run_query method. Get config from app.state.config
# Use dependency injection to get session and schema as above as well as an executor.
# Then execute query via executor. Handle ExecutorError. Provide full error message to client only if debug flag is set to true.


# Method computes sensitivity via query parameter, instantiates noise based on type in session, passes eps, delta, queries_total, sensitivity to noise. Then it evaluates sql query, drops id column, applies noise to the rest and returns computed value to the client


@app.post("/query")
async def run_query(
    request: QueryRequest,
    session: ServerSession = Depends(get_session),
    schema: Schema = Depends(get_schema),
    executor: ExecutorMixin = Depends(get_executor),
) -> RunQueryResponse | DebugQueryResponse:
    config: AppConfig = app.state.config
    query = request.query

    try:
        query.validate_against_schema()
    except DPValueError as e:
        raise HTTPException(
            status_code=400,
            detail={"error": str(e)} | {"traceback": traceback.format_exc()}
            if config.debug
            else {},
        )
    # Instantiate noise
    if session.noise_type == "laplacian":
        noise = LaplacianNoise(
            total_epsilon=session.max_eps,
            queries_total=session.queries_total,
            l1_sensitivity=query.l1_sensitivity,
        )
    else:
        noise = GaussianNoise(
            total_epsilon=session.max_eps,
            total_delta=session.max_delta,
            queries_total=session.queries_total,
            l2_sensitivity=query.l2_sensitivity,
        )
    try:
        data, debug_info = await executor.execute(query, schema, session)
    except ExecutorValueError as e:
        if config.debug:
            raise HTTPException(
                status_code=500,
                detail={"error": str(e), "traceback": traceback.format_exc()},
            )
        raise HTTPException(
            status_code=500,
            detail="Internal error",
        )
    try:
        noisy_data = noise.evaluate(data)
    except NotEnoughPrivacyBudget as e:
        raise HTTPException(
            status_code=403,
            detail={"error": str(e)} | {"traceback": traceback.format_exc()}
            if config.debug
            else {},
        )
    if config.debug:
        return DebugQueryResponse(
            result=noisy_data.tolist(),
            raw_result=data.tolist(),
            sensitivity={
                "l1": query.l1_sensitivity,
                "l2": query.l2_sensitivity,
            },
            session=session,
            executor_info=debug_info,
        )
    return RunQueryResponse(result=noisy_data.tolist(), session=session)
