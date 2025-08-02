import json
import traceback
from contextlib import asynccontextmanager
from io import BytesIO
from typing import Literal

import numpy as np
import uttlv
from fastapi import Depends, FastAPI, HTTPException, Response
from numpydantic import NDArray, Shape
from pydantic import BaseModel, Field

from dpch.common.noise import NotEnoughPrivacyBudget, create_noise_from_query
from dpch.common.queries.interface import DPValueError
from dpch.common.queries.queries import OneOfQueries
from dpch.common.schema import Schema, SchemaDataset
from dpch.common.session import Session
from dpch.server.auth.factory import get_auth_http_handler
from dpch.server.auth.interface import (
    AuthError,
    ServerSession,
    SessionTransactionHandlerMixin,
)
from dpch.server.config import AppConfig
from dpch.server.dependencies import (
    get_executor,
    get_schema,
    get_schema_provider,
    get_session,
    get_session_transaction_handler,
    load_config,
    get_use_tlv,
)
from dpch.server.executors.interface import ExecutorMixin, ExecutorValueError


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = load_config()
    app.state.config = config

    schema_provider = get_schema_provider(config)
    auth_http_handler = get_auth_http_handler(config.auth, schema_provider)
    if auth_http_handler is not None:
        app.mount("/auth", auth_http_handler, name="auth")
    yield


app = FastAPI(lifespan=lifespan, title="DPCH Proxy")


class QueryRequest(BaseModel):
    query: OneOfQueries = Field(discriminator="tp")


class TLVResponse(Response):
    """Custom response class for TLV serialization"""

    media_type = "application/x-tlv"


def serialize_numpy_to_bytes(arr: np.ndarray) -> bytes:
    """Serialize numpy array to bytes using np.save and BytesIO"""
    buffer = BytesIO()
    np.save(buffer, arr)
    return buffer.getvalue()


def deserialize_numpy_from_bytes(data: bytes) -> np.ndarray:
    """Deserialize numpy array from bytes using np.load and BytesIO"""
    buffer = BytesIO(data)
    return np.load(buffer, allow_pickle=False)


# Please add pydantic custom serializer for which dumps numpy arrays by first converting them to lists to RunQueryResponse and DebugQueryResponse
class RunQueryResponse(BaseModel):
    tp: Literal["RunQueryResponse"] = "RunQueryResponse"
    result: NDArray[Shape["*, *"], np.float64]  # noqa: F722
    new_session: Session

    def create_tlv_response(self) -> TLVResponse:
        """Create TLV response from this response object"""
        tlv_data = {}

        # Serialize numpy arrays as binary
        tlv_data["result"] = serialize_numpy_to_bytes(self.result)

        # Serialize other fields as JSON
        tlv_data["tp"] = self.tp.encode()
        tlv_data["new_session"] = self.new_session.model_dump_json().encode()

        tlv_bytes = uttlv.pack(tlv_data)
        return TLVResponse(content=tlv_bytes)


# Please make debug query response based on run query response but also containing data without noise, sensitivity, session, and result without noise
class DebugQueryResponse(RunQueryResponse):
    tp: Literal["DebugQueryResponse"] = "DebugQueryResponse"
    raw_result: NDArray[Shape["*, *"], np.float64]  # noqa: F722
    sensitivity: dict
    executor_info: dict

    def create_tlv_response(self) -> TLVResponse:
        """Create TLV response from this response object"""
        tlv_data = {}

        # Serialize numpy arrays as binary
        tlv_data["result"] = serialize_numpy_to_bytes(self.result)
        tlv_data["raw_result"] = serialize_numpy_to_bytes(self.raw_result)

        # Serialize other fields as JSON
        tlv_data["tp"] = self.tp.encode()
        tlv_data["new_session"] = self.new_session.model_dump_json().encode()

        tlv_data["sensitivity"] = json.dumps(self.sensitivity).encode()
        tlv_data["executor_info"] = json.dumps(self.executor_info).encode()

        tlv_bytes = uttlv.pack(tlv_data)
        return TLVResponse(content=tlv_bytes)


class SchemaResponse(BaseModel):
    ch_schema: SchemaDataset = Field(alias="schema")


# Please implement get_schema method. Method just loads config and returns response containing config.schema
# (which is pydantic model) to get config use Depends(get_session)
# Please write pydantic model for response
@app.get("/schema")
def handle_schema(
    schema: Schema = Depends(get_schema, use_cache=True),
    session: ServerSession = Depends(get_session, use_cache=True),
) -> SchemaResponse:
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
async def handle_query(
    request: QueryRequest,
    session_tx_handler: SessionTransactionHandlerMixin = Depends(
        get_session_transaction_handler, use_cache=True
    ),
    schema: Schema = Depends(get_schema, use_cache=True),
    executor: ExecutorMixin = Depends(get_executor, use_cache=True),
    use_tlv: bool = Depends(get_use_tlv, use_cache=True),
):
    config: AppConfig = app.state.config
    query = request.query

    # TODO: There exists an optimization here which can be done,
    # yet it will compilcate the interface:
    # Session consists of part which is to be updated (privacy budget left)
    # and a constant part (for example which dataset session is issued for).
    # executor.execute in fact requires only the constant part, so it can be done outside of session.
    #
    # The optimal pipeline will then be:
    # 1. Get some session without locking.
    # 2. Check if that session has enough privacy budget left.
    # 3. Execute query (using only constant part of session).
    # 4. Update the privacy budget inside a transaction (possibly failing and rejecting the query result).
    async def _tx(session: ServerSession):
        ds = schema.dataset_from_session(session)
        query.validate_against_schema(ds)
        new_sess, noise = create_noise_from_query(query, session, ds)
        data, debug_info = await executor.execute(query, schema, session)
        return ServerSession(
            **(
                new_sess.model_dump()
                | {"id": session.id, "lock_token": session.lock_token}
            )
        ), (data, debug_info, noise, ds)

    try:
        new_sess, (data, debug_info, noise, ds) = await session_tx_handler.transaction(
            _tx
        )
    except DPValueError as e:
        raise HTTPException(
            status_code=400,
            detail={"type": "DPValueError", "error": str(e)}
            | {"traceback": traceback.format_exc()}
            if config.debug
            else {},
        )
    except AuthError as e:
        raise HTTPException(
            status_code=403,
            detail={"type": "AuthError", "error": str(e)}
            | {"traceback": traceback.format_exc()}
            if config.debug
            else {},
        )
    except ExecutorValueError as e:
        if config.debug:
            raise HTTPException(
                status_code=500,
                detail={
                    "type": "ExecutorValueError",
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                },
            )
        raise HTTPException(
            status_code=500,
            detail="Internal error",
        )
    except NotEnoughPrivacyBudget as e:
        raise HTTPException(
            status_code=403,
            detail={"type": "NotEnoughPrivacyBudget", "error": str(e)}
            | {"traceback": traceback.format_exc()}
            if config.debug
            else {},
        )

    noisy_data = noise.apply(data)

    if config.debug:
        response = DebugQueryResponse(
            result=noisy_data,
            raw_result=data,
            sensitivity={
                "l1": query.sensitivity_over_columns(ds, "l1"),
                "l2": query.sensitivity_over_columns(ds, "l2"),
            },
            new_session=new_sess,
            executor_info=debug_info,
        )
    else:
        response = RunQueryResponse(
            result=noisy_data,
            new_session=new_sess,
        )

    if use_tlv:
        return response.create_tlv_response()
    else:
        # For JSON: the field serializers will convert numpy arrays to lists
        return response
