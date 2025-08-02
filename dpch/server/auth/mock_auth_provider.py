import secrets
from typing import Any, Awaitable, Callable, TypeVar

import jwt
from clickhouse_connect.driver.asyncclient import AsyncClient
from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, ValidationError

from dpch.common.schema import Schema
from dpch.common.session import Session
from dpch.server.auth.interface import (
    AuthError,
    AuthProviderMixin,
    ServerSession,
    SessionTransactionHandlerMixin,
)
from dpch.server.clickhouse import get_clickhouse_client as get_clickhouse_client_impl
from dpch.server.config import ClickHouseConfig
from dpch.server.schema.interface import SchemaProviderMixin
from dpch.server.utils import asynconce

# Please fix MockAuthSessionProvider to match interface

# 1. Please get_session directly in transaction call right before calling updater.
# Leave comment that such implementation doesn't guarantees safe concurrent use, but for mock we don't care.
# 2. Pass clickhouse client in constructor of MockSessionTransactionHandler, it must share the client with parent class


# This is a mock provider, it doesn't checks that user has permissions
# to access dataset. It is intentional.
def create_mock_auth_http_handler(
    schema_provider: SchemaProviderMixin, secret_key: str, clickhouse: Any
) -> FastAPI:
    app = FastAPI(title="DPCH Mock Auth")

    class CreateSessionRequest(BaseModel):
        session: Session

    @asynconce
    async def get_clickhouse_client() -> AsyncClient:
        return await get_clickhouse_client_impl(
            ClickHouseConfig.model_validate(clickhouse)
        )

    @app.post("/new")
    async def create_session(
        req: CreateSessionRequest,
        client: AsyncClient = Depends(get_clickhouse_client),
    ):
        session_id = secrets.randbelow(1000000) + 1
        session = ServerSession(
            id=session_id, lock_token=secrets.token_hex(16), **req.session.model_dump()
        )
        schema = await schema_provider.get_schema()
        # Find dataset in schema
        dataset = next((d for d in schema.datasets if d.name == session.dataset), None)
        if not dataset:
            raise HTTPException(
                status_code=400, detail=f"Dataset {session.dataset} not found"
            )

        # Validate DP constraints
        if session.max_eps > dataset.constraints.max_eps:
            raise HTTPException(
                status_code=400,
                detail=f"Requested epsilon {session.max_eps} exceeds maximum {dataset.constraints.max_eps}",
            )
        if session.max_delta > dataset.constraints.max_delta:
            raise HTTPException(
                status_code=400,
                detail=f"Requested delta {session.max_delta} exceeds maximum {dataset.constraints.max_delta}",
            )

        await client.command("""
            CREATE TABLE IF NOT EXISTS __dpch_sessions (
                id UInt64,
                dataset String,
                queries_total UInt32,
                queries_left UInt32,
                max_eps Float64,
                max_delta Float64,
                eps_used Float64,
                delta_used Float64,
                noise_type String,
                lock_token String,
                _timestamp DateTime DEFAULT now()
            ) ENGINE = MergeTree()
            ORDER BY (_timestamp, id)
        """)
        # Store session in ClickHouse
        await client.command(
            f"""
            INSERT INTO __dpch_sessions 
            ({",".join(session.model_dump().keys())})
            VALUES ({",".join(["%s"] * len(session.model_dump()))})
        """,
            parameters=list(session.model_dump().values()),
        )

        # Create token with only id
        token = jwt.encode({"id": session_id}, secret_key)
        return {"token": token}

    return app


T = TypeVar("T")


class MockSessionTransactionHandler(SessionTransactionHandlerMixin):
    def __init__(self, session_id: int, clickhouse_client: AsyncClient):
        self.session_id = session_id
        self.clickhouse_client = clickhouse_client

    async def transaction(
        self, updater: Callable[[ServerSession], Awaitable[tuple[ServerSession, T]]]
    ) -> tuple[ServerSession, T]:
        """
        Note: This implementation doesn't guarantee safe concurrent use as it fetches
        the session right before the updater call without proper locking. For a mock
        implementation, we don't care about this race condition. All calls for a single session
        MUST be ordered with happens-before relation.
        """
        try:
            # Fetch the latest session state right before calling updater
            result = await self.clickhouse_client.query(
                "SELECT * FROM __dpch_sessions WHERE id = %(id)s ORDER BY _timestamp DESC LIMIT 1",
                parameters={"id": self.session_id},
            )
        except Exception as e:
            raise AuthError("Failed fetching session") from e

        if not result.first_row:
            raise AuthError("Session not found")

        try:
            current_session = ServerSession.model_validate(
                dict(zip(result.column_names, result.first_row))
            )
        except ValidationError as e:
            raise AuthError("Failed validating session") from e

        # Call the updater function with current session
        updated_session, result = await updater(current_session)

        # Generate new lock token for the updated session
        updated_session.lock_token = secrets.token_hex(16)

        try:
            # Store updated session in ClickHouse
            await self.clickhouse_client.command(
                """
                INSERT INTO __dpch_sessions 
                ({fields})
                VALUES ({values})
            """.format(
                    fields=",".join(updated_session.model_dump().keys()),
                    values=",".join(["%s"] * len(updated_session.model_dump())),
                ),
                parameters=list(updated_session.model_dump().values()),
            )
        except Exception as e:
            # In a real implementation, this would handle rollback
            raise AuthError("Failed writing down new session") from e

        return updated_session, result


class MockAuthSessionProvider(AuthProviderMixin):
    def __init__(self, secret_key: str, clickhouse: Any):
        super().__init__()
        self.secret_key = secret_key
        self.clickhouse_config = ClickHouseConfig.model_validate(clickhouse)

    @asynconce
    async def get_clickhouse_client(self) -> AsyncClient:
        return await get_clickhouse_client_impl(self.clickhouse_config)

    def get_session_id(self, request: Request) -> int:
        """Extract session ID from JWT token in request"""
        auth = request.headers.get("Authorization")
        if not auth or not auth.startswith("Bearer "):
            raise AuthError("Forbidden")
        token = auth.split()[1]
        try:
            data = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            return data["id"]
        except jwt.InvalidTokenError:
            raise AuthError("Invalid token")

    async def get_session(self, request: Request, schema: Schema) -> ServerSession:
        session_id = self.get_session_id(request)
        try:
            client = await self.get_clickhouse_client()
        except Exception:
            raise AuthError("Failed connecting to ClickHouse")
        try:
            # Fetch session from ClickHouse
            result = await client.query(
                "SELECT * FROM __dpch_sessions WHERE id = %(id)s ORDER BY _timestamp DESC LIMIT 1",
                parameters={"id": session_id},
            )
        except Exception:
            raise AuthError("Failed fetching session")

        if not result.first_row:
            raise AuthError("Session not found")

        try:
            return ServerSession.model_validate(
                dict(zip(result.column_names, result.first_row))
            )
        except ValidationError as e:
            raise AuthError("Failed validating session") from e

    async def get_transaction_handler(
        self, request: Request, schema: Schema
    ) -> MockSessionTransactionHandler:
        # Extract session ID from token without querying ClickHouse
        session_id = self.get_session_id(request)
        # Get shared ClickHouse client
        try:
            client = await self.get_clickhouse_client()
        except Exception as e:
            raise AuthError("Failed connecting to ClickHouse") from e
        # Return a transaction handler for this session with shared client
        return MockSessionTransactionHandler(session_id, client)
