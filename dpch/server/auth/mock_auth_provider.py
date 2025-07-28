import secrets
from typing import Any

import jwt
from clickhouse_connect.driver.asyncclient import AsyncClient
from fastapi import Depends, FastAPI, HTTPException, Request

from dpch.common.session import Session
from dpch.server.auth.interface import AuthProviderMixin, ServerSession
from dpch.server.clickhouse import get_clickhouse_client as get_clickhouse_client_impl
from dpch.server.config import ClickHouseConfig
from dpch.server.schema.interface import SchemaProviderMixin
from dpch.server.utils import once


# This is a mock provider, it doesn't checks that user has permissions
# to access dataset. It is intentional.
def create_mock_auth_http_handler(
    schema_provider: SchemaProviderMixin, secret_key: str, clickhouse: Any
) -> FastAPI:
    app = FastAPI()

    @once
    def get_clickhouse_client() -> AsyncClient:
        return get_clickhouse_client_impl(ClickHouseConfig.model_validate(clickhouse))

    @app.post("/new")
    async def create_session(
        self,
        session: Session,
        client: AsyncClient = Depends(get_clickhouse_client),
    ):
        session_id = secrets.randbelow(1000000) + 1
        session = ServerSession(id=session_id, **session)
        schema = await self.schema_provider.get_schema()
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
            CREATE TABLE IF NOT EXISTS sessions (
                id UInt64,
                table String,
                queries_total UInt32,
                queries_left UInt32,
                max_eps Float64,
                max_delta Float64,
                eps_used Float64,
                delta_used Float64,
                noise_type String,
                _timestamp DateTime DEFAULT now()
            ) ENGINE = MergeTree()
            ORDER BY (_timestamp, user_id)
        """)
        # Store session in ClickHouse
        await client.command(
            f"""
            INSERT INTO sessions 
            ({",".join(session.model_dump().keys())})
            VALUES ({",".join(["%s"] * len(session.model_dump()))})
        """,
            parameters=list(session.model_dump().values()),
        )

        # Create token with only id
        token = jwt.encode({"id": id}, self.secret_key)
        return {"token": token}

    return app


class MockAuthSessionProvider(AuthProviderMixin):
    def __init__(self, secret_key: str, clickhouse: Any):
        super().__init__()
        self.secret_key = secret_key
        self.clickhouse_config = ClickHouseConfig.model_validate(clickhouse)

    @once
    def get_clickhouse_client(self) -> AsyncClient:
        return get_clickhouse_client_impl(self.clickhouse_config)

    async def get_session(
        self,
        request: Request,
    ) -> ServerSession:
        auth = request.headers.get("Authorization")
        if not auth or not auth.startswith("Bearer "):
            raise HTTPException(status_code=403, detail="Forbidden")
        token = auth.split()[1]
        try:
            data = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            session_id = data["id"]
            client = self.get_clickhouse_client()
            # Fetch session from ClickHouse
            result = await client.query(
                "SELECT * FROM sessions WHERE id = %(id)s ORDER BY _timestamp DESC LIMIT 1",
                parameters={"id": session_id},
            )
            if not result.rows:
                raise HTTPException(status_code=403, detail="Session not found")

            return ServerSession.model_validate(
                dict(zip(result.column_names, result.first_row))
            )
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=403, detail="Invalid token")
        except Exception:
            raise HTTPException(status_code=500, detail="Internal error")

    async def update_session(self, session: ServerSession) -> None:
        client = self.get_clickhouse_client()
        # Update session in ClickHouse
        # ClickHouse doesn't like updates, so we write all sessions and read last.
        await client.command(
            """
            INSERT INTO sessions 
            ({fields})
            VALUES ({values})
        """.format(
                fields=",".join(session.model_dump().keys()),
                values=",".join(["%s"] * len(session.model_dump())),
            ),
            parameters=list(session.model_dump().values()),
        )
