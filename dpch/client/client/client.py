import requests

from dpch.client.client.interface import ClientMixin
from dpch.common.api import (
    DebugQueryResponse,
    QueryRequest,
    RunQueryResponse,
    SchemaResponse,
    response_from_tlv_bytes,
)

# Lets implement Client using requests library. Use session for connection pooling.
# Don't care about auth here, it will be added in child classes


class Client(ClientMixin):
    def __init__(self, base_url: str, use_tlv: bool = True):
        self.base_url = base_url.rstrip("/")
        self.use_tlv = use_tlv
        self.session = requests.Session()

    def query(self, request: QueryRequest) -> RunQueryResponse | DebugQueryResponse:
        headers = {"Content-Type": "application/json"}
        if self.use_tlv:
            headers["Accept"] = "application/x-tlv"
        else:
            headers["Accept"] = "application/json"
        resp = self.session.post(
            f"{self.base_url}/query", data=request.model_dump_json(), headers=headers
        )
        if not resp.ok:
            raise requests.HTTPError(
                f"HTTP {resp.status_code}: {resp.text}", response=resp
            )
        if self.use_tlv:
            return response_from_tlv_bytes(resp.content)
        else:
            data = resp.json()
            if data.get("tp") == "DebugQueryResponse":
                return DebugQueryResponse.model_validate(data)
            else:
                return RunQueryResponse.model_validate(data)

    def schema(self) -> SchemaResponse:
        headers = {"Accept": "application/json"}
        resp = self.session.get(f"{self.base_url}/schema", headers=headers)
        if not resp.ok:
            raise requests.HTTPError(
                f"HTTP {resp.status_code}: {resp.text}", response=resp
            )
        return SchemaResponse.model_validate(resp.json())
