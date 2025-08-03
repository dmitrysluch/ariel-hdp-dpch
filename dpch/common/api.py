import json
from enum import IntEnum
from io import BytesIO
from typing import Literal

import numpy as np
import uttlv
from fastapi import Response
from numpydantic import NDArray, Shape
from pydantic import BaseModel, Field

from dpch.common.queries.queries import OneOfQueries
from dpch.common.schema import SchemaDataset
from dpch.common.session import Session


class TLVTag(IntEnum):
    """Binary tags for TLV encoding"""

    TP = 1
    RESULT = 2
    NEW_SESSION = 3
    RAW_RESULT = 4
    SENSITIVITY = 5
    EXECUTOR_INFO = 6


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


class QueryRequest(BaseModel):
    query: OneOfQueries = Field(discriminator="tp")


class RunQueryResponse(BaseModel):
    tp: Literal["RunQueryResponse"] = "RunQueryResponse"
    result: NDArray[Shape["*, *"], np.float64]  # noqa: F722
    new_session: Session

    def create_tlv_response(self) -> TLVResponse:
        """Create TLV response from this response object"""
        tlv = uttlv.TLV()

        # Serialize numpy arrays as binary
        tlv[TLVTag.RESULT] = serialize_numpy_to_bytes(self.result)

        # Serialize other fields as JSON
        tlv[TLVTag.TP] = self.tp.encode()
        tlv[TLVTag.NEW_SESSION] = self.new_session.model_dump_json().encode()

        tlv_bytes = tlv.to_byte_array()
        return TLVResponse(content=tlv_bytes)

    @classmethod
    def from_tlv_bytes(cls, tlv: uttlv.TLV) -> "RunQueryResponse":
        """Create RunQueryResponse from TLV object"""
        # Deserialize numpy arrays from binary
        result = deserialize_numpy_from_bytes(tlv[TLVTag.RESULT])

        # Deserialize other fields from JSON
        new_session = Session.model_validate_json(tlv[TLVTag.NEW_SESSION].decode())

        return cls(
            result=result,
            new_session=new_session,
        )


class DebugQueryResponse(RunQueryResponse):
    tp: Literal["DebugQueryResponse"] = "DebugQueryResponse"
    raw_result: NDArray[Shape["*, *"], np.float64]  # noqa: F722
    sensitivity: dict
    executor_info: dict

    def create_tlv_response(self) -> TLVResponse:
        """Create TLV response from this response object"""
        tlv = uttlv.TLV()

        # Serialize numpy arrays as binary
        tlv[TLVTag.RESULT] = serialize_numpy_to_bytes(self.result)
        tlv[TLVTag.RAW_RESULT] = serialize_numpy_to_bytes(self.raw_result)

        # Serialize other fields as JSON
        tlv[TLVTag.TP] = self.tp.encode()
        tlv[TLVTag.NEW_SESSION] = self.new_session.model_dump_json().encode()
        tlv[TLVTag.SENSITIVITY] = json.dumps(self.sensitivity).encode()
        tlv[TLVTag.EXECUTOR_INFO] = json.dumps(self.executor_info).encode()

        tlv_bytes = tlv.to_byte_array()
        return TLVResponse(content=tlv_bytes)

    @classmethod
    def from_tlv_bytes(cls, tlv: uttlv.TLV) -> "DebugQueryResponse":
        """Create DebugQueryResponse from TLV object"""
        # Deserialize numpy arrays from binary
        result = deserialize_numpy_from_bytes(tlv[TLVTag.RESULT])
        raw_result = deserialize_numpy_from_bytes(tlv[TLVTag.RAW_RESULT])

        # Deserialize other fields from JSON
        new_session = Session.model_validate_json(tlv[TLVTag.NEW_SESSION].decode())
        sensitivity = json.loads(tlv[TLVTag.SENSITIVITY].decode())
        executor_info = json.loads(tlv[TLVTag.EXECUTOR_INFO].decode())

        return cls(
            result=result,
            raw_result=raw_result,
            sensitivity=sensitivity,
            executor_info=executor_info,
            new_session=new_session,
        )


def response_from_tlv_bytes(data: bytes) -> RunQueryResponse | DebugQueryResponse:
    """Factory function to create appropriate response from TLV bytes"""
    tlv = uttlv.TLV()
    tlv.from_byte_array(data)

    # Get the response type from TLV
    response_type = tlv[TLVTag.TP].decode()

    if response_type == "RunQueryResponse":
        return RunQueryResponse.from_tlv_bytes(tlv)
    elif response_type == "DebugQueryResponse":
        return DebugQueryResponse.from_tlv_bytes(tlv)
    else:
        raise ValueError(f"Unknown response type: {response_type}")


class SchemaResponse(BaseModel):
    ch_schema: SchemaDataset = Field(alias="schema")
