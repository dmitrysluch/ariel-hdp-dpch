from typing import Literal

from pydantic import BaseModel


class Session(BaseModel):
    dataset: str
    queries_total: int
    queries_left: int
    max_eps: float
    max_delta: float
    eps_used: float
    delta_used: float
    noise_type: Literal["laplacian"] | Literal["gaussian"]
