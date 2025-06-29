from pydantic import BaseModel
from typing import Literal


class Session(BaseModel):
    table: str
    user_id: int
    queries_total: int
    queries_left: int
    max_eps: float
    max_delta: float
    eps_used: float
    delta_used: float
    noise_type: Literal["laplacian"] | Literal["gaussian"]
