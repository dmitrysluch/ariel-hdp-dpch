from dpch.client.client.mock_auth_client import MockAuthClient
from dpch.client.dataset import Dataset
from dpch.common.session import Session


def connect(
    base_url: str,
    dataset: str,
    queries_total: int,
    queries_left: int,
    max_eps: float,
    max_delta: float,
    eps_used: float,
    delta_used: float,
    noise_type: str,
    cache_ttl_seconds: int = 60,
) -> Dataset:
    """
    Create a MockAuthClient, authenticate it, and return the corresponding Dataset.
    """
    client = MockAuthClient(base_url)
    session = Session(
        dataset=dataset,
        queries_total=queries_total,
        queries_left=queries_left,
        max_eps=max_eps,
        max_delta=max_delta,
        eps_used=eps_used,
        delta_used=delta_used,
        noise_type=noise_type,
    )
    client.connect(session)
    return Dataset(client, dataset, cache_ttl_seconds=cache_ttl_seconds)
