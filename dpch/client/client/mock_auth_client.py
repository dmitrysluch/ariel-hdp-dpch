from dpch.client.client.client import Client
from dpch.common.session import Session

# Please make MockAuthClient. It it Client child but implements method connect which receives
# Session model and authenticates using it. Token is stored in requests session in client.


class MockAuthClient(Client):
    def connect(self, session: Session):
        """
        Authenticate using the given Session model. Stores the token in the requests session headers.
        """
        resp = self.session.post(
            f"{self.base_url}/auth/new", json={"session": session.model_dump()}
        )
        resp.raise_for_status()
        token = resp.json()["token"]
        self.session.headers["Authorization"] = f"Bearer {token}"
