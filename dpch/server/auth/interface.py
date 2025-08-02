from abc import ABC, abstractmethod
from typing import Awaitable, Callable, TypeVar

from fastapi import Request

from dpch.common.schema import Schema
from dpch.common.session import Session

T = TypeVar("T")


class ServerSession(Session):
    id: int

    """
    An arbitrary string token used for distributed synchronization by the underlying 
    auth provider. The DPCH framework treats this field as opaque and passes it through 
    without modification.
    
    This field supports two primary synchronization patterns:
    
    1. Compare-and-Swap (CAS) revision control: The auth provider may allow multiple 
       clients to obtain a session, but only accept updates from the client holding 
       the current revision token, rejecting all others.
       
    2. Distributed locking: The token represents a lease on a distributed lock for 
       the session. Since distributed locks are typically time-bound, the token 
       ensures that only the client currently holding the lock can perform updates.
    
    Common implementations use UUIDs or other unique identifiers as lock tokens.
    """
    lock_token: str


class AuthError(Exception):
    pass


class AuthProviderMixin(ABC):
    @abstractmethod
    async def get_session(self, request: Request, schema: Schema) -> ServerSession:
        pass

    @abstractmethod
    async def get_transaction_handler(
        self, request: Request, schema: Schema
    ) -> "SessionTransactionHandlerMixin":
        pass


class SessionTransactionHandlerMixin(ABC):
    # Please change transaction signature. It should recieve updater callable
    # which recieves session as argument, returns updated session
    # and value of generic type T, which is then returned
    # from transaction method.
    @abstractmethod
    async def transaction(
        self, updater: Callable[[ServerSession], Awaitable[tuple[ServerSession, T]]]
    ) -> tuple[ServerSession, T]:
        """
        Execute a transaction that atomically updates the server session.

        This method provides atomic session updates using either Compare-And-Swap (CAS)
        semantics or distributed locking mechanisms. The transaction ensures that all
        changes to the session are applied atomically or not at all.

        Args:
            updater (Callable[[ServerSession], Awaitable[tuple[ServerSession, T]]]):
                A coroutine function that receives the current session and returns a tuple
                containing the updated session and a result value. The updater function
                may perform any business logic and modifications to the session state.

        Returns:
            ServerSession: Server session which has been successfully written by transaction.
            T: The result value returned by the updater function, passed through as-is
               after successful atomic application of session updates.

        Raises:
            Any exception raised by the updater function will be propagated after
            ensuring all acquired locks are properly released and the session state
            remains unchanged.
            AuthError may be raised either if
        """
        pass
