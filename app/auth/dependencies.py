from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.auth.tokens import (
    AuthenticationConfigurationError,
    InvalidAccessTokenError,
    decode_access_token,
    require_jwt_secret,
)
from app.db.session import DatabaseConfigurationError
from app.services.user_registry import (
    UserIdentity,
    UserRegistryUnavailableError,
    get_user_by_id,
)


_BEARER = HTTPBearer(auto_error=False)


def _unauthorized() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication credentials are invalid.",
        headers={"WWW-Authenticate": "Bearer"},
    )


def require_authentication_configuration() -> None:
    try:
        require_jwt_secret()
    except AuthenticationConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication is temporarily unavailable.",
        ) from exc


def get_current_user(
    credentials: Annotated[
        HTTPAuthorizationCredentials | None,
        Depends(_BEARER),
    ],
) -> UserIdentity:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise _unauthorized()
    try:
        user_id = decode_access_token(credentials.credentials)
        user = get_user_by_id(user_id)
    except InvalidAccessTokenError as exc:
        raise _unauthorized() from exc
    except AuthenticationConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication is temporarily unavailable.",
        ) from exc
    except (DatabaseConfigurationError, UserRegistryUnavailableError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication storage is temporarily unavailable.",
        ) from exc
    if user is None:
        raise _unauthorized()
    return user
