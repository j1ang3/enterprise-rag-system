from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from uuid import UUID

import jwt
from jwt.exceptions import InvalidTokenError

from app.core.config import settings


JWT_ALGORITHM = "HS256"
MIN_JWT_SECRET_BYTES = 32


class AuthenticationConfigurationError(RuntimeError):
    """Raised when authentication cannot safely sign or validate tokens."""


class InvalidAccessTokenError(ValueError):
    """Raised when a Bearer token cannot resolve a valid stable user ID."""


@dataclass(frozen=True)
class AccessToken:
    value: str
    expires_in: int


def require_jwt_secret() -> str:
    secret = settings.jwt_secret_key.get_secret_value()
    if not secret or len(secret.encode("utf-8")) < MIN_JWT_SECRET_BYTES:
        raise AuthenticationConfigurationError(
            "JWT_SECRET_KEY must contain at least 32 bytes."
        )
    return secret


def create_access_token(
    user_id: UUID,
    *,
    now: datetime | None = None,
) -> AccessToken:
    issued_at = now or datetime.now(timezone.utc)
    if issued_at.tzinfo is None or issued_at.utcoffset() is None:
        raise ValueError("JWT timestamps must be timezone-aware.")
    issued_at = issued_at.astimezone(timezone.utc)
    expires_in = settings.jwt_access_token_expire_minutes * 60
    expires_at = issued_at + timedelta(seconds=expires_in)
    payload = {
        "sub": str(user_id),
        "iat": issued_at,
        "exp": expires_at,
    }
    return AccessToken(
        value=jwt.encode(
            payload,
            require_jwt_secret(),
            algorithm=JWT_ALGORITHM,
        ),
        expires_in=expires_in,
    )


def decode_access_token(token: str) -> UUID:
    if not isinstance(token, str) or not token:
        raise InvalidAccessTokenError("Access token is invalid.")
    try:
        payload = jwt.decode(
            token,
            require_jwt_secret(),
            algorithms=[JWT_ALGORITHM],
            options={"require": ["sub", "iat", "exp"]},
        )
        subject = payload["sub"]
        if not isinstance(subject, str):
            raise InvalidAccessTokenError("Access token subject is invalid.")
        return UUID(subject)
    except AuthenticationConfigurationError:
        raise
    except (InvalidTokenError, KeyError, TypeError, ValueError) as exc:
        raise InvalidAccessTokenError("Access token is invalid.") from exc
