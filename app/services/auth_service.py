from dataclasses import dataclass

from app.auth.passwords import PASSWORD_MAX_LENGTH, hash_password, verify_password
from app.auth.tokens import AccessToken, create_access_token
from app.services.user_registry import (
    InvalidUsernameError,
    SessionFactory,
    UserIdentity,
    UsernameAlreadyExistsError,
    create_user,
    get_user_by_username,
    get_user_credential_by_username,
    normalize_username,
)


class InvalidCredentialsError(ValueError):
    """Public login failures intentionally share one generic error type."""


@dataclass(frozen=True)
class LoginResult:
    user: UserIdentity
    token: AccessToken


def register_user(
    username: str,
    password: str,
    session_factory: SessionFactory | None = None,
) -> UserIdentity:
    canonical_username = normalize_username(username)
    if get_user_by_username(canonical_username, session_factory) is not None:
        raise UsernameAlreadyExistsError(
            "The canonical username already exists."
        )
    password_hash = hash_password(password)
    return create_user(
        canonical_username,
        session_factory,
        password_hash=password_hash,
    )


def login_user(
    username: str,
    password: str,
    session_factory: SessionFactory | None = None,
) -> LoginResult:
    if not isinstance(password, str) or len(password) > PASSWORD_MAX_LENGTH:
        raise InvalidCredentialsError("Invalid username or password.")
    try:
        canonical_username = normalize_username(username)
    except InvalidUsernameError as exc:
        raise InvalidCredentialsError("Invalid username or password.") from exc

    credential = get_user_credential_by_username(
        canonical_username,
        session_factory,
    )
    if credential is None or not verify_password(
        password,
        credential.password_hash,
    ):
        raise InvalidCredentialsError("Invalid username or password.")
    return LoginResult(
        user=credential.identity,
        token=create_access_token(credential.identity.user_id),
    )
