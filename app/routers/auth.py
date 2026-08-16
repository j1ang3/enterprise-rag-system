from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.auth.dependencies import (
    get_current_user,
    require_authentication_configuration,
)
from app.auth.passwords import PasswordValidationError
from app.db.session import DatabaseConfigurationError
from app.schemas.auth import (
    CurrentUserResponse,
    LoginRequest,
    LoginResponse,
    RegisterRequest,
    RegisterResponse,
)
from app.schemas.common import ErrorResponse
from app.services.auth_service import InvalidCredentialsError, login_user, register_user
from app.services.user_registry import (
    InvalidUsernameError,
    UserIdentity,
    UserRegistryUnavailableError,
    UsernameAlreadyExistsError,
)
from app.utils.response import success_response


router = APIRouter(
    prefix="/auth",
    tags=["auth"],
    dependencies=[Depends(require_authentication_configuration)],
)


@router.post(
    "/register",
    response_model=RegisterResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a user",
    description=(
        "Create a PostgreSQL-backed user after applying username and password policy. "
        "The response never includes the submitted password or password hash."
    ),
    responses={
        409: {
            "model": ErrorResponse,
            "description": "The normalized username is already registered.",
        },
        503: {
            "model": ErrorResponse,
            "description": "Authentication configuration or storage is unavailable.",
        },
    },
)
def register(request: RegisterRequest):
    try:
        user = register_user(
            request.username,
            request.password.get_secret_value(),
        )
    except (InvalidUsernameError, PasswordValidationError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc
    except UsernameAlreadyExistsError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username is already registered.",
        ) from exc
    except (DatabaseConfigurationError, UserRegistryUnavailableError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication storage is temporarily unavailable.",
        ) from exc
    return success_response(
        data=user,
        message="user registered",
    )


@router.post(
    "/login",
    response_model=LoginResponse,
    summary="Log in and issue an access token",
    description=(
        "Verify username and password, then return a Bearer JWT and its actual "
        "lifetime in seconds."
    ),
    responses={
        401: {
            "model": ErrorResponse,
            "description": "The username or password is invalid.",
        },
        503: {
            "model": ErrorResponse,
            "description": "Authentication configuration or storage is unavailable.",
        },
    },
)
def login(request: LoginRequest):
    try:
        result = login_user(
            request.username,
            request.password.get_secret_value(),
        )
    except InvalidCredentialsError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
    except (DatabaseConfigurationError, UserRegistryUnavailableError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication storage is temporarily unavailable.",
        ) from exc
    return success_response(
        data={
            "access_token": result.token.value,
            "token_type": "bearer",
            "expires_in": result.token.expires_in,
        },
        message="login successful",
    )


@router.get(
    "/me",
    response_model=CurrentUserResponse,
    summary="Get the authenticated user",
    description=(
        "Resolve the Bearer JWT subject to the current PostgreSQL user. "
        "Document permissions are not stored in the token."
    ),
    responses={
        401: {
            "model": ErrorResponse,
            "description": "Bearer credentials are missing, invalid, expired, or stale.",
        },
        503: {
            "model": ErrorResponse,
            "description": "Authentication configuration or storage is unavailable.",
        },
    },
)
def me(
    current_user: Annotated[UserIdentity, Depends(get_current_user)],
):
    return success_response(
        data=current_user,
        message="current user resolved",
    )
