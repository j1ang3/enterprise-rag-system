from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, SecretStr


class RegisterRequest(BaseModel):
    username: str
    # Length policy is applied after parsing so FastAPI's validation payload
    # never echoes the submitted plaintext in an `input` field.
    password: SecretStr

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "username": "api-learner",
                    "password": "synthetic-password-123",
                }
            ]
        }
    )


class LoginRequest(BaseModel):
    username: str
    password: SecretStr

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "username": "api-learner",
                    "password": "synthetic-password-123",
                }
            ]
        }
    )


class PublicUser(BaseModel):
    user_id: UUID
    username: str
    created_at: datetime


class RegisterData(PublicUser):
    pass


class RegisterResponse(BaseModel):
    success: bool
    data: RegisterData
    message: str


class LoginData(BaseModel):
    access_token: str
    token_type: str
    expires_in: int


class LoginResponse(BaseModel):
    success: bool
    data: LoginData
    message: str


class CurrentUserResponse(BaseModel):
    success: bool
    data: PublicUser
    message: str
