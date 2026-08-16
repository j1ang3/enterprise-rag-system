from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError


PASSWORD_MIN_LENGTH = 8
PASSWORD_MAX_LENGTH = 128
_PASSWORD_HASHER = PasswordHasher()


class PasswordValidationError(ValueError):
    """Raised when a plaintext password violates the first-version policy."""


def validate_password(password: str) -> None:
    if not isinstance(password, str):
        raise PasswordValidationError("Password must be a string.")
    if len(password) < PASSWORD_MIN_LENGTH:
        raise PasswordValidationError(
            f"Password must contain at least {PASSWORD_MIN_LENGTH} characters."
        )
    if len(password) > PASSWORD_MAX_LENGTH:
        raise PasswordValidationError(
            f"Password must not exceed {PASSWORD_MAX_LENGTH} characters."
        )


def hash_password(password: str) -> str:
    validate_password(password)
    return _PASSWORD_HASHER.hash(password)


def verify_password(password: str, encoded_hash: str) -> bool:
    if not isinstance(password, str) or not isinstance(encoded_hash, str):
        return False
    try:
        return _PASSWORD_HASHER.verify(encoded_hash, password)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False
