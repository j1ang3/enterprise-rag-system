from typing import Any, Optional


def success_response(data: Optional[Any] = None, message: str = "ok"):
    return {
        "success": True,
        "data": data,
        "message": message,
    }


def error_response(message: str = "error", data: Optional[Any] = None):
    return {
        "success": False,
        "data": data,
        "message": message,
    }