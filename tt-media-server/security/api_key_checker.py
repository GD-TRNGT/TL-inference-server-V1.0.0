# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import os
import secrets

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader
from starlette.status import HTTP_401_UNAUTHORIZED, HTTP_503_SERVICE_UNAVAILABLE


NO_AUTH = os.getenv("NO_AUTH", "").lower() in ("1", "true", "yes")
# Always let get_api_key produce the public contract: 503 when the server is
# missing its key configuration and 401 when a configured service receives a
# missing or incorrect client credential.
api_key_header = APIKeyHeader(name="Authorization", auto_error=False)


def configured_api_key() -> str | None:
    """Return the configured bearer token without exposing it in logs.

    ``OPENAI_API_KEY`` is the single supported deployment variable. There
    is deliberately no built-in production default: a published
    ``your-secret-key`` makes every deployment public.
    """
    value = os.getenv("OPENAI_API_KEY", "").strip()
    return value or None


def get_api_key(api_key: str | None = Security(api_key_header)):
    if NO_AUTH:
        return None
    expected_key = configured_api_key()
    if expected_key is None:
        raise HTTPException(
            status_code=HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server API key is not configured; set OPENAI_API_KEY",
        )
    expected_header = f"Bearer {expected_key}"
    if api_key is None or not secrets.compare_digest(api_key, expected_header):
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key",
        )
    return api_key
