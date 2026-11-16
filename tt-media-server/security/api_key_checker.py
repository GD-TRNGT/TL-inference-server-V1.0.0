# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import os
import secrets

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader
from starlette.status import HTTP_401_UNAUTHORIZED


NO_AUTH = os.getenv("NO_AUTH", "").lower() in ("1", "true", "yes")
DEFAULT_API_KEY = "turing123"
# Keep authentication behavior consistent across services: 401 means the
# client credential is missing or incorrect. The deployment key defaults to
# the local development value below and remains overrideable by environment.
api_key_header = APIKeyHeader(name="Authorization", auto_error=False)


def configured_api_key() -> str:
    """Return the configured bearer token without exposing it in logs.

    ``OPENAI_API_KEY`` is the single supported deployment variable. When it
    is omitted, the local deployment default is ``turing123``; deployments
    should override it with a secret value.
    """
    return os.getenv("OPENAI_API_KEY", DEFAULT_API_KEY).strip() or DEFAULT_API_KEY


def get_api_key(api_key: str | None = Security(api_key_header)):
    if NO_AUTH:
        return None
    expected_key = configured_api_key()
    expected_header = f"Bearer {expected_key}"
    if api_key is None or not secrets.compare_digest(api_key, expected_header):
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key",
        )
    return api_key
