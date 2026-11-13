# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import os

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader
from starlette.status import HTTP_401_UNAUTHORIZED

DEFAULT_API_KEY = "your-secret-key"
NO_AUTH = os.getenv("NO_AUTH", "").lower() in ("1", "true", "yes")
# auto_error=False when NO_AUTH so missing header is None, not 403.
api_key_header = APIKeyHeader(name="Authorization", auto_error=not NO_AUTH)


def configured_api_key() -> str:
    """Return the configured bearer token without exposing it in logs.

    ``OPENAI_API_KEY`` is the conventional deployment variable.  ``API_KEY``
    remains supported for existing deployments and scripts.  Prefer the former
    when both are present so an injected production secret overrides a legacy
    image/default setting.
    """
    return os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY") or DEFAULT_API_KEY


def get_api_key(api_key: str | None = Security(api_key_header)):
    if NO_AUTH:
        return None
    if api_key != f"Bearer {configured_api_key()}":
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key",
        )
    return api_key
