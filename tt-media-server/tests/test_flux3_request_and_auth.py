# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

from types import SimpleNamespace

import pytest
from domain.image_generate_request import ImageGenerateRequest
from pydantic import ValidationError
from security import api_key_checker
from starlette.status import HTTP_401_UNAUTHORIZED


@pytest.fixture
def flux3_runner(monkeypatch):
    monkeypatch.setattr(
        "domain.image_generate_request.get_settings",
        lambda: SimpleNamespace(model_runner="tt-flux.3-image-klein"),
    )


@pytest.mark.parametrize("width,height", [(256, 256), (256, 2048), (4096, 4096)])
def test_flux3_accepts_officially_supported_geometry(flux3_runner, width, height):
    request = ImageGenerateRequest(prompt="a black bird", width=width, height=height)

    assert (request.width, request.height) == (width, height)


@pytest.mark.parametrize(
    "width,height,reason",
    [
        (255, 1024, "greater than or equal to 256"),
        (257, 1024, "multiples of 16"),
        (256, 4096, "aspect ratio"),
        (4096, 4112, "must be <= 4096"),
    ],
)
def test_flux3_rejects_geometry_outside_official_contract(
    flux3_runner, width, height, reason
):
    with pytest.raises(ValidationError, match=reason):
        ImageGenerateRequest(prompt="a black bird", width=width, height=height)


def test_openai_api_key_takes_priority_over_legacy_api_key(monkeypatch):
    monkeypatch.setenv("API_KEY", "legacy-token")
    monkeypatch.setenv("OPENAI_API_KEY", "preferred-token")

    assert api_key_checker.configured_api_key() == "preferred-token"
    assert api_key_checker.get_api_key("Bearer preferred-token") == "Bearer preferred-token"
    with pytest.raises(Exception) as error:
        api_key_checker.get_api_key("Bearer legacy-token")
    assert getattr(error.value, "status_code", None) == HTTP_401_UNAUTHORIZED


def test_legacy_api_key_remains_supported(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("API_KEY", "legacy-token")

    assert api_key_checker.configured_api_key() == "legacy-token"
