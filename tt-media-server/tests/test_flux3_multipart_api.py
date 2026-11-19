import asyncio
import base64
import importlib.util
import io
import json
import pathlib
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException, UploadFile
from starlette.datastructures import FormData, Headers
from PIL import Image


def _load_image_module():
    path = pathlib.Path(__file__).resolve().parents[1] / "open_ai_api" / "image.py"
    spec = importlib.util.spec_from_file_location("flux3_multipart_image_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


image_mod = _load_image_module()


@pytest.fixture(autouse=True)
def flux3_request_settings(monkeypatch):
    monkeypatch.setattr(
        "domain.image_generate_request.get_settings",
        lambda: SimpleNamespace(model_runner="tt-flux.3-image-klein"),
    )


def _png_bytes(color=(20, 80, 140), size=(64, 64)):
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


def _upload(data, filename="reference.png", content_type="image/png"):
    return UploadFile(
        file=io.BytesIO(data),
        filename=filename,
        headers=Headers({"content-type": content_type}),
    )


class _Request:
    def __init__(self, form=None, json_payload=None, content_type=None):
        self._form = form
        self._json_payload = json_payload
        self.headers = Headers(
            {"content-type": content_type}
            if content_type
            else {"content-type": "multipart/form-data; boundary=test"}
        )

    async def form(self):
        return self._form

    async def json(self):
        return self._json_payload


def _multipart_request(*fields):
    return _Request(FormData(list(fields)))


def _data_url(data):
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def test_json_and_multipart_build_the_same_request():
    reference = _png_bytes()
    json_request = _Request(
        json_payload={
            "prompt": "compose the references",
            "width": 512,
            "height": 512,
            "num_inference_steps": 16,
            "seed": 42,
            "reference_images": [_data_url(reference)],
        },
        content_type="application/json",
    )
    multipart_request = _multipart_request(
        ("prompt", "compose the references"),
        ("width", "512"),
        ("height", "512"),
        ("num_inference_steps", "16"),
        ("seed", "42"),
        ("reference_images", _upload(reference)),
    )

    json_model = asyncio.run(image_mod._parse_image_generate_request(json_request))
    multipart_model = asyncio.run(
        image_mod._parse_image_generate_request(multipart_request)
    )

    assert multipart_model.prompt == json_model.prompt
    assert multipart_model.width == json_model.width == 512
    assert multipart_model.height == json_model.height == 512
    assert multipart_model.num_inference_steps == json_model.num_inference_steps == 16
    assert multipart_model.seed == json_model.seed == 42
    assert multipart_model.reference_images == json_model.reference_images


def test_multipart_preserves_repeated_reference_image_order():
    first = _png_bytes((255, 0, 0))
    second = _png_bytes((0, 255, 0))
    request = _multipart_request(
        ("prompt", "compose"),
        ("reference_images", _upload(first, "one.png")),
        ("reference_images", _upload(second, "two.png")),
    )

    parsed = asyncio.run(image_mod._parse_image_generate_request(request))

    assert parsed.reference_images == [_data_url(first), _data_url(second)]


@pytest.mark.parametrize("field_name", ["reference_image", "image"])
def test_multipart_accepts_legacy_singular_reference_fields(field_name):
    reference = _png_bytes()
    request = _multipart_request(
        ("prompt", "edit the subject"),
        (field_name, _upload(reference, "input.png")),
    )

    parsed = asyncio.run(image_mod._parse_image_generate_request(request))

    assert parsed.reference_images == [_data_url(reference)]
    assert parsed.reference_image is None


def test_multipart_accepts_base64_reference_value():
    reference = _png_bytes()
    request = _multipart_request(
        ("prompt", "compose"),
        ("reference_images", _data_url(reference)),
    )

    parsed = asyncio.run(image_mod._parse_image_generate_request(request))

    assert parsed.reference_images == [_data_url(reference)]


def test_multipart_rejects_non_image_file():
    request = _multipart_request(
        ("prompt", "compose"),
        ("reference_images", _upload(b"not an image", "input.txt", "text/plain")),
    )

    with pytest.raises(HTTPException) as error:
        asyncio.run(image_mod._parse_image_generate_request(request))

    assert error.value.status_code == 422


def test_multipart_rejects_empty_image_file():
    request = _multipart_request(
        ("prompt", "compose"),
        ("reference_images", _upload(b"", "empty.png")),
    )

    with pytest.raises(HTTPException) as error:
        asyncio.run(image_mod._parse_image_generate_request(request))

    assert error.value.status_code == 422


def test_multipart_enforces_upload_size_limit(monkeypatch):
    monkeypatch.setattr(image_mod, "_REFERENCE_UPLOAD_MAX_BYTES", 8)
    request = _multipart_request(
        ("prompt", "compose"),
        ("reference_images", _upload(b"0123456789", "large.png")),
    )

    with pytest.raises(HTTPException) as error:
        asyncio.run(image_mod._parse_image_generate_request(request))

    assert error.value.status_code == 413


def test_non_json_non_multipart_is_rejected():
    request = _Request(content_type="text/plain")

    with pytest.raises(HTTPException) as error:
        asyncio.run(image_mod._parse_image_generate_request(request))

    assert error.value.status_code == 415


def test_sync_route_accepts_multipart_request():
    reference = _png_bytes()
    request = _multipart_request(
        ("prompt", "a black bird"),
        ("width", "512"),
        ("height", "512"),
        ("num_inference_steps", "16"),
        ("reference_images", _upload(reference)),
    )
    service = SimpleNamespace(process_request=AsyncMock(return_value=["aW1hZ2U="]))

    response = asyncio.run(
        image_mod.generate_image(request, service=service, api_key="turing123")
    )

    assert response.status_code == 200
    assert service.process_request.await_count == 1
    parsed_request = service.process_request.await_args.args[0]
    assert parsed_request.reference_images == [_data_url(reference)]


def test_async_route_accepts_multipart_request():
    request = _multipart_request(
        ("prompt", "a black bird"),
        ("width", "512"),
        ("height", "512"),
        ("num_inference_steps", "16"),
        ("reference_images", _upload(_png_bytes())),
    )
    service = SimpleNamespace(
        create_image_job=AsyncMock(return_value={"id": "job-1", "status": "queued"})
    )

    response = asyncio.run(
        image_mod.submit_async_image(request, service=service, api_key="turing123")
    )

    assert response.status_code == 202
    assert json.loads(bytes(response.body)) == {"id": "job-1", "status": "queued"}
    assert service.create_image_job.await_count == 1

