import base64
import io
from types import SimpleNamespace

import pytest
from PIL import Image
from pydantic import ValidationError

from domain.image_generate_request import ImageGenerateRequest
from domain.image_to_image_request import ImageToImageRequest


@pytest.fixture(autouse=True)
def flux3_request_settings(monkeypatch):
    monkeypatch.setattr("domain.image_generate_request.get_settings",
                        lambda: SimpleNamespace(model_runner="tt-flux.3-image-klein"))


def _image_payload(index=0, size=(64, 64)):
    buffer = io.BytesIO()
    Image.new("RGB", size, (index, 30, 90)).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def test_image_generation_accepts_optional_reference_image():
    request = ImageGenerateRequest(
        prompt="Preserve the subject and move it to a snowy forest",
        reference_image="data:image/png;base64,AA==",
        width=512,
        height=512,
        num_inference_steps=16,
    )

    assert request.reference_image == "data:image/png;base64,AA=="


def test_text_only_request_keeps_reference_image_optional():
    request = ImageGenerateRequest(prompt="A black bird", num_inference_steps=16)

    assert request.reference_image is None
    assert request.reference_images is None


@pytest.mark.parametrize("count", [1, 2, 10])
def test_ordered_reference_images_are_accepted(count):
    values = [_image_payload(index) for index in range(count)]
    request = ImageGenerateRequest(prompt="compose the references", reference_images=values)
    assert request.reference_images == values
    assert request.model_dump()["reference_images"] == values


@pytest.mark.parametrize("values", [[], ["a"] * 11, [""], ["https://example.com/image.png"]])
def test_invalid_reference_lists_are_rejected(values):
    with pytest.raises(ValidationError):
        ImageGenerateRequest(prompt="compose", reference_images=values)


def test_single_and_multi_reference_fields_are_mutually_exclusive():
    with pytest.raises(ValidationError):
        ImageGenerateRequest(prompt="compose", reference_image="a", reference_images=["b"])


def test_image_to_image_and_multi_reference_fields_are_mutually_exclusive():
    with pytest.raises(ValidationError):
        ImageToImageRequest(prompt="compose", image="a", reference_images=["b"])


@pytest.mark.parametrize("value", ["invalid$base64", "AA==", "data:text/plain;base64,AA=="])
def test_corrupt_image_payload_is_rejected_before_enqueue(value):
    with pytest.raises(ValidationError, match="reference_images"):
        ImageGenerateRequest(prompt="compose", reference_images=[value])


def test_too_small_multi_reference_image_is_rejected():
    with pytest.raises(ValidationError, match="16 pixels"):
        ImageGenerateRequest(prompt="compose", reference_images=[_image_payload(size=(15, 64)), _image_payload()])


def test_payload_whitespace_and_padding_are_normalised_for_worker():
    payload = _image_payload().split(",", 1)[1]
    request = ImageGenerateRequest(prompt="compose", reference_images=[" \n" + payload.rstrip("=") + "\n"])
    assert request.reference_images == [payload]


def test_other_runners_do_not_silently_ignore_reference_images(monkeypatch):
    monkeypatch.setattr("domain.image_generate_request.get_settings",
                        lambda: SimpleNamespace(model_runner="tt-sdxl"))
    with pytest.raises(ValidationError, match="only supported by the FLUX.3 runner"):
        ImageGenerateRequest(prompt="compose", reference_images=[_image_payload()])
