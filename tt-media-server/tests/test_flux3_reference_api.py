from domain.image_generate_request import ImageGenerateRequest


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
