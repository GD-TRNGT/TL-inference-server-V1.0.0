# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import base64
import binascii
import io
import warnings
from typing import List, Literal, Optional, Tuple, Union

from PIL import Image

from config.settings import get_settings
from config.flux3_policy import (
    FLUX3_MAX_RESOLUTION as _FLUX3_MAX_RESOLUTION,
    FLUX3_MIN_RESOLUTION as _FLUX3_MIN_RESOLUTION,
    FLUX3_RUNNER as _FLUX3_RUNNER,
    validate_flux3_geometry,
)
from domain.base_request import BaseRequest
from pydantic import Field, PrivateAttr, field_validator, model_validator

# Flux models support fewer inference steps than other image models
_FLUX_RUNNERS = {"tt-flux.1-dev", "tt-flux.1-schnell", "tt-flux.1-kontext-dev"}
_DEFAULT_MAX_RESOLUTION = 1536
# The official FLUX.3 reference accepts any output whose two sides are a
# multiple of the VAE downsampling ratio. The smallest non-zero valid image is
# therefore 16x16. Its model card advertises an upper bound of 4K per side.
_FLUX3_SHAPE_MULTIPLE = 16
_FLUX_MIN_INFERENCE_STEPS = 4
_DEFAULT_MIN_INFERENCE_STEPS = 12
_SKIP_STEP_VALIDATION_RUNNERS = {"tt-z-image-turbo"}


class BaseImageRequest(BaseRequest):
    prompt: str
    negative_prompt: Optional[str] = None
    # FLUX.3 accepts a Base64 string or data URL here.  Other image runners
    # ignore the field unless they explicitly implement reference conditioning.
    reference_image: Optional[str] = Field(default=None)
    # Ordered references: FLUX.3 permits up to ten images per request.
    reference_images: Optional[List[str]] = Field(default=None, min_length=1, max_length=10)
    num_inference_steps: Optional[int] = Field(default=20, ge=4, le=50)
    seed: Optional[int] = None
    number_of_images: Optional[int] = Field(default=1, ge=1, le=4)

    # Image output settings
    image_return_format: Optional[str] = Field(default="JPEG")
    image_quality: Optional[int] = Field(default=85, ge=50, le=100)
    # Keep the existing JSON/Base64 response as the default. ``file`` returns
    # the generated image bytes directly from the HTTP endpoint.
    response_format: Literal["b64_json", "file"] = Field(default="b64_json")

    # Optional output resolution (width/height). Runners that support per-request
    # resolution (e.g. FLUX.1-Kontext) honor these; others ignore them.
    width: Optional[int] = Field(default=None, ge=_FLUX3_MIN_RESOLUTION)
    height: Optional[int] = Field(default=None, ge=_FLUX3_MIN_RESOLUTION)

    # Private fields for internal processing
    _segments: Optional[List[int]] = PrivateAttr(default=None)

    @field_validator("num_inference_steps")
    @classmethod
    def validate_num_inference_steps(cls, v):
        if v is None:
            return v
        model_runner = get_settings().model_runner
        if model_runner in _SKIP_STEP_VALIDATION_RUNNERS:
            return v
        min_steps = (
            _FLUX_MIN_INFERENCE_STEPS
            if model_runner in _FLUX_RUNNERS
            else _DEFAULT_MIN_INFERENCE_STEPS
        )
        if v < min_steps:
            raise ValueError(
                f"num_inference_steps must be >= {min_steps} for {model_runner}, got {v}"
            )
        return v

    @model_validator(mode="after")
    def validate_resolution_pair(self):
        # Both or neither: a lone width would pair with the runner's default
        # height (and vice versa), silently changing the aspect ratio.
        if (self.width is None) != (self.height is None):
            raise ValueError("width and height must be provided together")
        model_runner = get_settings().model_runner
        # Keep the established 1536 limit for other image runners. FLUX.3
        # follows the official reference contract: up to 4K per side, with
        # each side aligned to the VAE/DiT 16-pixel downsampling ratio. The
        # reference implementation has no separate output-pixel or aspect-ratio
        # cap, so do not impose one here.
        max_resolution = (
            _FLUX3_MAX_RESOLUTION if model_runner == _FLUX3_RUNNER else _DEFAULT_MAX_RESOLUTION
        )
        for name, value in (("width", self.width), ("height", self.height)):
            if value is not None and value > max_resolution:
                raise ValueError(
                    f"{name} must be <= {max_resolution} for "
                    f"{model_runner}, got {value}"
                )
        if model_runner == _FLUX3_RUNNER and self.width is not None:
            validate_flux3_geometry(
                self.width,
                self.height,
                model_runner=model_runner,
                mesh_shape=tuple(
                    getattr(get_settings(), "device_mesh_shape", (1, 1)) or (1, 1)
                ),
            )
        return self

    @model_validator(mode="after")
    def validate_reference_inputs(self):
        if self.reference_images is not None:
            if get_settings().model_runner != "tt-flux.3-image-klein":
                raise ValueError("reference_images is only supported by the FLUX.3 runner")
            if self.reference_image is not None or getattr(self, "image", None) is not None:
                raise ValueError("provide reference_images, reference_image, or image, not more than one")
            if any(not item.strip() for item in self.reference_images):
                raise ValueError("reference_images must not contain empty items")
            if any(item.lower().startswith(("http://", "https://")) for item in self.reference_images):
                raise ValueError("reference_images must be Base64 or data URLs, not HTTP URLs")
            for index, payload in enumerate(self.reference_images):
                encoded = payload.strip()
                normalised_prefix = ""
                if encoded.lower().startswith("data:"):
                    prefix, separator, encoded = encoded.partition(",")
                    if not separator or not prefix.lower().startswith("data:image/") or not prefix.lower().endswith(";base64"):
                        raise ValueError(f"reference_images[{index}] must be an image Base64 data URL")
                    normalised_prefix = prefix.lower() + ","
                try:
                    encoded = "".join(encoded.split())
                    encoded += "=" * (-len(encoded) % 4)
                    content = base64.b64decode(encoded, validate=True)
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", Image.DecompressionBombWarning)
                        with Image.open(io.BytesIO(content)) as image:
                            width, height = image.size
                            image.verify()
                except (binascii.Error, OSError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
                    raise ValueError(f"reference_images[{index}] is not a valid Base64 image") from error
                self.reference_images[index] = normalised_prefix + encoded
                if len(self.reference_images) > 1:
                    scale = min(1.0, math.sqrt((1024 * 1024) / (width * height)))
                    if int(width * scale) < 16 or int(height * scale) < 16:
                        raise ValueError(f"reference_images[{index}] must retain at least 16 pixels per side after multi-reference resizing")
        return self

    @field_validator("image_return_format")
    @classmethod
    def validate_image_return_format(cls, v):
        if v is not None and v not in ["JPEG", "PNG"]:
            raise ValueError("image_return_format must be 'JPEG' or 'PNG'")
        return v

    def update_object(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


class ImageGenerateRequest(BaseImageRequest):
    guidance_scale: float = Field(default=5.0, ge=1.0, le=20.0)
    prompt_2: Optional[str] = None
    negative_prompt_2: Optional[str] = None
    crop_coords_top_left: Optional[Tuple[int, float]] = Field(default=(0, 0))
    guidance_rescale: Optional[float] = Field(default=0.0, ge=0.0, le=1.0)
    timesteps: Optional[List[Union[int, float]]] = None
    sigmas: Optional[List[Union[int, float]]] = None

    # LoRA adapter settings
    lora_path: Optional[str] = Field(default=None)
    lora_scale: Optional[float] = Field(default=0.5, ge=0.0, le=2.0)
