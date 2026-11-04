# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import base64
import binascii
import io
import json
import os
import zipfile

from config.constants import ModelRunners
from config.settings import settings
from domain.image_edit_request import ImageEditRequest
from domain.image_generate_request import ImageGenerateRequest
from domain.image_to_image_request import ImageToImageRequest
from fastapi import APIRouter, Depends, File, Form, HTTPException, Security, UploadFile
from fastapi.responses import JSONResponse, Response
from model_services.base_service import BaseService
from model_services.image_service import ImageService
from resolver.service_resolver import service_resolver
from security.api_key_checker import get_api_key
from config.constants import JobTypes

_LORA_DIR = os.environ.get("LORA_DIR", "/loras")
_LORA_STATE = os.path.join(_LORA_DIR, "active_lora.json")
# FLUX.1 LoRAs run ~20-300 MB; the cap stops a stray upload from filling the
# container disk. Raise LORA_MAX_BYTES for unusually large adapters.
_LORA_MAX_BYTES = int(os.environ.get("LORA_MAX_BYTES", 512 * 1024 * 1024))
_LORA_CHUNK_BYTES = 1024 * 1024
# Upper bound on the LoRA fusion weight, matching ImageGenerateRequest.lora_scale.
_LORA_MAX_SCALE = 2.0
# safetensors caps its JSON header at 100 MB; anything larger is not a real file.
_SAFETENSORS_MAX_HEADER_BYTES = 100 * 1024 * 1024


def _image_bytes_from_base64(payload: str) -> bytes:
    """Decode a runner image payload for ``response_format=file``.

    Runners return raw Base64 today, but accepting a data URL here keeps the
    HTTP boundary tolerant of both representations. Whitespace is removed
    because ImageManager historically uses ``base64.encodebytes``.
    """
    if payload.startswith("data:"):
        try:
            payload = payload.split(",", 1)[1]
        except IndexError as e:
            raise ValueError("invalid image data URL") from e
    try:
        return base64.b64decode("".join(payload.split()), validate=True)
    except (ValueError, binascii.Error) as e:
        raise ValueError("runner returned invalid Base64 image data") from e


def _image_media_type(request: ImageGenerateRequest) -> tuple[str, str]:
    image_format = (request.image_return_format or "JPEG").upper()
    if image_format == "PNG":
        return "image/png", "png"
    if image_format == "JPEG":
        return "image/jpeg", "jpg"
    raise HTTPException(
        status_code=422,
        detail="image_return_format must be 'JPEG' or 'PNG' for file responses",
    )


def _build_image_response(result: list[str], request: ImageGenerateRequest, elapsed: float | None = None):
    """Return legacy JSON or direct image bytes according to ``response_format``.

    One image is returned as its native media type. Multiple images are packed
    into a ZIP because one HTTP response cannot have multiple image media types.
    """
    content = {"images": result}
    if elapsed is not None:
        content["generation_time"] = elapsed
    if request.response_format != "file":
        return JSONResponse(content=content)

    if not result:
        raise HTTPException(status_code=500, detail="image generation returned no images")
    try:
        image_bytes = [_image_bytes_from_base64(item) for item in result]
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    media_type, extension = _image_media_type(request)
    if len(image_bytes) == 1:
        return Response(
            content=image_bytes[0],
            media_type=media_type,
            headers={"Content-Disposition": f'attachment; filename="image.{extension}"'},
        )

    archive = io.BytesIO()
    with zipfile.ZipFile(archive, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for index, data in enumerate(image_bytes, start=1):
            zf.writestr(f"image_{index}.{extension}", data)
    return Response(
        content=archive.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="images.zip"'},
    )


def _check_safetensors(path: str) -> None:
    """Confirm the upload really is a safetensors file. The format is an 8-byte
    little-endian header length followed by that many bytes of JSON metadata, so
    this reads only the header — never the tensor payload."""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        raw_len = f.read(8)
        if len(raw_len) < 8:
            raise ValueError("file is too small to be a safetensors file")
        header_len = int.from_bytes(raw_len, "little")
        if header_len == 0 or header_len > _SAFETENSORS_MAX_HEADER_BYTES:
            raise ValueError("safetensors header length is out of range")
        if 8 + header_len > size:
            raise ValueError("safetensors header length exceeds the file size")
        try:
            header = json.loads(f.read(header_len))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise ValueError("safetensors header is not valid JSON") from e
    if not isinstance(header, dict):
        raise ValueError("safetensors header is not a JSON object")


async def _save_capped(upload: UploadFile, dest: str) -> int:
    """Stream the upload to dest, aborting once it exceeds _LORA_MAX_BYTES.
    Streaming (rather than upload.read() in one shot) keeps an oversized file
    from being buffered in full before the limit is noticed."""
    written = 0
    with open(dest, "wb") as f:
        while chunk := await upload.read(_LORA_CHUNK_BYTES):
            written += len(chunk)
            if written > _LORA_MAX_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"LoRA exceeds the {_LORA_MAX_BYTES} byte limit",
                )
            f.write(chunk)
    return written


generate_image_router = APIRouter()


@generate_image_router.post("/generations")
async def generate_image(
    image_generate_request: ImageGenerateRequest,
    service: BaseService = Depends(service_resolver),
    api_key: str = Security(get_api_key),
):
    """
    Generate an image based on the provided request.

    Returns:
        JSONResponse: The generated images as a list of base64 strings.

    Raises:
        HTTPException: If image generation fails.
    """
    try:
        import time as _time

        _t0 = _time.time()
        result = await service.process_request(image_generate_request)
        _elapsed = round(_time.time() - _t0, 2)
        return _build_image_response(result, image_generate_request, _elapsed)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@generate_image_router.post("/generations/async", status_code=202)
async def submit_async_image(
    image_generate_request: ImageGenerateRequest,
    service: ImageService = Depends(service_resolver),
    api_key: str = Security(get_api_key),
):
    try:
        return JSONResponse(content=await service.create_image_job(image_generate_request), status_code=202)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@generate_image_router.get("/generations/{job_id}")
def get_image_job(job_id: str, service: ImageService = Depends(service_resolver), api_key: str = Security(get_api_key)):
    data = service.image_job_metadata(job_id)
    if data is None or data.get("job_type") != JobTypes.IMAGE.value:
        raise HTTPException(status_code=404, detail="Image job not found")
    if data.get("status") == "completed":
        result = service.read_image_job_result(job_id)
        if result is not None:
            data["images"] = result["images"]
    return JSONResponse(content=data)


@generate_image_router.post("/generations/{job_id}/cancel")
def cancel_image_job(job_id: str, service: ImageService = Depends(service_resolver), api_key: str = Security(get_api_key)):
    data = service.cancel_image_job(job_id)
    if data is None or data.get("job_type") != JobTypes.IMAGE.value:
        raise HTTPException(status_code=404, detail="Image job not found")
    return JSONResponse(content=data)


@generate_image_router.delete("/generations/{job_id}")
async def delete_image_job(job_id: str, service: ImageService = Depends(service_resolver), api_key: str = Security(get_api_key)):
    try:
        deleted = await service.delete_image_job(job_id)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    if not deleted:
        raise HTTPException(status_code=404, detail="Image job not found")
    return JSONResponse(content={"id": job_id, "object": JobTypes.IMAGE.value, "deleted": True})


image_to_image_router = APIRouter()


@image_to_image_router.post("/image-to-image")
async def image_to_image(
    image_to_image_request: ImageToImageRequest,
    service: BaseService = Depends(service_resolver),
    api_key: str = Security(get_api_key),
):
    """
    Generate an image based on the provided request.
    Returns:
        JSONResponse: The generated images as a list of base64 strings.
    Raises:
        HTTPException: If image generation fails.
    """
    try:
        result = await service.process_request(image_to_image_request)
        return _build_image_response(result, image_to_image_request)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


edit_image_router = APIRouter()


@edit_image_router.post("/edits")
async def edit_image(
    image_edit_request: ImageEditRequest,
    service: BaseService = Depends(service_resolver),
    api_key: str = Security(get_api_key),
):
    """
    Edit an image based on the provided request.
    Returns:
        JSONResponse: The edited images as a list of base64 strings.
    Raises:
        HTTPException: If image editing fails.
    """
    try:
        result = await service.process_request(image_edit_request)
        return _build_image_response(result, image_edit_request)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------- FLUX.1-Kontext-dev -----------------------------
# Kontext reuses the shared image endpoints — /generations (text->image) and
# /edits (instruction edit; mask is optional and ignored by Kontext) — so it does
# not redefine its own copies. The only Kontext-specific surface is LoRA
# management (/lora/*): upload a FLUX.1 LoRA, record it as active, and rebuild the
# worker with the LoRA fused in. (Optional JP->EN prompt translation ships as a
# web-console add-on in a separate repo, not here.)
lora_router = APIRouter()


@lora_router.post("/lora/apply")
async def lora_apply(
    file: UploadFile = File(...),
    scale: float = Form(1.0),
    name: str = Form(None),
    service: BaseService = Depends(service_resolver),
    api_key: str = Security(get_api_key),
):
    """Upload a FLUX.1 LoRA and activate it. Saves the file, records it as the
    active LoRA, and restarts the device worker so the pipeline is rebuilt with
    the LoRA fused into the transformer. Returns immediately with status
    'rebuilding'; poll /v1/images/lora/status until model_ready is true again (~2-3 min)."""
    # The client-supplied name is kept only as a display label — it never enters
    # a filesystem path. The upload is written to a fixed, server-controlled
    # filename so no user-controlled data is used in a path expression (avoids
    # path traversal; satisfies CodeQL).
    display_name = os.path.basename(name or file.filename or "lora.safetensors")

    # Reject bad metadata before touching the disk.
    if not display_name.endswith(".safetensors"):
        raise HTTPException(status_code=400, detail="LoRA must be a .safetensors file")
    if not 0.0 <= scale <= _LORA_MAX_SCALE:
        raise HTTPException(
            status_code=422,
            detail=f"scale must be between 0.0 and {_LORA_MAX_SCALE}, got {scale}",
        )

    up = os.path.join(_LORA_DIR, "uploaded")
    dest = os.path.join(up, "active_lora.safetensors")
    # Stage under .part and swap only once the file validates, so a rejected
    # upload leaves the currently active LoRA in place.
    staged = dest + ".part"
    try:
        os.makedirs(up, exist_ok=True)
        await _save_capped(file, staged)
        try:
            _check_safetensors(staged)
        except ValueError as e:
            raise HTTPException(
                status_code=400, detail=f"invalid safetensors file: {e}"
            )
        os.replace(staged, dest)
        with open(_LORA_STATE, "w") as f:
            json.dump({"path": dest, "scale": float(scale), "name": display_name}, f)
        await service.deep_reset()  # restart worker -> rebuild pipeline with LoRA
        return JSONResponse(
            content={
                "status": "rebuilding",
                "active": {"name": display_name, "scale": float(scale)},
            }
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.isfile(staged):
            os.remove(staged)


@lora_router.post("/lora/clear")
async def lora_clear(
    service: BaseService = Depends(service_resolver),
    api_key: str = Security(get_api_key),
):
    """Deactivate any LoRA and rebuild the base pipeline (~2-3 min)."""
    try:
        if os.path.isfile(_LORA_STATE):
            os.remove(_LORA_STATE)
        await service.deep_reset()
        return JSONResponse(content={"status": "rebuilding"})
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@lora_router.get("/lora/status")
async def lora_status(
    service: BaseService = Depends(service_resolver),
    api_key: str = Security(get_api_key),
):
    """Current LoRA + whether the worker has finished (re)building."""
    active = None
    if os.path.isfile(_LORA_STATE):
        try:
            d = json.load(open(_LORA_STATE))
            active = {"name": d.get("name"), "scale": d.get("scale")}
        except Exception:
            active = None
    try:
        ready = bool(service.check_is_model_ready().get("model_ready"))
    except Exception:
        ready = False
    return JSONResponse(content={"active": active, "model_ready": ready})


router = APIRouter()

if settings.model_runner == ModelRunners.TT_SDXL_IMAGE_TO_IMAGE.value:
    router.include_router(image_to_image_router)
elif settings.model_runner == ModelRunners.TT_SDXL_EDIT.value:
    router.include_router(edit_image_router)
elif settings.model_runner == ModelRunners.TT_FLUX_1_KONTEXT_DEV.value:
    # Reuse the shared endpoints for text->image (/generations) and instruction
    # edit (/edits, mask optional); add only the Kontext-specific LoRA routes.
    router.include_router(generate_image_router)
    router.include_router(edit_image_router)
    router.include_router(lora_router)
else:
    router.include_router(generate_image_router)
