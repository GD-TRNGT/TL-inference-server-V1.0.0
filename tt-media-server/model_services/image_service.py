# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import json
import os

import numpy as np
from config.constants import JobTypes
from config.settings import settings
from domain.image_generate_request import ImageGenerateRequest
from model_services.base_service import BaseService
from model_services.cpu_workload_handler import CpuWorkloadHandler
from PIL import Image
from telemetry.image_request_metrics import observe_image_request
from utils.job_manager import get_job_manager


def create_image_worker_context():
    from utils.image_manager import ImageManager

    return ImageManager("img")


def image_worker_function(image_manager, image_data, input_request=None):
    return image_manager.images_to_base64_list(image_data, input_request)


class ImageService(BaseService):
    def __init__(self):
        super().__init__()
        self._job_manager = get_job_manager()
        self._job_output_dir = os.environ.get(
            "IMAGE_JOB_OUTPUT_DIR", "/tmp/tt-media-server/image-jobs"
        )

        warmup_task_data = [Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))]
        self._cpu_workload_handler = CpuWorkloadHandler(
            name="ImagePostprocessing",
            worker_count=self.scheduler.get_worker_count(),
            worker_function=image_worker_function,
            worker_context_setup=create_image_worker_context,
            warmup_task_data=warmup_task_data,
        )

    async def create_image_job(self, request: ImageGenerateRequest) -> dict:
        """Submit an image generation job using the shared JobManager."""
        async def run_and_persist(job_request):
            images = await self.process_request(job_request)
            os.makedirs(self._job_output_dir, exist_ok=True)
            path = os.path.join(self._job_output_dir, f"{job_request._task_id}.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump({"images": images}, handle)
            return path

        return await self._job_manager.create_job(
            job_id=request._task_id,
            job_type=JobTypes.IMAGE,
            model=os.environ.get("SERVED_MODEL_NAME") or settings.model_weights_path,
            request=request,
            task_function=run_and_persist,
            request_parameters=request.model_dump(mode="json"),
        )

    def image_job_metadata(self, job_id: str):
        return self._job_manager.get_job_metadata(job_id)

    def read_image_job_result(self, job_id: str):
        path = self._job_manager.get_job_result_path(job_id)
        if not path or not os.path.isfile(path):
            return None
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def cancel_image_job(self, job_id: str):
        return self._job_manager.cancel_job(job_id)

    async def delete_image_job(self, job_id: str):
        return await self._job_manager.delete_job(job_id)

    async def pre_process(self, request: ImageGenerateRequest):
        """Set up segments for multi-image generation"""
        # Record the requested shape here, before segmentation: a multi-image
        # request is fanned out into one request per image below, so recording
        # further down would count one client request several times and
        # misreport both the arrival rate and the batch-size distribution.
        observe_image_request(request, settings.model_runner)
        if request.number_of_images > 1:
            # Create segments list for parallel processing
            request._segments = list(range(request.number_of_images))
        return request

    async def post_process(self, result, input_request: ImageGenerateRequest):
        """Asynchronous postprocessing using queue-based workers"""
        try:
            image_file = await self._cpu_workload_handler.execute_task(
                result, input_request
            )
        except Exception as e:
            self.logger.error(f"Image postprocessing failed: {e}")
            raise
        return image_file

    def create_segment_request(
        self, original_request: ImageGenerateRequest, segment, segment_index: int
    ) -> ImageGenerateRequest:
        """Create a request for generating a single image with incremented seed"""
        field_values = original_request.model_dump()
        new_request = type(original_request)(**field_values)

        new_request.number_of_images = 1

        # Increment seed for each image if seed is specified
        if original_request.seed is not None:
            new_request.seed = original_request.seed + segment_index

        return new_request

    def combine_results(self, results):
        """Flatten list of image results into a single list"""
        flattened_results = []
        for result in results:
            if isinstance(result, list):
                flattened_results.extend(result)
            else:
                flattened_results.append(result)
        return flattened_results

    def stop_workers(self):
        self.logger.info("Shutting down image postprocessing workers")
        self._cpu_workload_handler.stop_workers()

        return super().stop_workers()
