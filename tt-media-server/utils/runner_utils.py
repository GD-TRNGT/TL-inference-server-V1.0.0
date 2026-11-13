# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import os

from config.constants import DeviceTypes, ModelRunners
from config.settings import settings
from telemetry.telemetry_client import get_telemetry_client

from utils.logger import TTLogger
from utils.torch_utils import set_torch_thread_limits

_BH_DEVICE_MESH_DESCRIPTORS = {
    "p150": "p150_mesh_graph_descriptor.textproto",
    "p150x4": "p150x4_mesh_graph_descriptor.textproto",
    "p150x8": "p150x8_mesh_graph_descriptor.textproto",
    "p300": "p150_mesh_graph_descriptor.textproto",
    "p300x2": "p150_mesh_graph_descriptor.textproto",
    "p100": "p100_mesh_graph_descriptor.textproto",
}


def setup_runner_environment(
    device_id: str, cpu_threads: str = "2", num_torch_threads: int = 1
):
    """Set up environment variables and configuration for a device runner"""
    _logger = TTLogger()
    _logger.info(
        f"setup_runner_environment: device_id={device_id!r}, "
        f"is_galaxy={settings.is_galaxy}, "
        f"device_mesh_shape={settings.device_mesh_shape}, "
        f"model_runner={settings.model_runner!r}"
    )
    setup_cpu_threading_limits(cpu_threads, num_torch_threads)

    flux3_sp4_auto_worker = (
        settings.model_runner == ModelRunners.TT_FLUX_3_IMAGE_KLEIN.value
        and os.environ.get("FLUX3_EXECUTION_PROFILE", "").strip().lower()
        in {"sp4", "sp-4"}
        and device_id == "sp4-auto"
    )
    if flux3_sp4_auto_worker:
        # A four-chip Galaxy Ring is selected by the fabric topology. Exporting
        # four visible device IDs here would hide that topology from the solver.
        os.environ.pop("TT_VISIBLE_DEVICES", None)
        _logger.info("setup_runner_environment: SP4 uses topology-selected Galaxy chips")
    elif device_id:
        os.environ["TT_VISIBLE_DEVICES"] = str(device_id)
        _logger.info(f"setup_runner_environment: TT_VISIBLE_DEVICES={device_id}")

    if (
        device_id
        and device_id != "-1"
        and not flux3_sp4_auto_worker
        and settings.model_runner == ModelRunners.TT_FLUX_3_IMAGE_KLEIN.value
    ):
        cache_root = os.environ.get("FLUX3_TT_METAL_CACHE_ROOT")
        if not cache_root:
            cache_root = os.path.join(
                os.environ.get("CACHE_ROOT", os.path.expanduser("~/.cache")),
                "flux3-tt-metal-workers",
            )
        worker_name = str(device_id).replace(",", "-").replace("(", "").replace(")", "")
        os.makedirs(os.path.join(cache_root, worker_name), exist_ok=True)
        os.environ["TT_METAL_CACHE"] = os.path.join(cache_root, worker_name)
    if settings.enable_telemetry:
        get_telemetry_client()

    tt_metal_home = os.environ.get("TT_METAL_HOME", "")

    _RUNNERS_REQUIRING_MESH_DESCRIPTOR = {
        ModelRunners.TT_WHISPER.value,
        ModelRunners.TT_SPEECHT5_TTS.value,
        ModelRunners.TT_FLUX_3_IMAGE_KLEIN.value,
        # Training runners init a tt-metal device directly (via torch-xla or
        # tt-crank). On a single Blackhole chip of a P300 board, tt-metal
        # otherwise selects the CUSTOM cluster type and asserts that a fabric
        # mesh graph descriptor path is set. Providing the BH mesh descriptor
        # avoids that.
        ModelRunners.TRAINING_LORA.value,
        ModelRunners.TRAINER_TRAINING_LORA.value,
    }
    if settings.model_runner in _RUNNERS_REQUIRING_MESH_DESCRIPTOR:
        if settings.is_galaxy:
            _logger.info("setup_runner_environment: applying galaxy mesh config")
            _setup_galaxy_mesh_config(tt_metal_home)
        elif (settings.device or "").lower() in _BH_DEVICE_MESH_DESCRIPTORS:
            _logger.info(
                f"setup_runner_environment: applying blackhole mesh config for device={settings.device!r}"
            )
            _setup_blackhole_mesh_config(tt_metal_home)

    _RUNNERS_REQUIRING_GRID_OVERRIDE = {
        ModelRunners.TT_SDXL_TRACE.value,
        ModelRunners.TT_SDXL_EDIT.value,
        ModelRunners.TT_SDXL_IMAGE_TO_IMAGE.value,
    }
    if settings.model_runner in _RUNNERS_REQUIRING_GRID_OVERRIDE:
        _setup_grid_override(settings.device)


def setup_cpu_threading_limits(cpu_threads: str, num_torch_threads: int = 1):
    """Set up CPU threading limits for PyTorch to prevent CPU oversubscription"""
    os.environ["OMP_NUM_THREADS"] = cpu_threads
    os.environ["MKL_NUM_THREADS"] = cpu_threads
    os.environ["TORCH_NUM_THREADS"] = str(num_torch_threads)
    set_torch_thread_limits(num_threads=num_torch_threads)
    if settings.default_throttle_level:
        os.environ["TT_MM_THROTTLE_PERF"] = settings.default_throttle_level


def _setup_blackhole_mesh_config(tt_metal_home: str):
    """Configure mesh graph descriptors for Blackhole hardware"""
    device = (settings.device or "").lower()
    descriptor = _BH_DEVICE_MESH_DESCRIPTORS.get(device)
    if descriptor:
        os.environ["TT_MESH_GRAPH_DESC_PATH"] = (
            f"{tt_metal_home}/tt_metal/fabric/mesh_graph_descriptors/{descriptor}"
        )


def _setup_galaxy_mesh_config(tt_metal_home: str):
    """Configure mesh graph descriptors for Galaxy hardware"""
    flux3_parallelism = os.environ.get("FLUX3_PARALLELISM", "").strip().lower()
    flux3_profile = os.environ.get("FLUX3_EXECUTION_PROFILE", "auto").strip().lower()
    is_flux3_sp8tp4 = (
        flux3_profile in {"sp8tp4", "sp8-tp4"}
        or (flux3_profile in {"", "auto"} and flux3_parallelism == "sp8tp4")
    )
    is_flux3_sequence_parallel = (
        flux3_profile in {"sp2", "sp4", "sp-4", "sp8tp4", "sp8-tp4"}
        or (flux3_profile in {"", "auto"} and flux3_parallelism in {"sp", "sp4", "sp8tp4"})
    )
    if (
        settings.model_runner == ModelRunners.TT_FLUX_3_IMAGE_KLEIN.value
        and is_flux3_sp8tp4
    ):
        # The upstream FLUX.3 Klein SP8xTP4 tables target a 12x9 worker grid.
        # ``11,9`` exposes a 12x10 physical grid and leaves its last row for
        # CCL, avoiding the 8x8-grid Ring-SDPA L1 circular-buffer overlap.
        os.environ["TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE"] = "11,9"
    elif not is_flux3_sequence_parallel:
        os.environ["TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE"] = "7,7"

    # SP2 requires the explicit two-chip Galaxy descriptor mounted by the
    # deployment.  Do not replace it with the single-board N300 descriptor.
    if is_flux3_sequence_parallel:
        explicit_descriptor = os.environ.get("TT_MESH_GRAPH_DESC_PATH")
        if explicit_descriptor:
            TTLogger().info(
                "Keeping explicit FLUX.3 SP mesh descriptor: "
                f"{explicit_descriptor}"
            )
            return

    mesh_descriptors = {
        (1, 1): "n150_mesh_graph_descriptor.textproto",
        (2, 1): "n300_mesh_graph_descriptor.textproto",
        (1, 4): "n300_2x2_mesh_graph_descriptor.textproto",
        # Each four-chip worker is a topology-local 2x2 N300 group, exposed
        # logically as a 4x1 sequence-parallel Ring.
        (4, 1): "n300_2x2_mesh_graph_descriptor.textproto",
        (2, 4): "t3k_mesh_graph_descriptor.textproto",
    }

    descriptor = mesh_descriptors.get(settings.device_mesh_shape)
    if descriptor:
        os.environ["TT_MESH_GRAPH_DESC_PATH"] = (
            f"{tt_metal_home}/tt_metal/fabric/mesh_graph_descriptors/{descriptor}"
        )


def _setup_grid_override(device: str):
    _logger = TTLogger()
    if (
        device == DeviceTypes.BLACKHOLE_GALAXY.value
        or device == DeviceTypes.GALAXY.value
    ):
        expected = "7,7" if device == DeviceTypes.GALAXY.value else "10,9"
        os.environ["TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE"] = expected
        _logger.info(
            f"Set TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE={expected} for {device}"
        )
