# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import os

from config.constants import DeviceTypes, ModelRunners
from config.flux3_profiles import get_flux3_descriptor, resolve_flux3_profile
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
        f"is_galaxy={getattr(settings, 'is_galaxy', False)}, "
        f"device_mesh_shape={getattr(settings, 'device_mesh_shape', (1, 1))}, "
        f"model_runner={getattr(settings, 'model_runner', None)!r}"
    )
    setup_cpu_threading_limits(cpu_threads, num_torch_threads)

    if device_id:
        visible_device_id = str(device_id)
        submesh_marker = "::submesh="
        if submesh_marker in visible_device_id:
            visible_device_id, _, submesh_index = visible_device_id.partition(
                submesh_marker
            )
            os.environ["FLUX3_SUBMESH_INDEX"] = submesh_index
            _logger.info(
                f"setup_runner_environment: FLUX3_SUBMESH_INDEX={submesh_index}"
            )
        os.environ["TT_VISIBLE_DEVICES"] = visible_device_id
        _logger.info(
            f"setup_runner_environment: TT_VISIBLE_DEVICES={visible_device_id}"
        )

        if (
            getattr(settings, "model_runner", None) == ModelRunners.TT_FLUX_3_IMAGE_KLEIN.value
            and getattr(settings, "is_galaxy", False)
        ):
            # Every worker in a Galaxy container is a separate Python
            # process.  TT-Metal Inspector RPC defaults to localhost:50051;
            # leaving it enabled makes all workers contend for that one
            # socket and can crash an MPI child during startup.  Keep the
            # diagnostic path opt-in, and isolate the filesystem logs even
            # when Inspector itself is enabled.
            if os.environ.get("FLUX3_ENABLE_INSPECTOR_RPC", "0") != "1":
                os.environ["TT_METAL_INSPECTOR_RPC"] = "0"
                _logger.info(
                    "setup_runner_environment: disabled TT-Metal Inspector RPC "
                    "for multi-process FLUX.3 Galaxy workers"
                )

            worker_name = (
                visible_device_id.replace(",", "-")
                .replace("(", "")
                .replace(")", "")
            )
            if submesh_marker in str(device_id):
                worker_name += f"_submesh{submesh_index}"
            logs_root = os.environ.get("TT_METAL_LOGS_PATH")
            if logs_root:
                worker_logs = os.path.join(logs_root, "flux3", worker_name)
                os.makedirs(worker_logs, exist_ok=True)
                os.environ["TT_METAL_LOGS_PATH"] = worker_logs
                _logger.info(
                    f"setup_runner_environment: TT_METAL_LOGS_PATH={worker_logs}"
                )

    if (
        device_id
        and device_id != "-1"
        and getattr(settings, "model_runner", None) == ModelRunners.TT_FLUX_3_IMAGE_KLEIN.value
    ):
        # Keep one private JIT cache per worker, matching the SD3.5 runner.
        # The parent-mesh worker id contains ``::submesh=N`` for scheduler
        # bookkeeping; never put that marker literally into the cache path.
        # TT-Metal's fabric JIT creates and atomically renames many files while
        # the router program is compiled. The stable ``TT_METAL_HOME/built``
        # layout is the upstream/reference layout and avoids the failed rename
        # seen with the ad-hoc ``/tmp/...::submesh=N`` path.
        cache_root = os.environ.get("FLUX3_TT_METAL_CACHE_ROOT")
        if not cache_root:
            cache_root = os.path.join(os.environ.get("TT_METAL_HOME", ""), "built")
        if not cache_root or cache_root == "built":
            cache_root = os.path.join(
                os.environ.get("CACHE_ROOT", os.path.expanduser("~/.cache")),
                "tt-metal-built",
            )
        worker_name = visible_device_id.replace(",", "-").replace("(", "").replace(")", "")
        if submesh_marker in str(device_id):
            worker_name += f"_submesh{submesh_index}"
        worker_cache = os.path.join(cache_root, worker_name)
        os.makedirs(worker_cache, exist_ok=True)
        os.environ["TT_METAL_CACHE"] = worker_cache
        _logger.info(f"setup_runner_environment: TT_METAL_CACHE={worker_cache}")
    if getattr(settings, "enable_telemetry", False):
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
    if getattr(settings, "model_runner", None) in _RUNNERS_REQUIRING_MESH_DESCRIPTOR:
        is_flux3_galaxy_worker = (
            getattr(settings, "model_runner", None) == ModelRunners.TT_FLUX_3_IMAGE_KLEIN.value
            and (getattr(settings, "device", "") or "").lower()
            == DeviceTypes.BLACKHOLE_GALAXY.value
        )
        if getattr(settings, "is_galaxy", False) or is_flux3_galaxy_worker:
            _logger.info("setup_runner_environment: applying galaxy mesh config")
            _setup_galaxy_mesh_config(tt_metal_home)
        elif (getattr(settings, "device", "") or "").lower() in _BH_DEVICE_MESH_DESCRIPTORS:
            _logger.info(
                f"setup_runner_environment: applying blackhole mesh config for device={getattr(settings, 'device', None)!r}"
            )
            _setup_blackhole_mesh_config(tt_metal_home)

    _RUNNERS_REQUIRING_GRID_OVERRIDE = {
        ModelRunners.TT_SDXL_TRACE.value,
        ModelRunners.TT_SDXL_EDIT.value,
        ModelRunners.TT_SDXL_IMAGE_TO_IMAGE.value,
        # FLUX.3 SP workers require a physical 12x10 Blackhole grid.  This
        # must run after mesh selection so the profile is already resolved.
        ModelRunners.TT_FLUX_3_IMAGE_KLEIN.value,
    }
    if getattr(settings, "model_runner", None) in _RUNNERS_REQUIRING_GRID_OVERRIDE:
        _setup_grid_override(getattr(settings, "device", None))


def setup_cpu_threading_limits(cpu_threads: str, num_torch_threads: int = 1):
    """Set up CPU threading limits for PyTorch to prevent CPU oversubscription"""
    os.environ["OMP_NUM_THREADS"] = cpu_threads
    os.environ["MKL_NUM_THREADS"] = cpu_threads
    os.environ["TORCH_NUM_THREADS"] = str(num_torch_threads)
    set_torch_thread_limits(num_threads=num_torch_threads)
    throttle_level = getattr(settings, "default_throttle_level", None)
    if throttle_level:
        os.environ["TT_MM_THROTTLE_PERF"] = throttle_level


def _setup_blackhole_mesh_config(tt_metal_home: str):
    """Configure mesh graph descriptors for Blackhole hardware"""
    device = (getattr(settings, "device", "") or "").lower()
    descriptor = _BH_DEVICE_MESH_DESCRIPTORS.get(device)
    if descriptor:
        os.environ["TT_MESH_GRAPH_DESC_PATH"] = (
            f"{tt_metal_home}/tt_metal/fabric/mesh_graph_descriptors/{descriptor}"
        )


def _flux3_runtime_profile():
    """Resolve the one profile used by settings, discovery and the runner."""
    return resolve_flux3_profile(tuple(getattr(settings, "device_mesh_shape", (1, 1))))


def _descriptor_path(tt_metal_home: str, descriptor: str) -> str:
    if os.path.isabs(descriptor):
        return descriptor
    return os.path.join(
        tt_metal_home, "tt_metal", "fabric", "mesh_graph_descriptors", descriptor
    )


def _setup_galaxy_mesh_config(tt_metal_home: str):
    """Configure the physical descriptor for the selected model mesh.

    FLUX.3 uses the canonical profile table for direct workers. Parent/child
    mode intentionally clears inherited descriptors and lets TT-Metal discover
    the native Galaxy topology before selecting a child mesh, matching the
    SD35 runner's validated startup path.
    """
    if "TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE" not in os.environ:
        _setup_grid_override(
            getattr(settings, "device", DeviceTypes.GALAXY.value)
        )
    if getattr(settings, "model_runner", None) != ModelRunners.TT_FLUX_3_IMAGE_KLEIN.value:
        descriptor = {
            (1, 1): "n150_mesh_graph_descriptor.textproto",
            (2, 1): "n300_mesh_graph_descriptor.textproto",
            (1, 4): "n300_2x2_mesh_graph_descriptor.textproto",
            (4, 1): "n300_2x2_mesh_graph_descriptor.textproto",
            (2, 4): "t3k_mesh_graph_descriptor.textproto",
            (2, 2): "bh_galaxy_flux3_tp2sp2_mesh.textproto",
            (1, 8): "bh_galaxy_flux3_tp8_ring.textproto",
            (8, 1): "bh_galaxy_flux3_sp8_ring.textproto",
        }.get(tuple(getattr(settings, "device_mesh_shape", (1, 1))))
        if descriptor:
            os.environ["TT_MESH_GRAPH_DESC_PATH"] = _descriptor_path(
                tt_metal_home, descriptor
            )
        return

    profile = _flux3_runtime_profile()
    # Settings selects parent mode explicitly for deployments that own the
    # complete Galaxy. Direct mode remains the safe standalone default and keeps
    # this helper usable by isolated workers and unit tests.
    parent_mode = os.environ.get("FLUX3_PARENT_MESH_MODE", "direct").strip().lower()
    parent_enabled = (
        parent_mode not in {"0", "false", "off", "direct", "legacy"}
        and profile.parent_mesh is not None
    )

    if parent_enabled:
        rows, cols = profile.parent_mesh
        os.environ["FLUX3_PARENT_MESH_SHAPE"] = f"{rows}x{cols}"
        os.environ["FLUX3_PARENT_MESH_WORKERS"] = "1"
        os.environ["FLUX3_PARENT_FABRIC_CONFIG"] = profile.fabric_config

        override = os.environ.get("FLUX3_PARENT_MESH_DESCRIPTOR", "").strip()
        if override:
            os.environ["TT_MESH_GRAPH_DESC_PATH"] = _descriptor_path(
                tt_metal_home, override
            )
            descriptor_note = (
                f" with explicit descriptor {os.environ['TT_MESH_GRAPH_DESC_PATH']}"
            )
        else:
            os.environ.pop("TT_MESH_GRAPH_DESC_PATH", None)
            os.environ.pop("TT_METAL_PHYSICAL_GROUPING_DESCRIPTOR_PATH", None)
            descriptor_note = " using native Galaxy topology"
        TTLogger().info(
            f"Configured FLUX.3 {profile.name.upper()} parent mesh "
            f"{rows}x{cols}, child={profile.worker_mesh}{descriptor_note}"
        )
        return

    descriptor = get_flux3_descriptor(profile)
    if descriptor:
        os.environ["TT_MESH_GRAPH_DESC_PATH"] = _descriptor_path(
            tt_metal_home, descriptor
        )
        TTLogger().info(
            f"Configured FLUX.3 {profile.name.upper()} direct descriptor: "
            f"{os.environ['TT_MESH_GRAPH_DESC_PATH']}"
        )


def _setup_grid_override(device: str):
    _logger = TTLogger()
    is_galaxy = (
        device in {DeviceTypes.BLACKHOLE_GALAXY.value, DeviceTypes.GALAXY.value}
        or getattr(settings, "is_galaxy", False)
    )
    if not is_galaxy:
        return

    if getattr(settings, "model_runner", None) != ModelRunners.TT_FLUX_3_IMAGE_KLEIN.value:
        override = "7,7" if device == DeviceTypes.GALAXY.value else "10,9"
        os.environ["TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE"] = override
        _logger.info(
            f"Set TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE={override} for {device}"
        )
        return

    # TT-Metal uses the maximum core coordinate.  The public FLUX3 setting is
    # expressed as WIDTHxHEIGHT, so 12x10 becomes the internal 11,9 override.
    requested = os.environ.get("FLUX3_CORE_GRID_OVERRIDE", "12x10")
    requested = requested.strip().lower().replace(" ", "")
    aliases = {"8x8": "7,7", "12x9": "11,8", "12x10": "11,9"}
    override = aliases.get(requested, requested)
    if "x" in override:
        width, height = override.split("x", 1)
        override = f"{int(width) - 1},{int(height) - 1}"
    parts = override.split(",")
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        raise ValueError(
            "FLUX3_CORE_GRID_OVERRIDE must be WIDTHxHEIGHT or X,Y, "
            f"got {requested!r}"
        )
    os.environ["TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE"] = override
    _logger.info(
        "Set FLUX.3 core grid to "
        f"{requested} (TT coordinate override={override})"
    )
