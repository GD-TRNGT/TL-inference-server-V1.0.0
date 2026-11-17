# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import os

from config.constants import DeviceTypes, ModelRunners
from config.flux3_profiles import canonical_profile_name
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
            settings.model_runner == ModelRunners.TT_FLUX_3_IMAGE_KLEIN.value
            and settings.is_galaxy
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
        and settings.model_runner == ModelRunners.TT_FLUX_3_IMAGE_KLEIN.value
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
        is_flux3_galaxy_worker = (
            settings.model_runner == ModelRunners.TT_FLUX_3_IMAGE_KLEIN.value
            and (settings.device or "").lower()
            == DeviceTypes.BLACKHOLE_GALAXY.value
        )
        if settings.is_galaxy or is_flux3_galaxy_worker:
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
        # FLUX.3 SP workers require a physical 12x10 Blackhole grid.  This
        # must run after mesh selection so the profile is already resolved.
        ModelRunners.TT_FLUX_3_IMAGE_KLEIN.value,
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
    flux3_profile = canonical_profile_name(
        os.environ.get("FLUX3_EXECUTION_PROFILE", "auto")
    )
    is_flux3_sp8tp4 = (
        flux3_profile in {"sp8tp4", "sp8-tp4"}
        or (flux3_profile in {"", "auto"} and flux3_parallelism == "sp8tp4")
    )
    is_flux3_tp2sp4 = (
        flux3_profile in {"tp2sp4", "tp2-sp4", "sp4tp2", "sp4-tp2"}
        or (flux3_profile in {"", "auto"} and flux3_parallelism == "tp2sp4")
    )
    is_flux3_tp2 = flux3_profile == "tp2"
    is_flux3_tp2linear = flux3_profile in {"tp2linear", "tp2-linear", "tp2_line"}
    is_flux3_tp4sp2 = (
        flux3_profile in {"tp4sp2", "tp4-sp2", "sp2tp4", "sp2-tp4"}
        or (flux3_profile in {"", "auto"} and flux3_parallelism == "tp4sp2")
    )
    is_flux3_tp2sp2 = (
        flux3_profile in {"tp2sp2", "tp2-sp2", "sp2tp2", "sp2-tp2"}
        or (flux3_profile in {"", "auto"} and flux3_parallelism == "tp2sp2")
    )
    parent_mode = os.environ.get("FLUX3_PARENT_MESH_MODE", "direct").strip().lower()
    flux3_parent_specs = {
        # Match the SD3.5 Galaxy implementation: open the complete physical
        # 4x8 Galaxy mesh, then select one topology-valid 4x1 child.  A bare
        # 4x1 open is a partial torus and can fail Fabric router placement.
        # Both SP4 and pure TP4 use the same physical child shape; the runner
        # still selects the model-parallel semantics from its profile.
        "sp4": {
            "child": (4, 1),
            "shape": "4x8",
            "descriptor": "single_bh_galaxy_mesh_graph_descriptor.textproto",
        },
        "tp4": {
            "child": (4, 1),
            "shape": "4x8",
            "descriptor": "single_bh_galaxy_mesh_graph_descriptor.textproto",
        },
        "tp4axis1": {
            "child": (1, 4),
            "shape": "4x8",
            "descriptor": "single_bh_galaxy_mesh_graph_descriptor.textproto",
        },
        "tp2sp2": {
            "child": (2, 2),
            "shape": "8x4",
            "descriptor": "single_bh_galaxy_mesh_graph_descriptor.textproto",
        },
        "tp2sp4": {
            "child": (2, 4),
            # Galaxy's official single-mesh descriptor is an 8x4 plain mesh.
            # Keep the worker child as 2x4; create_submeshes yields four such
            # children from the 32-chip 8x4 parent.
            "shape": "8x4",
            "descriptor": "single_bh_galaxy_mesh_graph_descriptor.textproto",
        },
        "tp4sp2": {
            "child": (4, 2),
            # The 4x2 child is cut from the official 8x4 Galaxy parent.
            # Axis 0 carries TP4 and axis 1 carries SP2.
            "shape": "8x4",
            "descriptor": "single_bh_galaxy_mesh_graph_descriptor.textproto",
        },
        "sp8": {
            "child": (8, 1),
            "shape": "8x4",
            "descriptor": "single_bh_galaxy_torus_xy_graph_descriptor.textproto",
        },
        "tp8ring": {
            "child": (1, 8),
            "shape": "4x8",
            "descriptor": "bh_galaxy_flux3_sp8tp4_ring.textproto",
        },
    }
    parent_spec = flux3_parent_specs.get(flux3_profile)
    is_flux3_parent = (
        settings.model_runner == ModelRunners.TT_FLUX_3_IMAGE_KLEIN.value
        and parent_spec is not None
        and tuple(settings.device_mesh_shape) == parent_spec["child"]
        and parent_mode == "parent"
    )
    is_flux3_sequence_parallel = (
        flux3_profile in {"sp2", "sp4", "sp-4", "sp8", "sp-8", "tp2sp2", "tp2-sp2", "sp2tp2", "sp2-tp2", "tp2sp4", "tp2-sp4", "sp4tp2", "sp4-tp2", "tp4sp2", "tp4-sp2", "sp2tp4", "sp2-tp4", "sp8tp4", "sp8-tp4"}
        or (
            flux3_profile in {"", "auto"}
            and flux3_parallelism in {"sp", "sp4", "sp8", "tp2sp2", "tp2sp4", "tp4sp2", "sp8tp4"}
        )
    )
    if (
        settings.model_runner == ModelRunners.TT_FLUX_3_IMAGE_KLEIN.value
        and is_flux3_parent
    ):
        # The process opens the complete Galaxy parent mesh and selects one
        # topology-valid child with create_submeshes().
        os.environ["FLUX3_PARENT_MESH_SHAPE"] = parent_spec["shape"]
        descriptor = parent_spec["descriptor"]
        descriptor_override = os.environ.get("FLUX3_PARENT_MESH_DESCRIPTOR", "").strip()
        if descriptor_override:
            descriptor = descriptor_override
            os.environ["TT_MESH_GRAPH_DESC_PATH"] = (
                f"{tt_metal_home}/tt_metal/fabric/mesh_graph_descriptors/"
                f"{descriptor}"
            )
        else:
            # Match the SD3.5 parent/child path: do not force an MGD when the
            # runtime can resolve the native Blackhole Galaxy topology.  A
            # forced MGD or PGD from an older image makes SAT placement reject
            # the full parent before create_submeshes() is reached.  The
            # reference SD3.5 image leaves both descriptor paths unset and
            # lets TT-Metal select the pair that matches the detected cluster.
            os.environ.pop("TT_MESH_GRAPH_DESC_PATH", None)
            os.environ.pop("TT_METAL_PHYSICAL_GROUPING_DESCRIPTOR_PATH", None)
        TTLogger().info(
            f"Configured FLUX.3 {flux3_profile.upper()} parent mesh "
            f"{parent_spec['shape']} with native Galaxy topology"
            + (
                f" and explicit descriptor {os.environ['TT_MESH_GRAPH_DESC_PATH']}"
                if descriptor_override
                else " (no forced MGD)"
            )
        )
        return
    elif (
        settings.model_runner == ModelRunners.TT_FLUX_3_IMAGE_KLEIN.value
        and is_flux3_sp8tp4
    ):
        # The upstream FLUX.3 Klein SP8xTP4 tables target a 12x9 worker grid.
        # ``11,9`` exposes a 12x10 physical grid and leaves its last row for
        # CCL, avoiding the 8x8-grid Ring-SDPA L1 circular-buffer overlap.
        os.environ["TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE"] = "11,9"
    elif not is_flux3_sequence_parallel:
        os.environ["TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE"] = "7,7"

    # The versioned FLUX.3 image is layered on a prior SP4 image, so it can
    # inherit an SP4 descriptor in its environment. An explicit non-SP
    # profile (notably legacy-single) must never retain that descriptor:
    # TT-Metal then tries to place a one-chip mesh using an SP4 PGD and fails
    # before the runner can load. Replace it below with the shape-specific
    # descriptor instead.
    if (
        settings.model_runner == ModelRunners.TT_FLUX_3_IMAGE_KLEIN.value
        and flux3_profile not in {"", "auto"}
        and not is_flux3_sequence_parallel
    ):
        inherited_descriptor = os.environ.pop("TT_MESH_GRAPH_DESC_PATH", None)
        if inherited_descriptor:
            TTLogger().info(
                "Removed inherited FLUX.3 sequence-parallel mesh descriptor "
                f"for profile {flux3_profile}"
            )

    if (
        settings.model_runner == ModelRunners.TT_FLUX_3_IMAGE_KLEIN.value
        and (is_flux3_tp2 or is_flux3_tp2linear)
    ):
        # A 2x1 child has no valid wrapping axis for the Galaxy 1D Ring
        # fabric.  Pure TP2 therefore uses the same physical linear
        # descriptor for both public TP2 profile names.  Keeping the
        # descriptor, FabricConfig and pipeline topology aligned avoids the
        # post-initialization warmup hang seen with tp2.
        descriptor = "bh_galaxy_flux3_sp2_linear.textproto"
        os.environ["TT_MESH_GRAPH_DESC_PATH"] = (
            f"{tt_metal_home}/tt_metal/fabric/mesh_graph_descriptors/"
            f"{descriptor}"
        )
        TTLogger().info(
            "Configured FLUX.3 pure TP2 Linear descriptor: "
            f"{os.environ['TT_MESH_GRAPH_DESC_PATH']}"
        )
        # Pure TP2 is not sequence parallel; return after the normal core-grid
        # and inherited-descriptor cleanup above.
        return

    # SP2 requires the explicit two-chip Galaxy descriptor mounted by the
    # deployment.  Do not replace it with the single-board N300 descriptor.
    if is_flux3_sequence_parallel:
        explicit_descriptor = os.environ.get("TT_MESH_GRAPH_DESC_PATH")
        if explicit_descriptor:
            is_flux3_sp8 = flux3_profile in {"sp8", "sp-8"} or (
                flux3_profile in {"", "auto"} and flux3_parallelism == "sp8"
            )
            expected_descriptor = (
                "bh_galaxy_flux3_sp2_ring.textproto"
                if flux3_profile in {"sp2", "sp-2"}
                or (flux3_profile in {"", "auto"} and flux3_parallelism == "sp")
                else "bh_galaxy_flux3_sp8_ring.textproto"
                if is_flux3_sp8
                else "bh_galaxy_flux3_tp2sp2_mesh.textproto"
                if is_flux3_tp2sp2
                else "bh_galaxy_flux3_tp2sp4_torus.textproto"
                if is_flux3_tp2sp4
                else "bh_galaxy_flux3_tp4sp2_mesh.textproto"
                if is_flux3_tp4sp2
                else "bh_galaxy_flux3_sp8tp4_ring.textproto"
                if is_flux3_sp8tp4
                else None
            )
            if expected_descriptor and os.path.basename(explicit_descriptor) != expected_descriptor:
                # Older images carry an SP4 or SP8 descriptor in their image
                # environment. It is invalid for a different FLUX.3 eight-chip
                # profile, so choose that profile's descriptor below instead.
                os.environ.pop("TT_MESH_GRAPH_DESC_PATH", None)
                TTLogger().info(
                    "Removed inherited FLUX.3 mesh descriptor for "
                    f"{expected_descriptor}"
                )
            else:
                TTLogger().info(
                    "Keeping explicit FLUX.3 SP mesh descriptor: "
                    f"{explicit_descriptor}"
                )
                return

        is_flux3_sp8 = flux3_profile in {"sp8", "sp-8"} or (
            flux3_profile in {"", "auto"} and flux3_parallelism == "sp8"
        )
        if is_flux3_sp8tp4:
            # The full Galaxy exposes a 4x8 Ring mesh: TP4 weights on axis 0
            # and SP8 image-token rows on axis 1. Never inherit a per-tray
            # SP4 descriptor here, because that limits the system mesh to 4x1.
            os.environ["TT_MESH_GRAPH_DESC_PATH"] = (
                f"{tt_metal_home}/tt_metal/fabric/mesh_graph_descriptors/"
                "bh_galaxy_flux3_sp8tp4_ring.textproto"
            )
            TTLogger().info(
                "Configured FLUX.3 SP8xTP4 full-Galaxy Ring descriptor: "
                f"{os.environ['TT_MESH_GRAPH_DESC_PATH']}"
            )
            return

        if flux3_profile in {"sp2", "sp-2"} or (
            flux3_profile in {"", "auto"} and flux3_parallelism == "sp"
        ):
            # TT_VISIBLE_DEVICES selects one adjacent pair for each of the
            # sixteen independent SP2 workers.
            os.environ["TT_MESH_GRAPH_DESC_PATH"] = (
                f"{tt_metal_home}/tt_metal/fabric/mesh_graph_descriptors/"
                "bh_galaxy_flux3_sp2_ring.textproto"
            )
            TTLogger().info(
                "Configured FLUX.3 SP2 Ring descriptor: "
                f"{os.environ['TT_MESH_GRAPH_DESC_PATH']}"
            )
            return

        if is_flux3_sp8:
            # TT_VISIBLE_DEVICES picks one of the four topology-local 8-chip
            # Rings. This descriptor supplies its Ring dimension semantics;
            # it deliberately contains no device IDs and is reusable for all
            # four independent worker processes.
            os.environ["TT_MESH_GRAPH_DESC_PATH"] = (
                f"{tt_metal_home}/tt_metal/fabric/mesh_graph_descriptors/"
                "bh_galaxy_flux3_sp8_ring.textproto"
            )
            TTLogger().info(
                "Configured FLUX.3 SP8 Ring descriptor: "
                f"{os.environ['TT_MESH_GRAPH_DESC_PATH']}"
            )
            return

        if is_flux3_tp2sp4 and not is_flux3_parent:
            # One topology-local tray per worker. Axis 0 is the two-chip TP
            # line and axis 1 is the four-chip SP line; TT_VISIBLE_DEVICES
            # binds each of the four workers to a distinct tray.
            os.environ["TT_MESH_GRAPH_DESC_PATH"] = (
                f"{tt_metal_home}/tt_metal/fabric/mesh_graph_descriptors/"
                "bh_galaxy_flux3_tp2sp4_torus.textproto"
            )
            TTLogger().info(
                "Configured FLUX.3 TP2xSP4 2D mesh descriptor: "
                f"{os.environ['TT_MESH_GRAPH_DESC_PATH']}"
            )
            return

        if is_flux3_tp4sp2:
            # Same physical 2x4 tray, transposed for a four-chip TP line and
            # a two-chip SP line. The descriptor remains a non-wrapping mesh.
            os.environ["TT_MESH_GRAPH_DESC_PATH"] = (
                f"{tt_metal_home}/tt_metal/fabric/mesh_graph_descriptors/"
                "bh_galaxy_flux3_tp4sp2_mesh.textproto"
            )
            TTLogger().info(
                "Configured FLUX.3 TP4xSP2 2D mesh descriptor: "
                f"{os.environ['TT_MESH_GRAPH_DESC_PATH']}"
            )
            return

    flux3_profile = os.environ.get("FLUX3_EXECUTION_PROFILE", "").strip().lower()
    mesh_descriptors = {
        (1, 1): "n150_mesh_graph_descriptor.textproto",
        (2, 1): "n300_mesh_graph_descriptor.textproto",
        (1, 4): "n300_2x2_mesh_graph_descriptor.textproto",
        # Each four-chip worker is a topology-local 2x2 N300 group, exposed
        # logically as a 4x1 sequence-parallel Ring.
        (4, 1): "n300_2x2_mesh_graph_descriptor.textproto",
        (2, 4): "t3k_mesh_graph_descriptor.textproto",
        (2, 2): "bh_galaxy_flux3_tp2sp2_mesh.textproto",
        (1, 8): "bh_galaxy_flux3_tp8_ring.textproto",
        (8, 1): "bh_galaxy_flux3_sp8_ring.textproto",
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
        or settings.is_galaxy
    ):
        flux3_profile = os.environ.get("FLUX3_EXECUTION_PROFILE", "auto").strip().lower()
        flux3_parallelism = os.environ.get("FLUX3_PARALLELISM", "").strip().lower()
        is_flux3_sequence_parallel = (
            flux3_profile
            in {"sp2", "sp4", "sp-4", "sp8", "sp-8", "tp2sp2", "tp2-sp2", "sp2tp2", "sp2-tp2", "tp2sp4", "tp2-sp4", "sp4tp2", "sp4-tp2", "tp4sp2", "tp4-sp2", "sp8tp4", "sp8-tp4"}
            or (
                flux3_profile in {"", "auto"}
                and flux3_parallelism in {"sp", "sp4", "sp8", "tp2sp2", "tp2sp4", "tp4sp2", "sp8tp4"}
            )
        )
        if settings.model_runner == ModelRunners.TT_FLUX_3_IMAGE_KLEIN.value:
            # TT-Metal encodes the maximum core coordinate.  ``11,9`` is
            # therefore the full physical 12x10 Blackhole worker grid.  Keep
            # this layout identical for TP1xSP1 and all SP/TP profiles.
            requested_grid = os.environ.get("FLUX3_CORE_GRID_OVERRIDE", "").strip().lower().replace(" ", "")
            grid_aliases = {
                "8x8": "7,7",
                "12x9": "11,8",
                "12x10": "11,9",
            }
            expected = grid_aliases.get(requested_grid, requested_grid or "11,9")
            if "x" in expected:
                width, height = expected.split("x", 1)
                expected = f"{int(width) - 1},{int(height) - 1}"
            if not all(part.isdigit() for part in expected.split(",")) or expected.count(",") != 1:
                raise ValueError(
                    "FLUX3_CORE_GRID_OVERRIDE must be WIDTHxHEIGHT or X,Y, "
                    f"got {requested_grid!r}"
                )
            effective_grid = requested_grid or "12x10"
        else:
            expected = "7,7" if device == DeviceTypes.GALAXY.value else "10,9"
            effective_grid = None
        os.environ["TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE"] = expected
        if effective_grid:
            grid_label = (
                "TP8 Ring core grid"
                if flux3_profile == "tp8ring"
                else "sequence-parallel core grid"
            )
            _logger.info(
                f"Set FLUX.3 {grid_label} to "
                f"{effective_grid} (TT coordinate override={expected})"
            )
        else:
            _logger.info(
                f"Set TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE={expected} for {device}"
            )
