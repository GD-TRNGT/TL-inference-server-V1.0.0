# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Canonical FLUX.3 Image Klein execution profiles.

The profile table is intentionally kept outside the runner implementation.
Settings, device discovery and the runner must agree on the same worker mesh,
parallel axes and parent-mesh layout. Keeping those values in one immutable
table prevents the three layers from silently drifting apart.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


MeshShape = tuple[int, int]
ParallelFactor = tuple[int, int]


@dataclass(frozen=True)
class Flux3Profile:
    name: str
    worker_mesh: MeshShape
    parent_mesh: MeshShape | None
    tensor_parallel: ParallelFactor
    sequence_parallel: ParallelFactor
    pipeline_topology: str
    fabric_config: str
    num_links: int
    uses_klein_pipeline: bool
    # Device discovery is a property of the physical worker group.  Keeping
    # it here prevents settings.py from growing another mesh-shape switch.
    discovery_kind: str = "single"
    # Most profiles shard the VAE over the logical TP axis.  Pure TP2 is the
    # deliberate exception: its VAE remains replicated until the TP2 VAE path
    # is validated independently.
    vae_parallelism: str = "tp_axis"
    # Startup-only compile shapes.  They do not constrain request geometry.
    pipeline_warmup_sizes: tuple[int, ...] = (1024, 2048, 4096)
    runner_warmup_sizes: tuple[int, ...] = (1024, 2048, 4096)

    @property
    def tp_degree(self) -> int:
        return self.tensor_parallel[0]

    @property
    def canonical_parallelism(self) -> str:
        if self.name in {"legacy-single", "tp2", "tp4", "tp4axis1"}:
            return "tp"
        if self.name == "sp2":
            return "sp"
        return self.name

    @property
    def supports_parent_mesh(self) -> bool:
        return self.parent_mesh is not None


# The descriptor is a property of the physical worker mesh, not of the model
# runner. Keeping it beside the profile prevents settings, discovery and the
# runner from maintaining separate descriptor tables.
_PROFILE_DESCRIPTORS = {
    "legacy-single": "n150_mesh_graph_descriptor.textproto",
    "sp2": "bh_galaxy_flux3_sp2_ring.textproto",
    "tp2": "bh_galaxy_flux3_sp2_linear.textproto",
    "sp4": "n300_2x2_mesh_graph_descriptor.textproto",
    "tp4": "n300_2x2_mesh_graph_descriptor.textproto",
    "tp4axis1": "n300_2x2_mesh_graph_descriptor.textproto",
    "tp2sp2": "bh_galaxy_flux3_tp2sp2_mesh.textproto",
    "tp2sp4": "bh_galaxy_flux3_tp2sp4_torus.textproto",
    "tp4sp2": "bh_galaxy_flux3_tp4sp2_mesh.textproto",
    "sp8": "bh_galaxy_flux3_sp8_ring.textproto",
    "tp8ring": "bh_galaxy_flux3_tp8_ring.textproto",
    "sp8tp4": "bh_galaxy_flux3_sp8tp4_ring.textproto",
}


_PROFILES = {
    "legacy-single": Flux3Profile(
        "legacy-single", (1, 1), None, (1, 1), (1, 0), "linear", "disabled", 1, False,
        "single", "replicated"
    ),
    "sp2": Flux3Profile(
        "sp2", (2, 1), (4, 8), (1, 1), (2, 0), "linear", "1d_ring", 1, True,
        "pairs", "tp_axis"
    ),
    "tp2": Flux3Profile(
        # A 2x1 Galaxy child is a line, not a valid 1D ring axis for the
        # native Fabric configuration.  The old Ring/FABRIC_1D_RING pair
        # initialized the device but stalled during the first warmup.
        # Keep pure TP2 on the reliable linear path; TP2 still shards the
        # hidden dimension across axis 0.
        "tp2", (2, 1), (4, 8), (2, 0), (1, 0), "linear", "1d", 1, True,
        "pairs", "replicated", runner_warmup_sizes=(1024,)
    ),
    "sp4": Flux3Profile(
        "sp4", (4, 1), (4, 8), (1, 1), (4, 0), "ring", "1d_ring", 2, True,
        "groups4", "tp_axis"
    ),
    "tp4": Flux3Profile(
        "tp4", (4, 1), (4, 8), (4, 0), (1, 0), "ring", "1d_ring", 1, False,
        "groups4", "tp_axis"
    ),
    "tp4axis1": Flux3Profile(
        "tp4axis1", (1, 4), (4, 8), (4, 1), (1, 0), "ring", "1d_ring", 1, False,
        "groups4", "tp_axis"
    ),
    "tp2sp2": Flux3Profile(
        "tp2sp2", (2, 2), (8, 4), (2, 0), (2, 1), "linear", "2d", 1, True,
        "groups4", "tp_axis", pipeline_warmup_sizes=(1024, 2048)
    ),
    "tp2sp4": Flux3Profile(
        "tp2sp4", (2, 4), (8, 4), (2, 0), (4, 1), "linear", "2d", 1, True,
        "groups8", "tp_axis", pipeline_warmup_sizes=(1024, 2048),
        runner_warmup_sizes=(1024, 2048)
    ),
    "tp4sp2": Flux3Profile(
        "tp4sp2", (4, 2), (8, 4), (4, 0), (2, 1), "linear", "2d", 1, True,
        "groups8", "tp_axis", pipeline_warmup_sizes=(1024, 2048),
        runner_warmup_sizes=(1024,)
    ),
    "sp8": Flux3Profile(
        "sp8", (8, 1), (8, 4), (1, 1), (8, 0), "ring", "1d_ring", 2, True,
        "sp8_ring", "tp_axis",
        # TP1xSP8 keeps 2048/4096 available for on-demand requests, but only
        # compiles the small representative shape during startup.  A warmup
        # matrix containing several sizes, reference layouts and text buckets
        # exhausts per-chip DRAM before the service becomes ready.  The Klein
        # pipeline still receives all three sizes for shape-aware VAE slicing;
        # larger request-time layouts are compiled lazily.
        runner_warmup_sizes=(1024,),
    ),
    "tp8ring": Flux3Profile(
        "tp8ring", (1, 8), (4, 8), (8, 1), (1, 0), "ring", "1d_ring", 1, False,
        "sp8_ring", "tp_axis"
    ),
    "sp8tp4": Flux3Profile(
        "sp8tp4", (4, 8), None, (4, 0), (8, 1), "ring", "1d_ring", 2, True,
        "groups32", "tp_axis"
    ),
}


PROFILE_ALIASES = {
    "": "auto",
    "auto": "auto",
    "legacy": "legacy-single",
    "single": "legacy-single",
    "single-card": "legacy-single",
    "legacy-single": "legacy-single",
    "sp2": "sp2",
    "tp2": "tp2",
    "sp4": "sp4",
    "sp-4": "sp4",
    "sp8": "sp8",
    "sp-8": "sp8",
    "tp4": "tp4",
    "tp4axis1": "tp4axis1",
    "tp4-axis1": "tp4axis1",
    "tp4_axis1": "tp4axis1",
    "tp2sp2": "tp2sp2",
    "tp2-sp2": "tp2sp2",
    "sp2tp2": "tp2sp2",
    "sp2-tp2": "tp2sp2",
    "tp2sp4": "tp2sp4",
    "tp2-sp4": "tp2sp4",
    "sp4tp2": "tp2sp4",
    "sp4-tp2": "tp2sp4",
    "tp4sp2": "tp4sp2",
    "tp4-sp2": "tp4sp2",
    "sp2tp4": "tp4sp2",
    "sp2-tp4": "tp4sp2",
    "sp8tp4": "sp8tp4",
    "sp8-tp4": "sp8tp4",
    "tp8ring": "tp8ring",
    "tp8-ring": "tp8ring",
    "pure-tp8-ring": "tp8ring",
    "pure_tp8_ring": "tp8ring",
}

_PARENT_DISABLED_MODES = frozenset({"", "0", "false", "off", "direct", "legacy"})
_PARENT_ENABLED_MODES = frozenset({"1", "true", "on", "enabled", "parent"})


def get_flux3_parent_mode(value: str | None = None, *, default: str = "direct") -> str:
    """Return the normalized FLUX.3 parent-mesh mode.

    A single helper is used by settings, scheduler, runner utils and the base
    mesh runner so a stale environment value cannot make only one layer think
    that parent/child mode is active.
    """
    return (default if value is None else value).strip().lower()


def is_flux3_parent_mode(value: str | None = None, *, default: str = "direct") -> bool:
    mode = get_flux3_parent_mode(value, default=default)
    if mode in _PARENT_DISABLED_MODES:
        return False
    if mode in _PARENT_ENABLED_MODES:
        return True
    raise ValueError(
        "FLUX3_PARENT_MESH_MODE must be one of: parent, direct, true, false"
    )


def flux3_parent_mesh_for(
    profile: Flux3Profile, *, mode: str | None = None, default: str = "direct"
) -> MeshShape | None:
    """Return the parent mesh only when this profile is explicitly enabled."""
    if not profile.supports_parent_mesh or not is_flux3_parent_mode(mode, default=default):
        return None
    return profile.parent_mesh


def canonical_profile_name(value: str | None) -> str:
    normalized = (value or "").strip().lower()
    try:
        name = PROFILE_ALIASES[normalized]
    except KeyError as exc:
        supported = ", ".join(sorted(_PROFILES))
        raise ValueError(
            f"FLUX3_EXECUTION_PROFILE must be auto or one of: {supported}"
        ) from exc
    return name


def get_flux3_profile(value: str | None) -> Flux3Profile:
    name = canonical_profile_name(value)
    if name == "auto":
        raise ValueError("auto does not identify a concrete FLUX3 profile")
    return _PROFILES[name]


def get_flux3_descriptor(value: Flux3Profile | str | None) -> str | None:
    """Return the direct-open mesh descriptor for a concrete profile.

    Parent/child mode deliberately does not use this descriptor: the parent
    mesh is opened using the runtime's native Galaxy topology and the worker
    selects a child view from it.
    """
    if isinstance(value, Flux3Profile):
        name = value.name
    else:
        name = canonical_profile_name(value)
    if name == "auto":
        return None
    return _PROFILE_DESCRIPTORS[name]


def resolve_flux3_profile(
    mesh_shape: MeshShape,
    requested: str | None = None,
) -> Flux3Profile:
    """Resolve one concrete profile without guessing ambiguous semantics."""
    normalized_mesh = tuple(mesh_shape)
    name = canonical_profile_name(
        os.environ.get("FLUX3_EXECUTION_PROFILE") if requested is None else requested
    )

    if name == "auto":
        parallelism = os.environ.get("FLUX3_PARALLELISM", "").strip().lower()
        tp_size = os.environ.get("FLUX3_TP_SIZE", "").strip()
        tp_mesh = os.environ.get("FLUX3_TP_MESH", "").strip().lower()
        if normalized_mesh == (4, 1):
            if parallelism in {"sp", "sp4", "sp-4"}:
                name = "sp4"
            elif parallelism in {"tp", "tp4"} or tp_size == "4" or tp_mesh == "4x1":
                name = "tp4"
            else:
                raise ValueError(
                    "FLUX3 mesh (4,1) is ambiguous: set "
                    "FLUX3_EXECUTION_PROFILE=sp4 or tp4"
                )
        elif normalized_mesh == (2, 1):
            name = "tp2" if parallelism == "tp" or tp_size == "2" else "sp2"
        else:
            name = {
                (1, 1): "legacy-single",
                (1, 4): "tp4axis1",
                (2, 2): "tp2sp2",
                (2, 4): "tp2sp4",
                (4, 2): "tp4sp2",
                (8, 1): "sp8",
                (1, 8): "tp8ring",
                (4, 8): "sp8tp4",
            }.get(normalized_mesh)
            if name is None:
                raise ValueError(f"Unsupported FLUX3 worker mesh: {normalized_mesh}")

    profile = _PROFILES[name]
    if profile.worker_mesh != normalized_mesh:
        raise ValueError(
            f"FLUX3_EXECUTION_PROFILE={profile.name} requires mesh "
            f"{profile.worker_mesh}, got {normalized_mesh}"
        )
    return profile


def resolve_flux3_environment_profile() -> Flux3Profile | None:
    """Resolve explicit and legacy FLUX.3 environment settings to one profile.

    FLUX3_EXECUTION_PROFILE is canonical. The older TP/SP variables remain
    accepted as compatibility inputs, but all interpretation lives here so
    settings, discovery, and runners cannot maintain different mesh rules.
    """
    requested = canonical_profile_name(os.environ.get("FLUX3_EXECUTION_PROFILE"))
    explicit_profile = requested != "auto"
    tp_size_raw = os.environ.get("FLUX3_TP_SIZE")
    parallelism = os.environ.get("FLUX3_PARALLELISM", "").strip().lower()
    tp_mesh = os.environ.get("FLUX3_TP_MESH", "").strip().lower().replace(" ", "")

    if requested != "auto":
        profile = get_flux3_profile(requested)
    elif tp_size_raw is None:
        return None
    else:
        try:
            tp_size = int(tp_size_raw)
        except ValueError as exc:
            raise ValueError("FLUX3_TP_SIZE must be one of 1, 2, 4, or 8") from exc

        if parallelism == "sp8":
            if tp_size != 1:
                raise ValueError("FLUX3_PARALLELISM=sp8 requires FLUX3_TP_SIZE=1")
            profile = get_flux3_profile("sp8")
        elif parallelism == "sp8tp4":
            if tp_size != 4:
                raise ValueError("FLUX3_PARALLELISM=sp8tp4 requires FLUX3_TP_SIZE=4")
            profile = get_flux3_profile("sp8tp4")
        else:
            if parallelism not in {"", "tp", "sp"}:
                raise ValueError(
                    "FLUX3_PARALLELISM must be 'tp', 'sp', 'sp8', or 'sp8tp4'"
                )
            if tp_size == 1:
                profile = get_flux3_profile("legacy-single")
            elif tp_size == 2:
                profile = get_flux3_profile("sp2" if parallelism == "sp" else "tp2")
            elif tp_size == 4:
                selected_mesh = tp_mesh or "1x4"
                if selected_mesh == "1x4":
                    profile = get_flux3_profile("tp4axis1")
                elif selected_mesh == "4x1":
                    profile = get_flux3_profile("sp4" if parallelism == "sp" else "tp4")
                else:
                    raise ValueError("FLUX3_TP_MESH must be '1x4' or '4x1'")
            elif tp_size == 8:
                if (tp_mesh or "1x8") != "1x8":
                    raise ValueError("pure FLUX.3 TP8 requires FLUX3_TP_MESH='1x8'")
                profile = get_flux3_profile("tp8ring")
            else:
                raise ValueError("FLUX3_TP_SIZE must be one of 1, 2, 4, or 8")

    expected_mesh = f"{profile.worker_mesh[0]}x{profile.worker_mesh[1]}"
    if explicit_profile and tp_size_raw is not None:
        try:
            requested_tp = int(tp_size_raw)
        except ValueError as exc:
            raise ValueError("FLUX3_TP_SIZE must be an integer") from exc
        if requested_tp != profile.tp_degree:
            raise ValueError(
                f"FLUX3_EXECUTION_PROFILE={profile.name} requires TP{profile.tp_degree}; "
                f"got FLUX3_TP_SIZE={tp_size_raw}"
            )
    if explicit_profile and tp_mesh and tp_mesh != expected_mesh:
        raise ValueError(
            f"FLUX3_EXECUTION_PROFILE={profile.name} requires "
            f"FLUX3_TP_MESH={expected_mesh}; got {tp_mesh}"
        )
    if explicit_profile and parallelism:
        normalized_parallelism = parallelism
        if normalized_parallelism not in {"tp", "sp"}:
            normalized_parallelism = canonical_profile_name(normalized_parallelism)
        if normalized_parallelism != profile.canonical_parallelism:
            raise ValueError(
                f"FLUX3_EXECUTION_PROFILE={profile.name} requires "
                f"FLUX3_PARALLELISM={profile.canonical_parallelism}; "
                f"got {parallelism}"
            )
    return profile


def expected_profile_names() -> tuple[str, ...]:
    return tuple(sorted(_PROFILES))
