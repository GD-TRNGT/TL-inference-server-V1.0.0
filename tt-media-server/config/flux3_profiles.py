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

    @property
    def tp_degree(self) -> int:
        return self.tensor_parallel[0]

    @property
    def canonical_parallelism(self) -> str:
        if self.name in {"legacy-single", "tp2", "tp2linear", "tp4", "tp4axis1"}:
            return "tp"
        if self.name == "sp2":
            return "sp"
        return self.name


_PROFILES = {
    "legacy-single": Flux3Profile(
        "legacy-single", (1, 1), None, (1, 0), (1, 0), "linear", "disabled", 1, False
    ),
    "sp2": Flux3Profile(
        "sp2", (2, 1), None, (1, 1), (2, 0), "linear", "1d_ring", 1, True
    ),
    "tp2": Flux3Profile(
        # A 2x1 Galaxy child is a line, not a valid 1D ring axis for the
        # native Fabric configuration.  The old Ring/FABRIC_1D_RING pair
        # initialized the device but stalled during the first warmup.
        # Keep pure TP2 on the reliable linear path; TP2 still shards the
        # hidden dimension across axis 0.
        "tp2", (2, 1), None, (2, 0), (1, 0), "linear", "1d", 1, True
    ),
    "tp2linear": Flux3Profile(
        # This profile was accidentally declared as Ring while selecting a
        # linear Fabric descriptor.  That topology/Fabric mismatch caused the
        # same warmup hang as tp2.  Both explicit TP2 profiles now describe
        # the actual 2x1 Linear execution path.
        "tp2linear", (2, 1), None, (2, 0), (1, 0), "linear", "1d", 1, True
    ),
    "sp4": Flux3Profile(
        "sp4", (4, 1), (4, 8), (1, 1), (4, 0), "ring", "1d_ring", 2, True
    ),
    "tp4": Flux3Profile(
        "tp4", (4, 1), (4, 8), (4, 0), (1, 0), "ring", "1d_ring", 1, False
    ),
    "tp4axis1": Flux3Profile(
        "tp4axis1", (1, 4), (4, 8), (4, 1), (1, 0), "ring", "1d_ring", 1, False
    ),
    "tp2sp2": Flux3Profile(
        "tp2sp2", (2, 2), (8, 4), (2, 0), (2, 1), "linear", "2d", 1, True
    ),
    "tp2sp4": Flux3Profile(
        "tp2sp4", (2, 4), (8, 4), (2, 0), (4, 1), "linear", "2d", 1, True
    ),
    "tp4sp2": Flux3Profile(
        "tp4sp2", (4, 2), (8, 4), (4, 0), (2, 1), "linear", "2d", 1, True
    ),
    "sp8": Flux3Profile(
        "sp8", (8, 1), (8, 4), (1, 1), (8, 0), "ring", "1d_ring", 2, True
    ),
    "tp8ring": Flux3Profile(
        "tp8ring", (1, 8), (4, 8), (8, 1), (1, 0), "ring", "1d_ring", 1, False
    ),
    "sp8tp4": Flux3Profile(
        "sp8tp4", (4, 8), None, (4, 0), (8, 1), "ring", "1d_ring", 2, True
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
    "tp2linear": "tp2linear",
    "tp2-linear": "tp2linear",
    "tp2_line": "tp2linear",
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


def expected_profile_names() -> tuple[str, ...]:
    return tuple(sorted(_PROFILES))
