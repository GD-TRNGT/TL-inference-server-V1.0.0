# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Shared FLUX.3 request and mesh policy.

The API validator, worker discovery and TTNN runner must make the same
decisions about geometry and topology. Keeping these rules here avoids
slightly different interpretations of a profile in each startup layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

from config.flux3_profiles import (
    Flux3Profile,
    MeshShape,
    flux3_parent_mesh_for,
    get_flux3_descriptor,
    get_flux3_parent_mode,
    resolve_flux3_profile,
)


FLUX3_RUNNER = "tt-flux.3-image-klein"
FLUX3_MIN_RESOLUTION = 16
FLUX3_MAX_RESOLUTION = 4096
FLUX3_SHAPE_MULTIPLE = 16


def flux3_mesh_size(mesh_shape: MeshShape | None) -> int:
    """Return the number of devices in a logical worker mesh."""
    normalized = tuple(mesh_shape or (1, 1))
    if len(normalized) != 2 or any(int(value) < 1 for value in normalized):
        raise ValueError(f"Invalid FLUX.3 mesh shape: {mesh_shape!r}")
    return prod(int(value) for value in normalized)


def validate_flux3_geometry(
    width: int,
    height: int,
    *,
    model_runner: str = FLUX3_RUNNER,
    mesh_shape: MeshShape | None = None,
) -> None:
    """Validate FLUX.3 output geometry and known worker-memory limits.

    FLUX.3 accepts 16-pixel aligned dimensions up to 4096 per side. The
    current 4-chip Klein decoder path cannot fit 4096x4096 in L1; larger
    workers use the validated distributed VAE path. Other runners are not
    constrained by this policy.
    """
    if model_runner != FLUX3_RUNNER:
        return
    width = int(width)
    height = int(height)
    if width < FLUX3_MIN_RESOLUTION or height < FLUX3_MIN_RESOLUTION:
        raise ValueError(
            f"FLUX.3 width and height must be >= {FLUX3_MIN_RESOLUTION}; "
            f"got {width}x{height}"
        )
    if width > FLUX3_MAX_RESOLUTION or height > FLUX3_MAX_RESOLUTION:
        raise ValueError(
            f"FLUX.3 width and height must be <= {FLUX3_MAX_RESOLUTION}; "
            f"got {width}x{height}"
        )
    if width % FLUX3_SHAPE_MULTIPLE or height % FLUX3_SHAPE_MULTIPLE:
        raise ValueError(
            "FLUX.3 width and height must be multiples of "
            f"{FLUX3_SHAPE_MULTIPLE}; got {width}x{height}"
        )
    if flux3_mesh_size(mesh_shape) == 4 and width == 4096 and height == 4096:
        raise ValueError(
            "FLUX.3 4096x4096 is not supported on a 4-card worker; "
            "use a worker with at least 8 cards"
        )


@dataclass(frozen=True)
class Flux3MeshPlan:
    """One canonical topology decision shared by startup and inference."""

    profile: Flux3Profile
    child_mesh: MeshShape
    parent_mesh: MeshShape | None
    parent_mode: str
    descriptor: str | None

    @property
    def parent_enabled(self) -> bool:
        return self.parent_mesh is not None


def resolve_flux3_mesh_plan(
    mesh_shape: MeshShape,
    *,
    requested: str | None = None,
    parent_mode: str | None = None,
) -> Flux3MeshPlan:
    """Resolve profile, descriptor and optional parent mesh as one decision."""
    profile = resolve_flux3_profile(tuple(mesh_shape), requested=requested)
    normalized_mode = get_flux3_parent_mode(parent_mode)
    return Flux3MeshPlan(
        profile=profile,
        child_mesh=profile.worker_mesh,
        parent_mesh=flux3_parent_mesh_for(profile, mode=normalized_mode),
        parent_mode=normalized_mode,
        descriptor=get_flux3_descriptor(profile),
    )
