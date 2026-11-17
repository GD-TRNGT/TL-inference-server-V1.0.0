# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.


import os

import ttnn
from config.flux3_profiles import canonical_profile_name, get_flux3_profile
from tt_model_runners.base_device_runner import BaseDeviceRunner


class BaseMetalDeviceRunner(BaseDeviceRunner):
    def __init__(self, device_id: str):
        super().__init__(device_id)
        # In Galaxy hybrid TP/DP mode the process owns one child mesh while
        # the parent remains alive for the whole model lifetime.
        self._parent_mesh_device = None

    def get_pipeline_device_params(self):
        return None

    def set_device(self):
        if self.ttnn_device is None:
            # for now use all available devices
            self.ttnn_device = self._mesh_device()
        self.max_batch_size = self.settings.max_batch_size
        return self.ttnn_device

    def close_device(self):
        try:
            self.logger.info(f"Device {self.device_id}: Closing mesh device...")
            mesh_device = self._parent_mesh_device or self.ttnn_device
            if mesh_device is not None:
                # A submesh is a view owned by its parent; close the parent
                # handle once, rather than closing the child view directly.
                ttnn.close_mesh_device(mesh_device)
                self.ttnn_device = None
                self._parent_mesh_device = None
                self.logger.info(
                    f"Device {self.device_id}: Successfully closed mesh device"
                )
            else:
                self.logger.info(
                    f"Device {self.device_id}: Device is None, no need to close"
                )
        except Exception as e:
            self.logger.error(f"Device {self.device_id}: Failed to close device: {e}")
            raise RuntimeError(
                f"Device {self.device_id}: Device cleanup failed: {str(e)}"
            ) from e

    def get_updated_device_params(self, device_params):
        if device_params is None:
            device_params = {}

        new_device_params = device_params.copy()

        dispatch_core_axis = new_device_params.pop("dispatch_core_axis", None)
        dispatch_core_type = new_device_params.pop("dispatch_core_type", None)
        fabric_tensix_config = new_device_params.get("fabric_tensix_config", None)

        if ttnn.device.is_blackhole():
            # Only when both fabric_config and fabric_tensix_config are set, we can use ROW dispatch, otherwise force to use COL dispatch
            fabric_config = new_device_params.get("fabric_config", None)
            if not (fabric_config and fabric_tensix_config):
                # When not both are set, force COL dispatch
                if dispatch_core_axis == ttnn.DispatchCoreAxis.ROW:
                    self.logger.warning(
                        "ROW dispatch requires both fabric and tensix config, using DispatchCoreAxis.COL instead."
                    )
                    dispatch_core_axis = ttnn.DispatchCoreAxis.COL
            elif fabric_config and fabric_tensix_config:
                self.logger.warning(
                    f"Blackhole with fabric_config and fabric_tensix_config enabled, using fabric_tensix_config={fabric_tensix_config}"
                )

        dispatch_core_config = ttnn.DispatchCoreConfig(
            dispatch_core_type, dispatch_core_axis, fabric_tensix_config
        )
        new_device_params["dispatch_core_config"] = dispatch_core_config

        return new_device_params

    def _mesh_device(self):
        try:
            device_ids = ttnn.get_device_ids()
            if not device_ids:
                raise RuntimeError("No TTNN devices available")
            self.logger.info(
                f"Device {self.device_id}: Found {len(device_ids)} available TTNN devices: {device_ids}"
            )

            child_shape = tuple(self.settings.device_mesh_shape)
            parent_shape = self._parent_mesh_shape()
            mesh_shape = ttnn.MeshShape(parent_shape or child_shape)

            device_params = self.get_pipeline_device_params()
            updated_device_params = self.get_updated_device_params(device_params)
            if parent_shape is not None:
                # The default parent path is the validated 4x8 Galaxy Ring.
                # Graph-based parents built from physical 2x4 meshes expose
                # mesh connectivity instead, so allow an explicit topology
                # override for those descriptors.
                parent_fabric = os.environ.get(
                    "FLUX3_PARENT_FABRIC_CONFIG", "1d_ring"
                ).strip().lower()
                parent_fabric_config = {
                    "1d_ring": ttnn.FabricConfig.FABRIC_1D_RING,
                    "ring": ttnn.FabricConfig.FABRIC_1D_RING,
                    "1d": ttnn.FabricConfig.FABRIC_1D,
                    "2d": ttnn.FabricConfig.FABRIC_2D,
                    "mesh": ttnn.FabricConfig.FABRIC_2D,
                }.get(parent_fabric)
                if parent_fabric_config is None:
                    raise ValueError(
                        "FLUX3_PARENT_FABRIC_CONFIG must be one of "
                        "1d_ring, 1d, or 2d"
                    )
                updated_device_params["fabric_config"] = parent_fabric_config
                self.logger.info(
                    f"Device {self.device_id}: parent mesh {parent_shape} "
                    f"uses {parent_fabric} fabric for placement"
                )
            fabric_config = self._configure_fabric(updated_device_params)
            parent_device = self._initialize_mesh_device(
                mesh_shape, updated_device_params, fabric_config
            )

            if parent_shape is not None:
                child_meshes = parent_device.create_submeshes(
                    ttnn.MeshShape(child_shape)
                )
                if not child_meshes:
                    raise RuntimeError(
                        f"Parent mesh {parent_shape} did not produce any "
                        f"submeshes of shape {child_shape}"
                    )
                try:
                    submesh_index = int(os.environ.get("FLUX3_SUBMESH_INDEX", "0"))
                except ValueError as exc:
                    raise RuntimeError("FLUX3_SUBMESH_INDEX must be an integer") from exc
                if submesh_index < 0 or submesh_index >= len(child_meshes):
                    raise RuntimeError(
                        f"FLUX3_SUBMESH_INDEX={submesh_index} is out of range for "
                        f"{len(child_meshes)} submeshes of shape {child_shape}"
                    )
                self._parent_mesh_device = parent_device
                mesh_device = child_meshes[submesh_index]
                self.logger.info(
                    f"Device {self.device_id}: Created parent mesh {parent_shape} "
                    f"and selected child mesh {child_shape} at index {submesh_index} "
                    f"({mesh_device.get_num_devices()} devices)"
                )
            else:
                mesh_device = parent_device

            self.logger.info(
                f"Device {self.device_id}: Created mesh device with {mesh_device.get_num_devices()} devices"
            )
            return mesh_device
        except Exception as e:
            self.logger.error(
                f"Device {self.device_id}: Device initialization failed: {e}"
            )
            raise RuntimeError(
                f"Unexpected device initialization error: {str(e)}"
            ) from e

    def _configure_fabric(self, updated_device_params):
        return None

    def _parent_mesh_shape(self):
        """Return the opt-in parent mesh for a hybrid Galaxy worker.

        Profile metadata is owned by ``config.flux3_profiles`` so the base
        device runner does not carry a second, diverging FLUX3 profile table.
        """
        raw = os.environ.get("FLUX3_PARENT_MESH_SHAPE", "").strip().lower()
        child = tuple(self.settings.device_mesh_shape)
        profile_name = canonical_profile_name(
            os.environ.get("FLUX3_EXECUTION_PROFILE", "")
        )
        if profile_name == "auto":
            return None
        profile = get_flux3_profile(profile_name)
        expected_parent = profile.parent_mesh
        if not raw or expected_parent is None or child != profile.worker_mesh:
            return None
        try:
            rows, cols = (
                int(part) for part in raw.replace("x", ",").split(",")
            )
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "FLUX3_PARENT_MESH_SHAPE must use '<rows>x<cols>'"
            ) from exc
        if rows < 1 or cols < 1:
            raise RuntimeError(f"Invalid FLUX3_PARENT_MESH_SHAPE={raw!r}")
        if rows * cols < child[0] * child[1]:
            raise RuntimeError(
                f"Parent mesh {(rows, cols)} is smaller than child mesh {child}"
            )
        if (rows, cols) != expected_parent:
            raise RuntimeError(
                f"FLUX3_EXECUTION_PROFILE={profile.name} requires parent mesh "
                f"{expected_parent[0]}x{expected_parent[1]}, got {raw!r}"
            )
        return (rows, cols)

    def _initialize_mesh_device(self, mesh_shape, device_params, fabric_config):
        try:
            mesh_device = ttnn.open_mesh_device(mesh_shape=mesh_shape, **device_params)
        except Exception as e:
            try:
                if fabric_config:
                    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
            except Exception as reset_error:
                self.logger.warning(
                    f"Device {self.device_id}: Failed to reset fabric after device initialization failure: {reset_error}"
                )
            self.logger.error(
                f"Device {self.device_id}: Mesh device initialization failed: {e}"
            )
            raise RuntimeError(f"Mesh device initialization failed: {str(e)}") from e
        return mesh_device
