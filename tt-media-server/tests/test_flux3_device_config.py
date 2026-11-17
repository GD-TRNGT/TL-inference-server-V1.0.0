import pytest
from types import SimpleNamespace

# tests/conftest.py installs a light-weight dit_runners stub for most tests.
# Load the real file through the shared isolated loader used by runner tests.
from tests.test_video_stage_metrics_wiring import _load_real_dit_runners

dit_runners = _load_real_dit_runners()
TTFlux3ImageKleinRunner = dit_runners.TTFlux3ImageKleinRunner
_flux3_execution_profile = dit_runners._flux3_execution_profile
_flux3_parallel_factors = dit_runners._flux3_parallel_factors
_flux3_tp_degree = dit_runners._flux3_tp_degree


def _legacy_runner_for_device_params():
    """Exercise device parameters without opening a real TT device."""
    runner = object.__new__(TTFlux3ImageKleinRunner)
    runner._execution_profile = "legacy-single"
    runner.settings = SimpleNamespace(device_mesh_shape=(1, 1))
    return runner

def test_flux3_legacy_profile_uses_distinct_factor_one_axes(monkeypatch):
    monkeypatch.setenv("FLUX3_EXECUTION_PROFILE", "legacy-single")
    runner = _legacy_runner_for_device_params()
    assert _flux3_parallel_factors((1, 1)) == ((1, 1), (1, 0))
    # TP1xSP1 intentionally stays on the established non-Klein FLUX3 T2I
    # path; the resident Klein pipeline starts at SP2 and above.
    assert not runner._uses_klein_pipeline()

def test_flux3_tp2sp4_profile_uses_2d_tensor_and_sequence_parallelism(monkeypatch):
    monkeypatch.setenv("FLUX3_EXECUTION_PROFILE", "tp2sp4")

    assert _flux3_execution_profile((2, 4)) == "tp2sp4"
    assert _flux3_tp_degree((2, 4)) == 2
    assert _flux3_parallel_factors((2, 4)) == ((2, 0), (4, 1))


@pytest.mark.parametrize("alias", ["tp2-sp4", "sp4tp2", "sp4-tp2"])
def test_flux3_tp2sp4_profile_aliases(monkeypatch, alias):
    monkeypatch.setenv("FLUX3_EXECUTION_PROFILE", alias)
    assert _flux3_execution_profile((2, 4)) == "tp2sp4"


def test_flux3_tp2sp4_uses_physical_2d_fabric():
    runner = _legacy_runner_for_device_params()
    runner._execution_profile = "tp2sp4"
    runner.settings = SimpleNamespace(device_mesh_shape=(2, 4))

    assert (
        runner.get_pipeline_device_params()["fabric_config"]
        == dit_runners.ttnn.FabricConfig.FABRIC_2D
    )


def test_flux3_tp4sp2_profile_uses_2d_tensor_and_sequence_parallelism(monkeypatch):
    monkeypatch.setenv("FLUX3_EXECUTION_PROFILE", "tp4sp2")

    assert _flux3_execution_profile((4, 2)) == "tp4sp2"
    assert _flux3_tp_degree((4, 2)) == 4
    assert _flux3_parallel_factors((4, 2)) == ((4, 0), (2, 1))


@pytest.mark.parametrize("alias", ["tp4-sp2", "sp2tp4", "sp2-tp4"])
def test_flux3_tp4sp2_profile_aliases(monkeypatch, alias):
    monkeypatch.setenv("FLUX3_EXECUTION_PROFILE", alias)
    assert _flux3_execution_profile((4, 2)) == "tp4sp2"


def test_flux3_tp4sp2_uses_physical_2d_fabric():
    runner = _legacy_runner_for_device_params()
    runner._execution_profile = "tp4sp2"
    runner.settings = SimpleNamespace(device_mesh_shape=(4, 2))

    assert (
        runner.get_pipeline_device_params()["fabric_config"]
        == dit_runners.ttnn.FabricConfig.FABRIC_2D
    )


def test_flux3_tp2sp2_profile_uses_2d_tensor_and_sequence_parallelism(monkeypatch):
    monkeypatch.setenv("FLUX3_EXECUTION_PROFILE", "tp2sp2")

    assert _flux3_execution_profile((2, 2)) == "tp2sp2"
    assert _flux3_tp_degree((2, 2)) == 2
    assert _flux3_parallel_factors((2, 2)) == ((2, 0), (2, 1))


@pytest.mark.parametrize("alias", ["tp2-sp2", "sp2tp2", "sp2-tp2"])
def test_flux3_tp2sp2_profile_aliases(monkeypatch, alias):
    monkeypatch.setenv("FLUX3_EXECUTION_PROFILE", alias)
    assert _flux3_execution_profile((2, 2)) == "tp2sp2"


def test_flux3_tp2sp2_uses_physical_2d_fabric():
    runner = _legacy_runner_for_device_params()
    runner._execution_profile = "tp2sp2"
    runner.settings = SimpleNamespace(device_mesh_shape=(2, 2))

    assert (
        runner.get_pipeline_device_params()["fabric_config"]
        == dit_runners.ttnn.FabricConfig.FABRIC_2D
    )

def test_flux3_reserves_128_kib_by_default(monkeypatch):
    monkeypatch.setenv("FLUX3_EXECUTION_PROFILE", "auto")
    monkeypatch.delenv('FLUX3_L1_SMALL_SIZE', raising=False)
    assert _legacy_runner_for_device_params().get_pipeline_device_params()['l1_small_size'] == 131072


@pytest.mark.parametrize('value', [65536, 98304, 131072])
def test_flux3_l1_small_reservation_is_configurable(monkeypatch, value):
    monkeypatch.setenv("FLUX3_EXECUTION_PROFILE", "auto")
    monkeypatch.setenv('FLUX3_L1_SMALL_SIZE', str(value))
    assert _legacy_runner_for_device_params().get_pipeline_device_params()['l1_small_size'] == value


@pytest.mark.parametrize('value', ['abc', '-1', '65535', '131073', '262144'])
def test_flux3_invalid_l1_small_reservation_is_rejected(monkeypatch, value):
    monkeypatch.setenv("FLUX3_EXECUTION_PROFILE", "auto")
    monkeypatch.setenv('FLUX3_L1_SMALL_SIZE', value)
    with pytest.raises(ValueError):
        _legacy_runner_for_device_params().get_pipeline_device_params()
