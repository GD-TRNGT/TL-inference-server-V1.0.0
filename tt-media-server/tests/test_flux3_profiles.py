import pytest

from config.flux3_profiles import (
    canonical_profile_name,
    get_flux3_profile,
    resolve_flux3_profile,
)


@pytest.mark.parametrize(
    ("name", "mesh"),
    [
        ("legacy-single", (1, 1)),
        ("sp2", (2, 1)),
        ("tp2", (2, 1)),
        ("tp4", (4, 1)),
        ("tp4axis1", (1, 4)),
        ("tp2sp2", (2, 2)),
        ("tp2sp4", (2, 4)),
        ("tp4sp2", (4, 2)),
        ("sp4", (4, 1)),
        ("sp8", (8, 1)),
        ("tp8ring", (1, 8)),
        ("sp8tp4", (4, 8)),
    ],
)
def test_explicit_profile_matches_worker_mesh(monkeypatch, name, mesh):
    monkeypatch.setenv("FLUX3_EXECUTION_PROFILE", name)
    profile = resolve_flux3_profile(mesh)
    assert profile.name == name
    assert profile.worker_mesh == mesh


def test_four_chip_auto_requires_semantics(monkeypatch):
    monkeypatch.setenv("FLUX3_EXECUTION_PROFILE", "auto")
    monkeypatch.delenv("FLUX3_PARALLELISM", raising=False)
    monkeypatch.delenv("FLUX3_TP_SIZE", raising=False)
    monkeypatch.delenv("FLUX3_TP_MESH", raising=False)
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_flux3_profile((4, 1))


def test_aliases_are_canonical():
    assert canonical_profile_name("sp4-tp2") == "tp2sp4"
    assert canonical_profile_name("pure_tp8_ring") == "tp8ring"
    assert get_flux3_profile("tp4-axis1").worker_mesh == (1, 4)


@pytest.mark.parametrize("name", ["tp2", "tp2linear"])
def test_pure_tp2_profiles_use_matching_linear_transport(name):
    profile = get_flux3_profile(name)
    assert profile.worker_mesh == (2, 1)
    assert profile.pipeline_topology == "linear"
    assert profile.fabric_config == "1d"
    assert profile.num_links == 1
