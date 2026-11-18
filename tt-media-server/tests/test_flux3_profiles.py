import pytest

from config.flux3_profiles import (
    canonical_profile_name,
    flux3_parent_mesh_for,
    get_flux3_profile,
    is_flux3_parent_mode,
    resolve_flux3_environment_profile,
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


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({"FLUX3_TP_SIZE": "2", "FLUX3_PARALLELISM": "sp"}, ("sp2", (2, 1))),
        ({"FLUX3_TP_SIZE": "2", "FLUX3_PARALLELISM": "tp"}, ("tp2", (2, 1))),
        (
            {"FLUX3_TP_SIZE": "4", "FLUX3_TP_MESH": "4x1"},
            ("tp4", (4, 1)),
        ),
        (
            {
                "FLUX3_TP_SIZE": "4",
                "FLUX3_TP_MESH": "4x1",
                "FLUX3_PARALLELISM": "sp",
            },
            ("sp4", (4, 1)),
        ),
        (
            {"FLUX3_TP_SIZE": "8", "FLUX3_TP_MESH": "1x8"},
            ("tp8ring", (1, 8)),
        ),
        ({"FLUX3_TP_SIZE": "1", "FLUX3_PARALLELISM": "sp8"}, ("sp8", (8, 1))),
    ],
)
def test_legacy_environment_resolves_through_canonical_profiles(
    monkeypatch, env, expected
):
    for key in (
        "FLUX3_EXECUTION_PROFILE",
        "FLUX3_TP_SIZE",
        "FLUX3_TP_MESH",
        "FLUX3_PARALLELISM",
    ):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    profile = resolve_flux3_environment_profile()
    assert (profile.name, profile.worker_mesh) == expected


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


@pytest.mark.parametrize("name", ["tp2"])
def test_pure_tp2_profiles_use_matching_linear_transport(name):
    profile = get_flux3_profile(name)
    assert profile.worker_mesh == (2, 1)
    assert profile.pipeline_topology == "linear"
    assert profile.fabric_config == "1d"
    assert profile.num_links == 1


@pytest.mark.parametrize(
    ("name", "discovery_kind"),
    [
        ("legacy-single", "single"),
        ("sp2", "pairs"),
        ("tp2", "pairs"),
        ("sp4", "groups4"),
        ("tp2sp4", "groups8"),
        ("sp8", "sp8_ring"),
        ("tp8ring", "sp8_ring"),
        ("sp8tp4", "groups32"),
    ],
)
def test_profile_owns_physical_discovery_strategy(name, discovery_kind):
    assert get_flux3_profile(name).discovery_kind == discovery_kind


def test_parent_mesh_requires_explicit_non_direct_mode():
    profile = get_flux3_profile("sp4")
    assert not is_flux3_parent_mode("direct")
    assert flux3_parent_mesh_for(profile, mode="direct") is None
    assert is_flux3_parent_mode("parent")
    assert flux3_parent_mesh_for(profile, mode="parent") == (4, 8)
