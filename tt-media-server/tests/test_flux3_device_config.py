import pytest

from tt_model_runners.dit_runners import TTFlux3ImageKleinRunner


def test_flux3_reserves_128_kib_by_default(monkeypatch):
    monkeypatch.delenv('FLUX3_L1_SMALL_SIZE', raising=False)
    assert TTFlux3ImageKleinRunner.get_pipeline_device_params(None)['l1_small_size'] == 131072


@pytest.mark.parametrize('value', [65536, 98304, 131072])
def test_flux3_l1_small_reservation_is_configurable(monkeypatch, value):
    monkeypatch.setenv('FLUX3_L1_SMALL_SIZE', str(value))
    assert TTFlux3ImageKleinRunner.get_pipeline_device_params(None)['l1_small_size'] == value


@pytest.mark.parametrize('value', ['abc', '-1', '65535', '131073', '262144'])
def test_flux3_invalid_l1_small_reservation_is_rejected(monkeypatch, value):
    monkeypatch.setenv('FLUX3_L1_SMALL_SIZE', value)
    with pytest.raises(ValueError):
        TTFlux3ImageKleinRunner.get_pipeline_device_params(None)
