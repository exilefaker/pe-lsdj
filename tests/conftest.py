import pytest
from pylsdj import load_lsdsng


@pytest.fixture
def tohou_bytes():
    return load_lsdsng("data/tohou-final.lsdsng")._raw_bytes

@pytest.fixture
def organelle_bytes():
    return load_lsdsng("data/organelle.lsdsng")._raw_bytes

@pytest.fixture
def ofd_bytes():
    return load_lsdsng("data/orbital-final.lsdsng")._raw_bytes

@pytest.fixture
def equus_bytes():
    return load_lsdsng("data/equus.lsdsng")._raw_bytes

@pytest.fixture
def crshhh_bytes():
    return load_lsdsng("data/crshhh.lsdsng")._raw_bytes

@pytest.fixture
def organelle_bytes():
    return load_lsdsng("data/organelle.lsdsng")._raw_bytes
