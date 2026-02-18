import pytest
from pylsdj import load_lsdsng
from pe_lsdj.songfile import SongFile


SONG_FILES = [
    "data/tohou-final.lsdsng",
    "data/organelle.lsdsng",
    "data/orbital-final.lsdsng",
    "data/equus.lsdsng",
    "data/crshhh.lsdsng",
]


@pytest.fixture(params=SONG_FILES)
def raw_bytes(request):
    return load_lsdsng(request.param)._raw_bytes


@pytest.fixture(params=SONG_FILES)
def song_file(request):
    return SongFile(request.param)