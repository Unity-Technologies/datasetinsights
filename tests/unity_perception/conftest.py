from pathlib import Path

import pytest


@pytest.fixture
def mock_data_base_dir():
    parent_dir = Path(__file__).parent.parent.absolute()
    mock_data_dir = parent_dir / "mock_data"

    return mock_data_dir

@pytest.fixture
def mock_data_dir(mock_data_base_dir):
    mock_data_dir = mock_data_base_dir / "simrun"

    return mock_data_dir