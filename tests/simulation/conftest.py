from pathlib import Path

import pytest


@pytest.fixture
def mock_data_dir():
    parent_dir = Path(__file__).parent.parent.absolute()
    mock_data_dir = parent_dir / "mock_data" / "simrun"

    return mock_data_dir
