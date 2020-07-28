import tempfile
from datasetinsights.data.datasets import GroceriesReal

tmp_dir = tempfile.TemporaryDirectory()
tmp_name = tmp_dir.name

def test_groceriesreal_download():
    from datasetinsights.data.datasets import GroceriesReal
    with tempfile.TemporaryDirectory as tmpdirname:
        GroceriesReal

@patch("datasetinsights.data.datasets.GroceriesReal.download.download_file)
def test_download_http(mock_download):
     mock_download.return_value = tmp_name
     #tmp_name.add(chesumfile)
    GroceriesReal(Dataset)._download_http("dummmy", tmp_name)
        expected_checksum = self.GROCERIES_REAL_DATASET_TABLES[
        self.version
    ].checksum

    validate_checksum.assert_called_with(tmp_name, expected_checksum)


def test_groceriesreal_raises():
    pass
