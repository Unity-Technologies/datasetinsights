from datasetinsights.io.downloader.base import DatasetDownloader
from datasetinsights.io.gcs import GCSClient


class GCSDownloader(DatasetDownloader, protocol="gs://"):
    """ This class is used to download data from GCS
    """

    def __init__(self, **kwargs):
        """ initiating GCSDownloader
        """
        self.client = GCSClient()

    def download(self, source_uri=None, output=None, **kwargs):
        """

        Args:
            source_uri: This is the downloader-uri that indicates where on
                GCS the dataset should be downloaded from.
                The expected source-uri follows these patterns
                gs://bucket/folder or gs://bucket/folder/data.zip

            output: This is the path to the directory
                where the download will store the dataset.
                Examples:
            >>> cloud_path = "gs://bucket/folder or gs://bucket/folder/data.zip"
            >>> local_path = "/tmp/folder"
            >>> output ="/tmp/folder or /tmp/folder/data.zip"
        """
        return self.client.download(output, url=source_uri)
