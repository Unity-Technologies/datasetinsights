from datasetinsights.io.downloader.base import DatasetDownloader
from datasetinsights.io.gcs import download_folder_from_gcs


class GCSDownloader(DatasetDownloader, protocol="gs://"):
    """ This class is used to download data from GCS
    """

    def __init__(self, **kwargs):
        """

        Args:
            access_token: Access token to be used to authenticate to
            unity simulation for downloading the dataset
        """

    def download(self, source_uri, output, **kwargs):
        """

        Args:
            source_uri: This is the downloader-uri that indicates where on
                GCS the dataset should be downloaded from.
                The expected source-uri follows these patterns
                usim: gs://bucket/folder

            output: This is the path to the directory
                where the download will store the dataset.

        """
        download_folder_from_gcs(source_uri, output)
