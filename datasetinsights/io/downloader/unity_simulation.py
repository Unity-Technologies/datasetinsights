import logging
import os
import re

from datasetinsights.io.downloader.base import DatasetDownloader
from datasetinsights.io.usim import Downloader, download_manifest

logger = logging.getLogger(__name__)


class UnitySimulationDownloader(DatasetDownloader, protocol="usim://"):
    """ This class is used to download data from Unity Simulation

        For more on Unity Simulation please see these
        `docs <https://github.com/Unity-Technologies/Unity-Simulation-Docs>`

        Args:
        access_token: "Access token to be used to authenticate to
         unity simulation for downloading the dataset"

    """

    SOURCE_URI_PATTERN = r"usim://([^@]*)?@?" \
              r"([a-fA-F0-9]{8}-" \
              r"[a-fA-F0-9]{4}-" \
              r"[a-fA-F0-9]{4}-" \
              r"[a-fA-F0-9]{4}-" \
              r"[a-fA-F0-9]{12})" \
              r"/(\w+)"

    def __init__(self, access_token=None, **kwargs):
        """

        Args:
            access_token: "Access token to be used to authenticate to
            unity simulation for downloading the dataset"
        """
        self.access_token = access_token
        self.run_execution_id = None
        self.project_id = None

    def download(self, source_uri, output, include_binary=False, **kwargs):
        """

        Args:
            source_uri: This is the downloader-uri that indicates where on
                unity simulation the dataset should be downloaded from.

                The expected source-uri follows these patterns

                usim://access-token@project-id/run-execution-id
                or
                usim://project-id/run-execution-id


            output: This is the path to the directory
                where the download will store the dataset.

            include_binary: Whether to download binary files
                such as images or LIDAR point
                clouds. This flag applies to Datasets where metadata
                (e.g. annotation json, dataset catalog, ...)
                 can be separated from binary files.

        """
        self.parse_source_uri(source_uri)
        manifest_file = os.path.join(output, f"{self.run_execution_id}.csv")
        if self.access_token:
            manifest_file = download_manifest(
                self.run_execution_id,
                manifest_file,
                self.access_token,
                project_id=self.project_id,
            )
        else:
            logger.info(
                f"No auth token is provided. Assuming you already have "
                f"a manifest file located in {manifest_file}"
            )

        dl_worker = Downloader(manifest_file, output)
        dl_worker.download_references()
        dl_worker.download_metrics()
        dl_worker.download_captures()
        if include_binary:
            dl_worker.download_binary_files()

    def parse_source_uri(self, source_uri):
        """

        Args:
            source_uri: Parses source-uri in the following format
            usim://access-token@project-id/run-execution-id
            or
            usim://project-id/run-execution-id

        """
        pattern = re.compile(self.SOURCE_URI_PATTERN)
        if pattern.findall(source_uri):
            (access_token, project_id, run_execution_id,) = pattern.findall(
                source_uri
            )[0]
            if not self.access_token:
                if access_token:
                    self.access_token = access_token
                else:
                    raise ValueError(f"Missing access token")
            if project_id:
                self.project_id = project_id
            if run_execution_id:
                self.run_execution_id = run_execution_id

        else:
            raise ValueError(
                f"{source_uri} needs to be in format"
                f" usim://access_token@project_id/run_execution_id "
                f"or usim://project_id/run_execution_id "
            )
