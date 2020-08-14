import logging
import os
import re
from pathlib import Path

from datasetinsights.io.downloader.base import DatasetDownloader
from datasetinsights.io.usim import Downloader, download_manifest

logger = logging.getLogger(__name__)


class UnitySimulationDownloader(DatasetDownloader):
    """ Downloads unity simulation datasets
        Args:
        access_token: "Access token to be used to authenticate to
         unity simulation for downloading the dataset"

    """

    PROTOCOL = "usim://"

    def __init__(self, access_token=None):
        self.access_token = access_token
        self.run_execution_id = None
        self.project_id = None

    def download(self, source_uri, output, include_binary):
        """

        Args:
            source_uri: This is the downloader-uri that indicates where on
             unity simulation the dataset should be downloaded from.

            output: This is the path to the directory
            where the download will store the dataset.

            include_binary: Whether to download binary files
             such as images or LIDAR point
            clouds. This flag applies to Datasets where metadata
            (e.g. annotation json, dataset catalog, ...) can be separated from
            binary files.

        """
        self.parse_source_uri(source_uri)
        # use_cache = not args.no_cache
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

        subfolder = Path(manifest_file).stem
        output_directory = os.path.join(output, subfolder)
        dl_worker = Downloader(manifest_file, output_directory)
        dl_worker.download_references()
        dl_worker.download_metrics()
        dl_worker.download_captures()
        if include_binary:
            dl_worker.download_binary_files()

    def parse_source_uri(self, source_uri):
        if self.access_token:
            self._parse_with_no_access_token(source_uri)
            if not self.project_id and not self.run_execution_id:
                self._parse_potential_overridden_access_token(source_uri)
        else:
            self._parse_source_uri(source_uri)

        if (
            not self.access_token
            and not self.project_id
            and not self.run_execution_id
        ):
            raise ValueError(
                f"{source_uri} needs to be in format"
                f" usim://access_token@project_id/run_execution_id"
            )

    def _parse_with_no_access_token(self, source_uri):
        match = re.compile(r"usim://(\w+-\w+-\w+-\w+-\w+)/(\w+)")
        result = match.findall(source_uri)
        if result:
            self.project_id, self.run_execution_id = result[0]

    def _parse_potential_overridden_access_token(self, source_uri):
        match = re.compile(r"usim://\w+@(\w+-\w+-\w+-\w+-\w+)/(\w+)")
        result = match.findall(source_uri)
        if result:
            self.project_id, self.run_execution_id = result[0]
        else:
            raise ValueError(
                f"{source_uri} needs to be in format"
                f" usim://access_token@project_id/run_execution_id"
            )

    def _parse_source_uri(self, source_uri):
        match = re.compile(r"usim://(\w+)@(\w+-\w+-\w+-\w+-\w+)/(\w+)")
        if match.findall(source_uri):
            (
                self.access_token,
                self.project_id,
                self.run_execution_id,
            ) = match.findall(source_uri)[0]
        else:
            raise ValueError(
                f"{source_uri} needs to be in format"
                f" usim://access_token@project_id/run_execution_id"
            )
