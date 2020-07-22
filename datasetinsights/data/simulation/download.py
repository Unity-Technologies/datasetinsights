import concurrent.futures
import logging
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from codetiming import Timer
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from tqdm import tqdm

import datasetinsights.constants as const

from .exceptions import DownloadError
from .tables import DATASET_TABLES, FileType

logger = logging.getLogger(__name__)
# number of workers for ThreadPoolExecutor. This is the default value
# in python3.8
MAX_WORKER = min(32, os.cpu_count() + 4)
# Timeout of requests (in seconds)
DEFAULT_TIMEOUT = 1800
# Retry after failed request
DEFAULT_MAX_RETRIES = 5


def _filter_unsuccessful_attempts(manifest_df):
    """
    remove all rows from a dataframe where a greater attempt_id exists for
    the 'instance_id'. This is necessary so that we avoid using data from
    a failed USim run and only get the most recent retry.
    Args:
        manifest_df (pandas df): must have columns 'attempt_id', 'app_param_id'
        and 'instance_id'

    Returns(pandas df): where all rows for earlier attempt ids have been
    removed

    """
    last_attempt_per_instance = manifest_df.groupby("instance_id")[
        "attempt_id"
    ].agg(["max"])
    merged = manifest_df.merge(
        how="outer",
        right=last_attempt_per_instance,
        left_on="instance_id",
        right_on="instance_id",
    )
    filtered = merged[merged["attempt_id"] == merged["max"]]
    filtered = filtered.reset_index(drop=True)
    filtered = filtered.drop(columns="max")
    return filtered


class Downloader:
    """Parse a given manifest file to download simulation output

    For more on Unity Simulation please see these
    `docs <https://github.com/Unity-Technologies/Unity-Simulation-Docs>`_

    Attributes:
        manifest (DataFrame): the csv manifest file stored in a pandas dataframe
        data_root (str): root directory where the simulation output should
            be downloaded
        use_cache (bool): use cache instead of re-download if file exists
    """

    MANIFEST_FILE_COLUMNS = (
        "run_execution_id",
        "app_param_id",
        "instance_id",
        "attempt_id",
        "file_name",
        "download_uri",
    )

    def __init__(
        self, manifest_file: str, data_root: str, use_cache: bool = True
    ):
        """ Initialize Downloader

        Args:
            manifest_file (str): path to a manifest file
            data_root (str): root directory where the simulation output should
                be downloaded
            use_cache (bool): use cache instead of re-download if file exists
        """
        self.manifest = pd.read_csv(
            manifest_file, header=0, names=self.MANIFEST_FILE_COLUMNS
        )
        self.manifest = _filter_unsuccessful_attempts(manifest_df=self.manifest)
        self.manifest["filetype"] = self.match_filetypes(self.manifest)
        self.data_root = data_root
        self.use_cache = use_cache

    @staticmethod
    def match_filetypes(manifest):
        """ Match filetypes for every rows in the manifest file.

        Args:
            manifest (pd.DataFrame): the manifest csv file

        Returns:
            a list of filetype strings
        """
        filenames = manifest.file_name
        filetypes = []
        for name in filenames:
            for _, table in DATASET_TABLES.items():
                if re.match(table.pattern, name):
                    filetypes.append(table.filetype)
                    break
            else:
                filetypes.append(FileType.BINARY)

        return filetypes

    @Timer(name="download_all", text=const.TIMING_TEXT, logger=logging.info)
    def download_all(self):
        """ Download all files in the manifest file.
        """
        matched_rows = np.ones(len(self.manifest), dtype=bool)
        downloaded = self._download_rows(matched_rows)
        logger.info(
            f"Total {len(downloaded)} files in manifest are successfully "
            f"downloaded."
        )

    @Timer(
        name="download_references", text=const.TIMING_TEXT, logger=logging.info
    )
    def download_references(self):
        """ Download all reference files.
        All reference tables are static tables during the simulation.
        This typically comes from the definition of the simulation and should
        be created before tasks running distributed at different instances.
        """
        logger.info("Downloading references files...")
        matched_rows = self.manifest.filetype == FileType.REFERENCE
        downloaded = self._download_rows(matched_rows)

        logger.info(
            f"Total {len(downloaded)} reference files are successfully "
            f"downloaded."
        )

    @Timer(name="download_metrics", text=const.TIMING_TEXT, logger=logging.info)
    def download_metrics(self):
        """ Download all metrics files.
        See :ref:`metrics`
        """
        logger.info("Downloading metrics files...")
        matched_rows = self.manifest.filetype == FileType.METRIC
        downloaded = self._download_rows(matched_rows)
        logger.info(
            f"Total {len(downloaded)} metric files are successfully downloaded."
        )

    @Timer(
        name="download_captures", text=const.TIMING_TEXT, logger=logging.info
    )
    def download_captures(self):
        """ Download all captures files. See :ref:`captures`
        """
        logger.info("Downloading captures files...")
        matched_rows = self.manifest.filetype == FileType.CAPTURE
        downloaded = self._download_rows(matched_rows)
        logger.info(
            f"Total {len(downloaded)} capture files are successfully "
            f"downloaded."
        )

    @Timer(
        name="download_binary_files",
        text=const.TIMING_TEXT,
        logger=logging.info,
    )
    def download_binary_files(self):
        """ Download all binary files.
        """
        logger.info("Downloading binary files...")
        matched_rows = self.manifest.filetype == FileType.BINARY
        downloaded = self._download_rows(matched_rows)
        logger.info(
            f"Total {len(downloaded)} binary files are successfully "
            f"downloaded."
        )

    def _download_rows(self, matched_rows):
        """ Download matched rows in a manifest file.

        Note:
        We might need to download 1M+ of simulation output files, in this case
        we don't want to have a single file transfer failure holding back on
        getting the simulation data. Here download exception are captured.
        We only log an error message and requires uses to pay attention to
        this error.

        Args:
            matched_rows (pd.Series): boolean series indicator of the manifest
                file that should be downloaded

        Returns:
            list of strings representing the downloaded destination path.
        """
        n_expected = sum(matched_rows)
        future_downloaded = []
        downloaded = []
        with concurrent.futures.ThreadPoolExecutor(MAX_WORKER) as executor:
            for _, row in self.manifest[matched_rows].iterrows():
                source_uri = row.download_uri
                relative_path = row.file_name
                dest_path = os.path.join(self.data_root, relative_path)
                future = executor.submit(
                    _download_file, source_uri, dest_path, self.use_cache
                )
                future_downloaded.append(future)

            for future in tqdm(
                concurrent.futures.as_completed(future_downloaded),
                total=n_expected,
            ):
                try:
                    downloaded.append(future.result())
                except DownloadError as ex:
                    logger.error(ex)

        n_downloaded = len(downloaded)
        if n_downloaded != n_expected:
            logger.warning(
                f"Found {n_expected} matching records in the manifest file, "
                f"but only {n_downloaded} are downloaded."
            )

        return downloaded


class TimeoutHTTPAdapter(HTTPAdapter):
    def __init__(self, timeout, *args, **kwargs):
        self.timeout = timeout
        super().__init__(*args, **kwargs)

    def send(self, request, **kwargs):
        kwargs["timeout"] = self.timeout
        return super().send(request, **kwargs)


def _download_file(source_uri: str, dest_path: str, use_cache: bool = True):
    """Download a file specified from a source uri

    Args:
        source_uri (str): source url where the file should be downloaded
        dest_path (str): destination path of the file
        use_cache (bool): use_cache (bool): use cache instead of
                re-download if file exists

    Returns:
        String of destination path.
    """
    dest_path = Path(dest_path)
    if dest_path.exists() and use_cache:
        return dest_path

    logger.debug(f"Trying to download file from {source_uri} -> {dest_path}")
    adapter = TimeoutHTTPAdapter(
        timeout=DEFAULT_TIMEOUT, max_retries=Retry(total=DEFAULT_MAX_RETRIES)
    )
    with requests.Session() as http:
        http.mount("https://", adapter)
        try:
            response = http.get(source_uri)
            response.raise_for_status()
        except requests.exceptions.RequestException as ex:
            logger.error(ex)
            err_msg = (
                f"The request download from {source_uri} -> {dest_path} can't "
                f"be completed."
            )

            raise DownloadError(err_msg)
        else:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            with open(dest_path, "wb") as f:
                f.write(response.content)

    return dest_path


def download_manifest(
    run_execution_id, manifest_file, auth_token, project_id, use_cache=True
):
    """ Download manifest file from a single run_execution_id
    For more on Unity Simulation see these
    `docs <https://github.com/Unity-Technologies/Unity-Simulation-Docs>`_


    Args:
        run_execution_id (str): Unity Simulation run execution id
        manifest_file (str): path to the destination of the manifest_file
        auth_token (str): short lived authorization token
        project_id (str): Unity project id that has Unity Simulation enabled
        use_cache (bool, optional): indicator to skip download if manifest
            file already exists. Default: True.

    Returns:
        str: Full path to the manifest_file
    """
    api_endpoint = const.USIM_API_ENDPOINT
    project_url = f"{api_endpoint}/v1/projects/{project_id}/"
    data_url = f"{project_url}runs/{run_execution_id}/data"
    if Path(manifest_file).exists() and use_cache:
        logger.info(
            f"Mainfest file {manifest_file} already exists. Skipping downloads."
        )
        return manifest_file

    logger.info(
        f"Trying to download manifest file for run-execution-id "
        f"{run_execution_id}"
    )
    adapter = TimeoutHTTPAdapter(
        timeout=DEFAULT_TIMEOUT, max_retries=Retry(total=DEFAULT_MAX_RETRIES)
    )
    headers = {
        "Authorization": f"Bearer {auth_token}",
        "Content-Type": "application/json",
    }
    with requests.Session() as http:
        http.mount("https://", adapter)
        try:
            resp = http.get(data_url, headers=headers)
            resp.raise_for_status()
        except requests.exceptions.RequestException as ex:
            logger.error(ex)
            err_msg = (
                f"Failed to download manifest file for run-execution-id: "
                f"{run_execution_id}."
            )
            raise DownloadError(err_msg)
        else:
            Path(manifest_file).parent.mkdir(parents=True, exist_ok=True)
            with open(manifest_file, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024):
                    f.write(chunk)

    logger.info(
        f"Manifest file {manifest_file} downloaded for run-execution-id "
        f"{run_execution_id}"
    )

    return manifest_file
