import logging
import os
import re
from pathlib import Path

import datasetinsights.constants as const
from datasetinsights.datasets.base import DatasetDownloader
from datasetinsights.io.usim import Downloader, download_manifest

logger = logging.getLogger(__name__)


class UnitySimulationDownloader(DatasetDownloader):

    SOURCE_URI_SCHEMA = "usim://"

    def download(self, source_uri, output, include_binary):
        (
            auth_token,
            project_id,
            run_execution_id,
        ) = self._parse_source_uri_to_usim(source_uri=source_uri)

        # use_cache = not args.no_cache
        manifest_file = os.path.join(
            output, const.SYNTHETIC_SUBFOLDER, f"{run_execution_id}.csv"
        )
        if auth_token:
            manifest_file = download_manifest(
                run_execution_id,
                manifest_file,
                auth_token,
                project_id=project_id,
            )
        else:
            logger.info(
                f"No auth token is provided. Assuming you already have "
                f"a manifest file located in {manifest_file}"
            )

        subfolder = Path(manifest_file).stem
        root = os.path.join(output, const.SYNTHETIC_SUBFOLDER, subfolder)
        dl_worker = Downloader(manifest_file, root)
        dl_worker.download_references()
        dl_worker.download_metrics()
        dl_worker.download_captures()
        if include_binary:
            dl_worker.download_binary_files()

    def _parse_source_uri_to_usim(self, source_uri):
        match = re.compile(r"usim://(\w+)@(\w+-\w+-\w+-\w+-\w+)/(\w+)")
        auth_token, project_id, run_execution_id = match.findall(source_uri)[0]
        return auth_token, project_id, run_execution_id
