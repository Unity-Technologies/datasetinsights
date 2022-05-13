# Dataset Insights

[![PyPI python](https://img.shields.io/pypi/pyversions/datasetinsights)](https://pypi.org/project/datasetinsights)
[![PyPI version](https://badge.fury.io/py/datasetinsights.svg)](https://pypi.org/project/datasetinsights)
[![Downloads](https://pepy.tech/badge/datasetinsights)](https://pepy.tech/project/datasetinsights)
[![Tests](https://github.com/Unity-Technologies/datasetinsights/actions/workflows/linting-and-unittests.yaml/badge.svg?branch=master&event=push)](https://github.com/Unity-Technologies/datasetinsights/actions/workflows/linting-and-unittests.yaml?query=branch%3Amaster+event%3Apush)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

Unity Dataset Insights is a python package for downloading, parsing and analyzing synthetic datasets generated using the Unity [Perception package](https://github.com/Unity-Technologies/com.unity.perception).

## Installation

Datasetinsights is published to PyPI. You can simply run `pip install datasetinsights` command under a supported python environments:

## Getting Started

### Dataset Statistics

We provide a sample [notebook](notebooks/Perception_Statistics.ipynb) to help you load synthetic datasets generated using [Perception package](https://github.com/Unity-Technologies/com.unity.perception) and visualize dataset statistics. We plan to support other sample Unity projects in the future.

### Download Datasets

You can download the datasets using the [download](https://datasetinsights.readthedocs.io/en/latest/datasetinsights.commands.html#datasetinsights-commands-download) command:

```bash
datasetinsights download --source-uri=<xxx> --output=$HOME/data
```

The download command supports HTTP(s), and GCS.

Alternatively, you can download dataset directly from python [interface](https://datasetinsights.readthedocs.io/en/latest/datasetinsights.io.downloader.html#module-datasetinsights.io.downloader).

`GCSDatasetDownloader` can download a dataset from GCS locations.
```python
from datasetinsights.io.downloader import GCSDatasetDownloader

source_uri=gs://url/to/file.zip # or gs://url/to/folder
dest = "~/data"
downloader = GCSDatasetDownloader()
downloader.download(source_uri=source_uri, output=dest)
```

`HTTPDatasetDownloader` can a dataset from any HTTP(S) url.
```python
from datasetinsights.io.downloader import HTTPDatasetDownloader

source_uri=http://url.to.file.zip
dest = "~/data"
downloader = HTTPDatasetDownloader()
downloader.download(source_uri=source_uri, output=dest)
```


## Docker

You can use the pre-build docker image [unitytechnologies/datasetinsights](https://hub.docker.com/r/unitytechnologies/datasetinsights) to interact with datasets.

## Documentation

You can find the API documentation on [readthedocs](https://datasetinsights.readthedocs.io/en/latest/).

## Contributing

Please let us know if you encounter a bug by filing an issue. To learn more about making a contribution to Dataset Insights, please see our Contribution [page](CONTRIBUTING.md).

## License

Dataset Insights is licensed under the Apache License, Version 2.0. See [LICENSE](LICENCE) for the full license text.

## Citation
If you find this package useful, consider citing it using:
```
@misc{datasetinsights2020,
    title={Unity {D}ataset {I}nsights Package},
    author={{Unity Technologies}},
    howpublished={\url{https://github.com/Unity-Technologies/datasetinsights}},
    year={2020}
}
```
