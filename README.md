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

### Load Datasets

The [Unity Perception](https://datasetinsights.readthedocs.io/en/latest/datasetinsights.datasets.unity_perception.html#datasetinsights-datasets-unity-perception) package provides datasets under this [schema](https://datasetinsights.readthedocs.io/en/latest/Synthetic_Dataset_Schema.html#synthetic-dataset-schema). The datasetinsighs package also provide convenient python modules to parse datasets.

For example, you can load `AnnotationDefinitions` into a python dictionary by providing the corresponding annotation definition ID:

```python
from datasetinsights.datasets.unity_perception import AnnotationDefinitions

annotation_def = AnnotationDefinitions(data_root=dest, version="my_schema_version")
definition_dict = annotation_def.get_definition(def_id="my_definition_id")
```

Similarly, for `MetricDefinitions`:
```python
from datasetinsights.datasets.unity_perception import MetricDefinitions

metric_def = MetricDefinitions(data_root=dest, version="my_schema_version")
definition_dict = metric_def.get_definition(def_id="my_definition_id")
```

The `Captures` table provide the collection of simulation captures and annotations. You can load these records directly as a Pandas `DataFrame`:

```python
from datasetinsights.datasets.unity_perception import Captures

captures = Captures(data_root=dest, version="my_schema_version")
captures_df = captures.filter(def_id="my_definition_id")
```


The `Metrics` table can store simulation metrics for a capture or annotation. You can also load these records as a Pandas `DataFrame`:

```python
from datasetinsights.datasets.unity_perception import Metrics

metrics = Metrics(data_root=dest, version="my_schema_version")
metrics_df = metrics.filter_metrics(def_id="my_definition_id")
```

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

### Convert Datasets

If you are interested in converting the synthetic dataset to COCO format for
annotations that COCO supports, you can run the `convert` command:

```bash
datasetinsights convert -i <input-directory> -o <output-directory> -f COCO-Instances
```
or
```bash
datasetinsights convert -i <input-directory> -o <output-directory> -f COCO-Keypoints
```

You will need to provide 2D bounding box definition ID in the synthetic dataset. We currently only support 2D bounding box and human keypoint annotations for COCO format.

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
