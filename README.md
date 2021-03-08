# Dataset Insights

Unity Dataset Insights is a python package for downloading, parsing and analyzing synthetic datasets generated using the Unity [Perception package](https://github.com/Unity-Technologies/com.unity.perception).

## Installation

Dataset Insights maintains a pip package for easy installation. It can work in any standard Python environment using `pip install datasetinsights` command. We support Python 3 (3.7 and 3.8).

## Getting Started

### Dataset Statistics

We provide a sample [notebook](notebooks/Perception_Statistics.ipynb) to help you load synthetic datasets generated using [Perception package](https://github.com/Unity-Technologies/com.unity.perception) and visualize dataset statistics. We plan to support other sample Unity projects in the future.

### Dataset Download

You can download the datasets from HTTP(s), GCS, and Unity simulation projects using the 'download' command from CLI or API.

[CLI](https://datasetinsights.readthedocs.io/en/latest/datasetinsights.commands.html#datasetinsights-commands-download)

```bash
datasetinsights download \
  --source-uri=<xxx> \
  --output=$HOME/data
```
[Programmatically](https://datasetinsights.readthedocs.io/en/latest/datasetinsights.io.downloader.html#module-datasetinsights.io.downloader.gcs_downloader)

```python3

from datasetinsights.io.downloader import (UnitySimulationDownloader,
GCSDatasetDownloader, HTTPDatasetDownloader)

downloader = UnitySimulationDownloader(access_token=access_token)
downloader.download(source_uri=source_uri, output=data_root)

downloader = GCSDatasetDownloader()
downloader.download(source_uri=source_uri, output=data_root)

downloader = HTTPDatasetDownloader()
downloader.download(source_uri=source_uri, output=data_root)

```
### Dataset Explore
You can explore the downloaded dataset [schema](https://datasetinsights.readthedocs.io/en/latest/Synthetic_Dataset_Schema.html#synthetic-dataset-schema) by using following API:

[Unity Perception](https://datasetinsights.readthedocs.io/en/latest/datasetinsights.datasets.unity_perception.html#datasetinsights-datasets-unity-perception)


```python3

from datasetinsights.datasets.unity_perception import (AnnotationDefinitions, 
MetricDefinitions, Captures, Metrics, Egos, Sensors)

captures = Captures(data_root="/data", version="my_schema_version")
captures_df = captures.filter(def_id="my_definition_id")

metrics = Metrics(data_root="/data", version="my_schema_version")
metrics_df = metrics.filter_metrics(def_id="my_definition_id")

annotation_def = AnnotationDefinitions(data_root="/data", version="my_schema_version")
definition_dict = annotation_def.get_definition(def_id="my_definition_id")

metric_def = MetricDefinitions(data_root="/data", version="my_schema_version")
definition_dict = metric_def.get_definition(def_id="my_definition_id")

egos = Egos(data_root="/data", version="my_schema_version")
egos_df = egos.load_egos(data_root="/data", version="my_schema_version")

sensors = Sensors(data_root="/data", version="my_schema_version")
sensors_df = sensors.load_sensors(data_root="/data", version="my_schema_version")

```


## Docker

You can use the pre-build docker image [unitytechnologies/datasetinsights](https://hub.docker.com/r/unitytechnologies/datasetinsights) to run similar commands.

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
