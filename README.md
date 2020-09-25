# Dataset Insights

Unity Dataset Insights is a python package for understanding synthetic datasets.
This package enables users to analyze synthetic datasets generated using the [Perception SDK](https://github.com/Unity-Technologies/com.unity.perception).

## Installation

Dataset Insights maintains a pip package for easy installation. It can work in any standard Python environment using `pip install datasetinsights` command. We support Python 3 (>= 3.7).

## Getting Started

### Dataset Statistics

We provide a sample [notebook](notebooks/SynthDet_Statistics.ipynb) to help you get started with dataset statistics for the [SynthDet](https://github.com/Unity-Technologies/SynthDet) project. We plan to support other sample Unity projects in the future.

### Dataset Evaluation

Dataset evaluation provides tools to train and evaluate ML models for different datasets. You can run `download`, `train` and `evaluate` commands:

[Download Dataset](https://datasetinsights.readthedocs.io/en/latest/datasetinsights.commands.html#datasetinsights-commands-download)

```bash
datasetinsights download \
  --source-uri=<xxx> \
  --output=$HOME/data
```

[Train](https://datasetinsights.readthedocs.io/en/latest/datasetinsights.commands.html#datasetinsights-commands-train)

```bash
datasetinsights train \
 --config=datasetinsights/configs/faster_rcnn.yaml \
 --train-data=$HOME/data
```

[Evaluate](https://datasetinsights.readthedocs.io/en/latest/datasetinsights.commands.html#datasetinsights-commands-evaluate)

```bash
datasetinsights evaluate \
 --config=datasetinsights/configs/faster_rcnn.yaml \
 --test-data=$HOME/data
```

To learn more, see this [tutorial](https://datasetinsights.readthedocs.io/en/latest/Evaluation_Tutorial.html).

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
