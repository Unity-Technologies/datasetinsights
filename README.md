Dataset Insights
================
Unity Dataset Insights is a python package for understanding synthetic datasets. This package enables users to analyze synthetic datasets generated using the `Perception SDK <https://github.com/Unity-Technologies/com.unity.perception>`_
for the `Unity game engine <https://unity.com/>`_ and, optionally, at scale using `Unity Simulations <https://unity.com/products/simulation>`_. Dataset Insights exposes the metrics collected when the dataset was created e.g. object count, label distribution, etc. To use our sample notebooks pull our docker image `unitytechnologies/datasetinsights <https://hub.docker.com/r/unitytechnologies/datasetinsights>`_. It can also train and evaluate your model.
Dataset Insights in three points:
* Understand their synthetic datasets by exposing the metrics collected when the dataset
was created e.g. object count, label distribution, etc. 
* Train a model based on the selected dataset
* Evaluate the model

Installing
============
Install and update using `pip <https://pip.pypa.io/en/stable/quickstart/>`_.
```bash
pip install datasetinsights
```

Quick Start
============
The Dataset Insight notebooks assume that the user has already generated a synthetic dataset using the Unity Perception package. To learn how to create a synthetic dataset using Unity please see the
[perception documentation](https://github.com/Unity-Technologies/com.unity.perception).

## Running the Dataset Insights Jupyter Notebook Locally
You can either run the notebook by installing our python package or by using our docker image.

### Running a Notebook Locally Using Docker
The easiest way to use Dataset Insights is to run our jupyter notebook provided in our docker image `unitytechnologies/datasetinsights <https://hub.docker.com/r/unitytechnologies/datasetinsights>`_.

#### Requirements
[Docker](https://docs.docker.com/get-docker/) installed.

#### Steps
1. Run notebook server using docker

```bash
docker run \
  -p 8888:8888 \
  -e GOOGLE_APPLICATION_CREDENTIALS=/tmp/key.json \
  -v $GOOGLE_APPLICATION_CREDENTIALS:/tmp/key.json:ro \
  -v $HOME/data:/data \
  -t unitytechnologies/datasetinsights:latest
```

2. Go to `http://localhost:8888` in a web browser to open the Jupyter browser.
3. Open and run the [example notebook](https://hub.docker.com/r/unitytechnologies/datasetinsights) or create your own.

## Running a Dataset Insights Jupyter Notebook Locally

#### Requirements
Install all dependencies using `poetry <https://python-poetry.org/docs/#installation>`_.
```bash
poetry install
```
#### Steps
1. Run notebook server locally. Go to `dataset-insights/` folder, and type:
```bash
jupyter notebook
```
2. Go to `http://localhost:8888` in a web browser to open the Jupyter browser.

## Download Dataset

You can download your dataset from three different places:
* [Unity Simulation](https://unity.com/products/simulation)
* our public datasets: [UnityGroceries-Real](https://storage.googleapis.com/datasetinsights/data/groceries/v3.zip) and [UnityGroceries-SyntheticSample](https://storage.googleapis.com/datasetinsights/data/synthetic/SynthDet.zip)
* GCS source `gs://`.

```bash
datasetinsights download --source-uri=<xxx> --output=$HOME/data
```

## Train Model

```
datasetinsights train \
 --config=datasetinsights/configs/faster_rcnn.yaml \
 --train-data=path_to_data
```

## Evaluate model

```
datasetinsights evaluate \
 --config=datasetinsights/configs/faster_rcnn.yaml \
 --test-data=<path_to_data>
```
