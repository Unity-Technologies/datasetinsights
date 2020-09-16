Dataset Insights
================
Unity Dataset Insights is a python package for understanding synthetic datasets. This package enables users to analyze synthetic datasets generated using the `Perception SDK <https://github.com/Unity-Technologies/com.unity.perception>`_
for the `Unity game engine <https://unity.com/>`_ and, optionally, at scale using `Unity Simulations <https://unity.com/products/simulation>`_. Dataset Insights exposes the metrics collected when the dataset was created e.g. object count, label distribution, etc. To use our sample notebooks pull our docker image `unitytechnologies/datasetinsights <https://hub.docker.com/r/unitytechnologies/datasetinsights>`_. It can also train and evaluate your model.
Dataset Insights in three points:
* Understand their synthetic datasets by exposing the metrics collected when the dataset
was created e.g. object count, label distribution, etc. 
* Train a model based on the selected dataset
* Evaluate the model
The easiest way to use Dataset Insights is
to run our jupyter notebook provided in our docker image `unitytechnologies/datasetinsights`

Requirements
============

The Dataset Insight notebooks assume that the user has already generated a synthetic dataset using the Unity Perception package.
To learn how to create a synthetic dataset using Unity please see the
[perception documentation](https://github.com/Unity-Technologies/com.unity.perception).


## Running the Dataset Insights Jupyter Notebook Locally
You can either run the notebook by installing our python package or by using our docker image.

### Running a Notebook Locally Using Docker

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
This command mounts directory `$HOME/data` in your local filesystem to `/data` inside the container.
If you are loading a dataset generated locally from a Unity app, replace this path with the root of your app's persistent data folder. This command assumes you have an environment variable GOOGLE_APPLICATION_CREDENTIALS in the host machine that points to a GCP service account credential file.

Example persistent data paths from [SynthDet](https://github.com/Unity-Technologies/synthdet):
* OSX: `~/Library/Application\ Support/UnityTechnologies/SynthDet`
* Linux: `$XDG_CONFIG_HOME/unity3d/UnityTechnologies/SynthDet`
* Windows: `%userprofile%\AppData\LocalLow\UnityTechnologies\SynthDet`


2. Go to `http://localhost:8888` in a web browser to open the Jupyter browser.
3. Open and run the [example notebook](https://hub.docker.com/r/unitytechnologies/datasetinsights) or create your own.

## Running a Dataset Insights Jupyter Notebook via Google Cloud Platform (GCP)
- To run the notebook on GCP's AI platform follow
[these instructions](https://cloud.google.com/ai-platform/notebooks/docs/custom-container) and use the container `unitytechnologies/datasetinsights:latest`
- Alternately, to run the notebook on kubeflow follow [these steps](https://www.kubeflow.org/docs/notebooks/setup/)

### Download Dataset from Unity Simulation

[Unity Simulation](https://unity.com/products/simulation) provides a powerful platform for running simulations at large scale. You can use the provided cli script to download Perception datasets generated in Unity Simulation:

```bash
datasetinsights download \
  --source-uri=<xxx> \
  --output=$HOME/data \
  --access-token=<xxx>
```
The `source-uri` is the URI of where this data should be downloaded. Supported source uri patterns ^gs://|^http(s)?://|^usim://. The `output` is the directory on localhost where datasets should be (Default: /data). The `access-token` can be generated using the Unity Simulation [CLI](https://github.com/Unity-Technologies/Unity-Simulation-Docs/blob/master/doc/cli.md#usim-inspect-auth). This script will download the synthetic dataset for the requested [run-execution-id](https://github.com/Unity-Technologies/Unity-Simulation-Docs/blob/master/doc/cli.md#argument-descriptions).

If the `--include-binary` flag is present, the images will also be downloaded. This might take a long time, depending on the size of the generated dataset.

## Download dataset

Depending on where the dataset is stored there are different options for downloading datasets

You can specify project_it, run_execution_id, access_token in source-uri

Downloading from Unity Simulation
```
datasetinsights download
--source-uri=usim://<access_token>@<project_id>/<run_execution_id>
--output=$HOME/data
```
Alternatively, you can also override access_token such as
```
datasetinsights download --source-uri=usim://<project_id>/<run_execution_id>
 --output=$HOME/data --access-token=<access_token>
```
Downloading from a http source:
```
datasetinsights download --source-uri=http://url.to.file.zip
 --output=$HOME/data
```
Available public datasets:
* UnityGroceries-Real: `https://storage.googleapis.com/datasetinsights/data/groceries/v3.zip`
* UnityGroceries-SyntheticSample: `https://storage.googleapis.com/datasetinsights/data/synthetic/SynthDet.zip`

Downloading from a gcs source:
```
datasetinsights download --source-uri=gs://url/to/file.zip
 --output=$HOME/data

datasetinsights download --source-uri=gs://url/to/folder
 --output=$HOME/data
```

## Train Model

```
datasetinsights train \
 --config=datasetinsights/configs/faster_rcnn.yaml \
 --train-data=path_to_data
```

## Evaluate model

```
datasetinsights train \
 --config=datasetinsights/configs/faster_rcnn.yaml \
 --test-data=path_to_data
```
