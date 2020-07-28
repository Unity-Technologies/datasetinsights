Dataset Insights
================
This repo enables users to understand their synthetic datasets by exposing the metrics collected when the dataset
was created e.g. object count, label distribution, etc. The easiest way to use Dataset Insights is
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
  -v $HOME/data:/data \
  -t unitytechnologies/datasetinsights:latest
```
This command mounts directory `$HOME/data` in your local filesystem to `/data` inside the container.
If you are loading a dataset generated locally from a Unity app, replace this path with the root of your app's persistent data folder.

Example persistent data paths from [SynthDet](https://github.com/Unity-Technologies/synthdet):
* OSX: `~/Library/Application\ Support/UnityTechnologies/SynthDet`
* Linux: `$XDG_CONFIG_HOME/unity3d/UnityTechnologies/SynthDet`
* Windows: `%userprofile%\AppData\LocalLow\UnityTechnologies\SynthDet`


2. Go to `http://localhost:8888` in a web browser to open the Jupyter browser.
3. Open and run the example notebook in `/datasetinsights/notebooks/` or create your own.
   (todo replace docker container gcr.io/unity-ai-thea-test/thea with public links)

## Running a Dataset Insights Jupyter Notebook via Google Cloud Platform (GCP)
- To run the notebook on GCP's AI platform follow
[these instructions](https://cloud.google.com/ai-platform/notebooks/docs/custom-container) and use the container `unitytechnologies/datasetinsights:latest`
- Alternately, to run the notebook on kubeflow follow [these steps](https://www.kubeflow.org/docs/notebooks/setup/)

### Download Dataset from Unity Simulation

[Unity Simulation](https://unity.com/products/simulation) provides a powerful platform for running simulations at large scale. You can use the provided cli script to download Perception datasets generated in Unity Simulation:

```bash
python -m datasetinsights.scripts.usim_download \
  --data-root=$HOME/data \
  --run-execution-id=<run-execution-id> \
  --auth-token=<xxx>
```

The `auth-token` can be generated using the Unity Simulation [CLI](https://github.com/Unity-Technologies/Unity-Simulation-Docs/blob/master/doc/cli.md#usim-inspect-auth). This script will download the synthetic dataset for the requested [run-execution-id](https://github.com/Unity-Technologies/Unity-Simulation-Docs/blob/master/doc/cli.md#argument-descriptions).

If the `--include-binary` flag is present, the images will also be downloaded. This might take a long time, depending on the size of the generated dataset.

### Download SynthDet Dataset

Download SynthDet public dataset from GCS, including GroceriesReal and Synthetic dataset. You can use the provided cli script to download public dataset to reproduce our work.

Here is the command line for GroceriesReal dataset download:

```bash
python -m datasetinsights.scripts.public_download \
  --name=GroceriesReal \
  --data-root=$HOME/data \
```
