import kfp.dsl as dsl
import kfp.gcp as gcp

DATA_PATH = "/data"


def volume_op(*, volume_size):
    """ Create Kubernetes persistant volume to store data.

    https://kubernetes.io/docs/concepts/storage/volumes/#persistentvolumeclaim

    Args:
        volume_size (str): Size of the persistant volume claim.

    Returns:
        kfp.dsl.VolumeOp: Represents an op which will be translated into a
            resource template which will be creating a PVC.
    """
    vop = dsl.VolumeOp(
        name="pvc",
        resource_name="pvc",
        size=volume_size,
        modes=dsl.VOLUME_MODE_RWO,
    )

    return vop


def download_op(*, docker, source_uri, output, volume, memory_limit):
    """ Create a Kubeflow ContainerOp to download a dataset.

    Args:
        docker (str): Docker image registry URI.
        source_uri (str): Source URI of the dataset.
        output (str): Path where dataset should be downloaded.
        volume (kfp.dsl.PipelineVolume): The volume where data will be stored.
        memory_limit (str): Set memory limit for this operator. For simplicity,
            we set memory_request = memory_limit.

    Returns:
        kfp.dsl.ContainerOp: Represents an op implemented by a container image
            to download a dataset.
    """
    arguments = [
        f"--source-uri={source_uri}",
        f"--output={output}",
        f"--include-binary",
    ]

    download = dsl.ContainerOp(
        name="download",
        image=docker,
        command=["datasetinsights", "download"],
        arguments=arguments,
        pvolumes={DATA_PATH: volume},
    )
    download.set_memory_request(memory_limit)
    download.set_memory_limit(memory_limit)
    download.apply(gcp.use_gcp_secret("user-gcp-sa"))

    return download


def train_op(
    *,
    docker,
    config,
    train_data,
    val_data,
    tb_log_dir,
    checkpoint_dir,
    volume,
    memory_limit,
    num_gpu,
    gpu_type,
    checkpoint_file=None,
):
    """ Create a Kubeflow ContainerOp to train an estimator.

    Args:
        docker (str): Docker image registry URI.
        config (str): Path to estimator config file.
        train_data (str): Path to train dataset directory.
        val_data (str): Path to val dataset directory.
        checkpoint_file (str): Path to an estimator checkpoint file.
            If specified, model will resume from previous checkpoints.
        tb_log_dir (str): Path to tensorboard log directory.
        checkpoint_dir (str): Path to checkpoint file directory.
        volume (kfp.dsl.PipelineVolume): The volume where datasets are stored.
        memory_limit (str): Set memory limit for this operator. For simplicity,
            we set memory_request = memory_limit.
        num_gpu (int): Set the number of GPU for this operator
        gpu_type (str): Set the type of GPU

    Returns:
        kfp.dsl.ContainerOp: Represents an op implemented by a container image
            to train an estimator.
    """
    if num_gpu > 1:
        command = [
            "python",
            "-m",
            "torch.distributed.launch",
            f"--nproc_per_node={num_gpu}",
            "--use_env",
            "datasetinsights",
            "train",
        ]
    else:
        command = ["datasetinsights", "train"]

    arguments = [
        f"--config={config}",
        f"--train-data={train_data}",
        f"--val-data={val_data}",
        f"--tb-log-dir={tb_log_dir}",
        f"--checkpoint-dir={checkpoint_dir}",
    ]
    if checkpoint_file:
        arguments.append(f"--checkpoint-file={checkpoint_file}")

    train = dsl.ContainerOp(
        name="train",
        image=docker,
        command=command,
        arguments=arguments,
        pvolumes={DATA_PATH: volume},
    )
    # GPU
    train.set_gpu_limit(num_gpu)
    train.add_node_selector_constraint(
        "cloud.google.com/gke-accelerator", gpu_type
    )

    train.set_memory_request(memory_limit)
    train.set_memory_limit(memory_limit)

    train.apply(gcp.use_gcp_secret("user-gcp-sa"))

    return train


def evaluate_op(
    *,
    docker,
    config,
    checkpoint_file,
    test_data,
    tb_log_dir,
    volume,
    memory_limit,
    num_gpu,
    gpu_type,
):
    """ Create a Kubeflow ContainerOp to evaluate an estimator.

    Args:
        docker (str): Docker image registry URI.
        config (str): Path to estimator config file.
        checkpoint_file (str): Path to an estimator checkpoint file.
        test_data (str): Path to test dataset directory.
        tb_log_dir (str): Path to tensorbload log directory.
        checkpoint_dir (str): Path to checkpoint file directory.
        volume (kfp.dsl.PipelineVolume): The volume where datasets are stored.
        memory_limit (str): Set memory limit for this operator. For simplicity,
            we set memory_request = memory_limit.
        num_gpu (int): Set the number of GPU for this operator
        gpu_type (str): Set the type of GPU

    Returns:
        kfp.dsl.ContainerOp: Represents an op implemented by a container image
            to evaluate an estimator.
    """
    evaluate = dsl.ContainerOp(
        name="evaluate",
        image=docker,
        command=["datasetinsights", "evaluate"],
        arguments=[
            f"--config={config}",
            f"--checkpoint-file={checkpoint_file}",
            f"--test-data={test_data}",
            f"--tb-log-dir={tb_log_dir}",
        ],
        pvolumes={DATA_PATH: volume},
    )
    # GPU
    evaluate.set_gpu_limit(num_gpu)
    evaluate.add_node_selector_constraint(
        "cloud.google.com/gke-accelerator", gpu_type
    )

    evaluate.set_memory_request(memory_limit)
    evaluate.set_memory_limit(memory_limit)

    evaluate.apply(gcp.use_gcp_secret("user-gcp-sa"))

    return evaluate


@dsl.pipeline(
    name="Train on the SynthDet sample",
    description="Train on the SynthDet sample",
)
def train_on_synthdet_sample(
    docker: str = "unitytechnologies/datasetinsights:latest",
    source_uri: str = (
        "https://storage.googleapis.com/datasetinsights/data/"
        "synthetic/SynthDet.zip"
    ),
    config: str = "datasetinsights/configs/faster_rcnn_synthetic.yaml",
    tb_log_dir: str = "gs://<bucket>/runs/yyyymmdd-hhmm",
    checkpoint_dir: str = "gs://<bucket>/checkpoints/yyyymmdd-hhmm",
    volume_size: str = "100Gi",
):
    output = train_data = val_data = DATA_PATH

    # The following parameters can't be `PipelineParam` due to this issue:
    # https://github.com/kubeflow/pipelines/issues/1956
    # Instead, they have to be configured when the pipeline is compiled.
    memory_limit = "64Gi"
    num_gpu = 8
    gpu_type = "nvidia-tesla-v100"

    # Pipeline definition
    vop = volume_op(volume_size=volume_size)
    download = download_op(
        docker=docker,
        source_uri=source_uri,
        output=output,
        volume=vop.volume,
        memory_limit=memory_limit,
    )
    train_op(
        docker=docker,
        config=config,
        train_data=train_data,
        val_data=val_data,
        tb_log_dir=tb_log_dir,
        checkpoint_dir=checkpoint_dir,
        volume=download.pvolumes[DATA_PATH],
        memory_limit=memory_limit,
        num_gpu=num_gpu,
        gpu_type=gpu_type,
    )


@dsl.pipeline(
    name="Evaluate the model", description="Evaluate the model",
)
def evaluate_the_model(
    docker: str = "unitytechnologies/datasetinsights:latest",
    source_uri: str = (
        "https://storage.googleapis.com/datasetinsights/data/groceries/v3.zip"
    ),
    config: str = "datasetinsights/configs/faster_rcnn_groceries_real.yaml",
    checkpoint_file: str = (
        "https://storage.googleapis.com/datasetinsights/models/"
        "fine-tuned-sim2real/FasterRCNN.estimator"
    ),
    tb_log_dir: str = "gs://<bucket>/runs/yyyymmdd-hhmm",
    volume_size: str = "100Gi",
):
    output = test_data = DATA_PATH

    # The following parameters can't be `PipelineParam` due to this issue:
    # https://github.com/kubeflow/pipelines/issues/1956
    # Instead, they have to be configured when the pipeline is compiled.
    memory_limit = "64Gi"
    num_gpu = 1
    gpu_type = "nvidia-tesla-v100"

    # Pipeline definition
    vop = volume_op(volume_size=volume_size)
    download = download_op(
        docker=docker,
        source_uri=source_uri,
        output=output,
        volume=vop.volume,
        memory_limit=memory_limit,
    )
    evaluate_op(
        docker=docker,
        config=config,
        checkpoint_file=checkpoint_file,
        test_data=test_data,
        tb_log_dir=tb_log_dir,
        volume=download.pvolumes[DATA_PATH],
        memory_limit=memory_limit,
        num_gpu=num_gpu,
        gpu_type=gpu_type,
    )


@dsl.pipeline(
    name="Train on real world dataset",
    description="Train on real world dataset",
)
def train_on_real_world_dataset(
    docker: str = "unitytechnologies/datasetinsights:latest",
    source_uri: str = (
        "https://storage.googleapis.com/datasetinsights/data/groceries/v3.zip"
    ),
    config: str = "datasetinsights/configs/faster_rcnn_groceries_real.yaml",
    tb_log_dir: str = "gs://<bucket>/runs/yyyymmdd-hhmm",
    checkpoint_dir: str = "gs://<bucket>/checkpoints/yyyymmdd-hhmm",
    volume_size: str = "100Gi",
):
    output = train_data = val_data = DATA_PATH

    # The following parameters can't be `PipelineParam` due to this issue:
    # https://github.com/kubeflow/pipelines/issues/1956
    # Instead, they have to be configured when the pipeline is compiled.
    memory_limit = "64Gi"
    num_gpu = 1
    gpu_type = "nvidia-tesla-v100"

    # Pipeline definition
    vop = volume_op(volume_size=volume_size)
    download = download_op(
        docker=docker,
        source_uri=source_uri,
        output=output,
        volume=vop.volume,
        memory_limit=memory_limit,
    )
    train_op(
        docker=docker,
        config=config,
        train_data=train_data,
        val_data=val_data,
        tb_log_dir=tb_log_dir,
        checkpoint_dir=checkpoint_dir,
        volume=download.pvolumes[DATA_PATH],
        memory_limit=memory_limit,
        num_gpu=num_gpu,
        gpu_type=gpu_type,
    )


@dsl.pipeline(
    name="Train on Synthetic + Real World Dataset",
    description="Train on Synthetic + Real World Dataset",
)
def train_on_synthetic_and_real_dataset(
    docker: str = "unitytechnologies/datasetinsights:latest",
    source_uri: str = (
        "https://storage.googleapis.com/datasetinsights/data/groceries/v3.zip"
    ),
    config: str = "datasetinsights/configs/faster_rcnn_fine_tune.yaml",
    checkpoint_file: str = (
        "https://storage.googleapis.com/datasetinsights/models/Synthetic"
        "/FasterRCNN.estimator"
    ),
    tb_log_dir: str = "gs://<bucket>/runs/yyyymmdd-hhmm",
    checkpoint_dir: str = "gs://<bucket>/checkpoints/yyyymmdd-hhmm",
    volume_size: str = "100Gi",
):
    output = train_data = val_data = DATA_PATH

    # The following parameters can't be `PipelineParam` due to this issue:
    # https://github.com/kubeflow/pipelines/issues/1956
    # Instead, they have to be configured when the pipeline is compiled.
    memory_limit = "64Gi"
    num_gpu = 1
    gpu_type = "nvidia-tesla-v100"

    # Pipeline definition
    vop = volume_op(volume_size=volume_size)
    download = download_op(
        docker=docker,
        source_uri=source_uri,
        output=output,
        volume=vop.volume,
        memory_limit=memory_limit,
    )
    train_op(
        docker=docker,
        config=config,
        train_data=train_data,
        val_data=val_data,
        tb_log_dir=tb_log_dir,
        checkpoint_dir=checkpoint_dir,
        volume=download.pvolumes[DATA_PATH],
        memory_limit=memory_limit,
        num_gpu=num_gpu,
        gpu_type=gpu_type,
        checkpoint_file=checkpoint_file,
    )


@dsl.pipeline(
    name="Train on synthetic dataset Unity Simulation",
    description="Train on synthetic dataset Unity Simulation",
)
def train_on_synthetic_dataset_unity_simulation(
    docker: str = "unitytechnologies/datasetinsights:latest",
    project_id: str = "<unity-project-id>",
    run_execution_id: str = "<unity-simulation-run-execution-id>",
    access_token: str = "<unity-simulation-access-token>",
    config: str = "datasetinsights/configs/faster_rcnn_synthetic.yaml",
    tb_log_dir: str = "gs://<bucket>/runs/yyyymmdd-hhmm",
    checkpoint_dir: str = "gs://<bucket>/checkpoints/yyyymmdd-hhmm",
    volume_size: str = "100Gi",
):
    output = train_data = val_data = DATA_PATH

    # The following parameters can't be `PipelineParam` due to this issue:
    # https://github.com/kubeflow/pipelines/issues/1956
    # Instead, they have to be configured when the pipeline is compiled.
    memory_limit = "64Gi"
    num_gpu = 8
    gpu_type = "nvidia-tesla-v100"

    source_uri = f"usim://{access_token}@{project_id}/{run_execution_id}"

    # Pipeline definition
    vop = volume_op(volume_size=volume_size)
    download = download_op(
        docker=docker,
        source_uri=source_uri,
        output=output,
        volume=vop.volume,
        memory_limit=memory_limit,
    )
    train_op(
        docker=docker,
        config=config,
        train_data=train_data,
        val_data=val_data,
        tb_log_dir=tb_log_dir,
        checkpoint_dir=checkpoint_dir,
        volume=download.pvolumes[DATA_PATH],
        memory_limit=memory_limit,
        num_gpu=num_gpu,
        gpu_type=gpu_type,
    )


@dsl.pipeline(
    name="Train AB test on synthetic dataset Unity Simulation",
    description="Train AB test on synthetic dataset Unity Simulation",
)
def train_ab_test_on_synthetic_dataset_unity_simulation(
    docker: str = "unitytechnologies/datasetinsights:latest",
    project_id: str = "<unity-project-id>",
    run_execution_id: str = "<unity-simulation-run-execution-id>",
    access_token: str = "<unity-simulation-access-token>",
    config: str = "datasetinsights/configs/faster_rcnn_synthetic_ab_test.yaml",
    tb_log_dir: str = "gs://<bucket>/runs/yyyymmdd-hhmm",
    checkpoint_dir: str = "gs://<bucket>/checkpoints/yyyymmdd-hhmm",
    volume_size: str = "1.5Ti",
):
    output = train_data = val_data = DATA_PATH

    # The following parameters can't be `PipelineParam` due to this issue:
    # https://github.com/kubeflow/pipelines/issues/1956
    # Instead, they have to be configured when the pipeline is compiled.
    memory_limit = "256Gi"
    num_gpu = 8
    gpu_type = "nvidia-tesla-v100"

    source_uri = f"usim://{access_token}@{project_id}/{run_execution_id}"

    # Pipeline definition
    vop = volume_op(volume_size=volume_size)
    download = download_op(
        docker=docker,
        source_uri=source_uri,
        output=output,
        volume=vop.volume,
        memory_limit=memory_limit,
    )
    train_op(
        docker=docker,
        config=config,
        train_data=train_data,
        val_data=val_data,
        tb_log_dir=tb_log_dir,
        checkpoint_dir=checkpoint_dir,
        volume=download.pvolumes[DATA_PATH],
        memory_limit=memory_limit,
        num_gpu=num_gpu,
        gpu_type=gpu_type,
    )
