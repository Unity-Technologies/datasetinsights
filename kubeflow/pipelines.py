import kfp.dsl as dsl
import kfp.gcp as gcp

DATA_PATH = "/data"


def volume_op(volume_size):
    # Create large persistant volume to store data.

    vop = dsl.VolumeOp(
        name="pvc",
        resource_name="pvc",
        size=volume_size,
        modes=dsl.VOLUME_MODE_RWO,
    )

    return vop


def download_op(docker, source_uri, output, volume, memory_limit):
    download = dsl.ContainerOp(
        name="download",
        image=docker,
        command=["datasetinsights", "download"],
        arguments=[
            f"--source-uri={source_uri}",
            f"--output={output}",
            f"--include-binary",
        ],
        pvolumes={DATA_PATH: volume},
    )
    download.set_memory_request(memory_limit)
    download.set_memory_limit(memory_limit)

    return download


def train_op(
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
):
    train = dsl.ContainerOp(
        name="train",
        image=docker,
        command=[
            "python",
            "-m",
            "torch.distributed.launch",
            f"--nproc_per_node={num_gpu}" "datasetinsights",
            "train",
        ],
        arguments=[
            f"--config={config}",
            f"--train-data={train_data}",
            f"--val-date={val_data}",
            f"--tb_log_dir={tb_log_dir}",
            f"--checkpoint_dir={checkpoint_dir}",
        ],
        # Refer to pvloume in previous step to explicitly call out dependency
        pvolumes={DATA_PATH: volume},
    )
    # GPU
    train.set_gpu_limit(num_gpu)
    train.add_node_selector_constraint(
        "cloud.google.com/gke-accelerator", gpu_type
    )

    train.set_memory_request(memory_limit)
    train.set_memory_limit(memory_limit)

    return train


@dsl.pipeline(
    name="Train on the SynthDet sample",
    description="Train on the SynthDet sample",
)
def train_on_synthdet_sample(
    docker: str = ("unitytechnologies/datasetinsights:latest"),
    source_uri: str = (
        "https://storage.googleapis.com/datasetinsights/data/"
        "synthetic/SynthDet.zip"
    ),
    tb_log_dir: str = "gs://<bucket>/runs/yyyymmdd-hhmm",
    checkpoint_dir: str = "gs://<bucket>/checkpoints/yyyymmdd-hhmm",
    volume_size: str = "100Gi",
    config: str = "datasetinsights/configs/faster_rcnn_synthetic.yaml",
):
    output = train_data = val_data = DATA_PATH

    # Due to this issue, the following parameters can not be `PipelineParam`
    # https://github.com/kubeflow/pipelines/issues/1956
    # Instead, they have to be configured when the pipeline is compiled.
    memory_limit = "64Gi"
    num_gpu = 8
    gpu_type = "nvidia-tesla-v100"

    # Pipeline definition
    vop = volume_op(volume_size)
    download = download_op(docker, source_uri, output, vop.volume, memory_limit)
    train = train_op(
        docker,
        config,
        train_data,
        val_data,
        tb_log_dir,
        checkpoint_dir,
        download.pvolumes[DATA_PATH],
        memory_limit,
        num_gpu,
        gpu_type,
    )

    # Use GCP Service Accounts to allow writing logs to GCS
    ops = [download, train]
    for op in ops:
        op.apply(gcp.use_gcp_secret("user-gcp-sa"))
