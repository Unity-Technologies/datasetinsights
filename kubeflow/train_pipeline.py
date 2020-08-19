import kfp.dsl as dsl
import kfp.gcp as gcp

MAX_GPU = 8
MAX_MEMORY = "256Gi"


@dsl.pipeline(
    name="train pipeline", description="train model using kubeflow pipeline"
)
def train_pipeline(
    num_proc: int = 8,
    volume_size: str = "2Ti",
    logdir: str = "gs://thea-dev/runs/yyyymmdd-hhmm",
    docker_image: str = (
        "gcr.io/unity-ai-thea-test/datasetinsights:<git-comit-sha>"
    ),
    config_file: str = "datasetinsights/configs/faster_rcnn_synthetic.yaml",
    auth_token: str = "xxxxxx",
    run_execution_id: str = "EjPQYAN",
):
    """Train Pipeline

    This is currently configured as a three-step pipeline. 1) Create
    a persistent volume that can be used to store data. 2) Download
    data for the pipeline. 3) Kick off training jobs.
    """
    # Create large persistant volume to store training data.
    vop = dsl.VolumeOp(
        name="train-pvc",
        resource_name="train-pvc",
        size=volume_size,
        modes=dsl.VOLUME_MODE_RWO,
    )

    # Usim Download
    download = dsl.ContainerOp(
        name="usim download",
        image=docker_image,
        command=["python", "-m", "datasetinsights.scripts.usim_download"],
        arguments=[
            f"--run-execution-id={run_execution_id}",
            f"--include-binary",
            f"--auth-token={auth_token}",
        ],
        pvolumes={"/data": vop.volume},
    )
    # Memory limit of download run
    download.set_memory_limit(MAX_MEMORY)
    # Use GCP Service Accounts to allow access to GCP resources
    download.apply(gcp.use_gcp_secret("user-gcp-sa"))

    # Train
    train = dsl.ContainerOp(
        name="train",
        image=docker_image,
        command=[
            "python",
            "-m",
            "torch.distributed.launch",
            f"--nproc_per_node={num_proc}",
        ],
        arguments=[
            "datasetinsights",
            "train",
            f"--config={config_file}",
            f"--tb-log-dir={logdir}",
        ],
        # Refer to pvloume in previous step to explicitly call out dependency
        pvolumes={"/data": download.pvolumes["/data"]},
    )
    # GPU limit here has to be hard coded integer instead of derived from
    # num_proc, otherwise it will fail kubeflow validation as it will create
    # yaml with palceholder like {{num_proc}}...
    train.set_gpu_limit(MAX_GPU)
    # Request GPUs
    train.add_node_selector_constraint(
        "cloud.google.com/gke-accelerator", "nvidia-tesla-v100"
    )
    # Request master machine with larger memory. Same as gpu limit, this has to
    # be hard coded constants.
    train.set_memory_limit(MAX_MEMORY)
    # Use GCP Service Accounts to allow access to GCP resources
    train.apply(gcp.use_gcp_secret("user-gcp-sa"))


if __name__ == "__main__":
    import kfp.compiler as compiler

    compiler.Compiler().compile(train_pipeline, __file__ + ".tar.gz")
