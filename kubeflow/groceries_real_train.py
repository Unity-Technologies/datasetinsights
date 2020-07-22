import kfp.dsl as dsl
import kfp.gcp as gcp

NUM_GPU = 1
GPU_TYPE = "nvidia-tesla-v100"
MEMORY_LIMIT = "32Gi"
DEFAULT_TIMEOUT = 259200  # Timeout in 3 days (seconds)


@dsl.pipeline(
    name="train groceries real",
    description=(
        "Train pipeline for Faster RCNN estimator using groceries real dataset"
    ),
)
def train_pipeline(
    num_proc: int = 1,
    volume_size: str = "50Gi",
    logdir: str = "gs://thea-dev/runs/yyyymmdd-hhmm",
    docker_image: str = (
        "gcr.io/unity-ai-thea-test/datasetinsights:<git-comit-sha>"
    ),
    config_file: str = (
        "datasetinsights/configs/faster_rcnn_google_groceries_real.yaml"
    ),
    epochs: int = 50,
):
    # Train
    train = dsl.ContainerOp(
        name="train",
        image=docker_image,
        command=["python", "-m", "datasetinsights.cli"],
        arguments=[
            "--local_rank=0",
            "train",
            f"--config={config_file}",
            f"--logdir={logdir}",
            "train.epochs",
            epochs,
        ],
    )
    # Request GPU
    train.set_gpu_limit(NUM_GPU)
    train.add_node_selector_constraint(
        "cloud.google.com/gke-accelerator", GPU_TYPE
    )
    # Request Memory
    train.set_memory_request(MEMORY_LIMIT)
    train.set_memory_limit(MEMORY_LIMIT)
    # Use GCP Service Accounts to access to GCP resources
    train.apply(gcp.use_gcp_secret("user-gcp-sa"))
    #
    train.set_timeout(DEFAULT_TIMEOUT)


if __name__ == "__main__":
    import kfp.compiler as compiler

    compiler.Compiler().compile(train_pipeline, __file__ + ".tar.gz")
