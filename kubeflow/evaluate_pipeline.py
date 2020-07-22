import kfp.dsl as dsl
import kfp.gcp as gcp

MAX_GPU = 1
MAX_MEMORY = "16Gi"


@dsl.pipeline(
    name="evaluation pipeline",
    description="evaluate model using kubeflow pipeline",
)
def evaluate_pipeline(
    num_proc: int = 1,
    test_split="test",
    logdir: str = "gs://thea-dev/runs/yyyymmdd-hhmm",
    docker_image: str = (
        "gcr.io/unity-ai-thea-test/datasetinsights:<git-comit-sha>"
    ),
    checkpoint_file: str = "",
):
    """Eval Pipeline
    """
    evaluate = dsl.ContainerOp(
        name="evaluate",
        image=docker_image,
        command=["python", "-m", "torch.distributed.launch"],
        arguments=[
            f"--nproc_per_node={num_proc}",
            "-m",
            "datasetinsights.cli",
            "evaluate",
            "--verbose",
            "--config=datasetinsights/configs/faster_rcnn_synthetic.yaml",
            f"--logdir={logdir}",
            f"checkpoint_file",
            checkpoint_file,
            f"test.dataset.args.split",
            test_split,
        ],
    )
    # GPU limit here has to be hard coded integer instead of derived from
    # num_proc, otherwise it will fail kubeflow validation as it will create
    # yaml with palceholder like {{num_proc}}...
    evaluate.set_gpu_limit(MAX_GPU)
    # Request GPUs
    evaluate.add_node_selector_constraint(
        "cloud.google.com/gke-accelerator", "nvidia-tesla-v100"
    )
    # Request master machine with larger memory. Same as gpu limit, this has to
    # be hard coded constants.
    evaluate.set_memory_limit(MAX_MEMORY)
    # Use GCP Service Accounts to allow access to GCP resources
    evaluate.apply(gcp.use_gcp_secret("user-gcp-sa"))


if __name__ == "__main__":
    import kfp.compiler as compiler

    compiler.Compiler().compile(evaluate_pipeline, __file__ + ".tar.gz")
