import argparse

import kfp
import kfp.dsl as dsl
import kfp.gcp as gcp


MEMORY_LIMIT = "64Gi"
PVC_SIZE = "2Ti"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Experiments using Kubeflow Pipeline for profiling"
    )

    job_group = parser.add_argument_group("Job Args")

    # Job parameters
    job_group.add_argument(
        "--docker-image",
        help="Docker registry path.",
        default="gcr.io/unity-ai-thea-test/thea:master",
        type=str,
    )
    job_group.add_argument(
        "--run-execution-id",
        help="USim run execution id.",
        default="wjZR6Zj",
        type=str,
    )
    job_group.add_argument(
        "--auth-token",
        help="USim auth token used to download manifest file.",
        type=str,
        required=True
    )

    kfp_group = parser.add_argument_group("KFP Args")

    # kfp parameters
    kfp_group.add_argument(
        "--exp-name",
        help="Grouping multiple runs into experiments.",
        type=str,
        required=True,
    )
    kfp_group.add_argument(
        "--run-name",
        help="Unique name for this run.",
        type=str,
        required=True,
    )
    args = parser.parse_args()

    return args


@dsl.pipeline(
    name="train pipeline",
    description="train thea model using kubeflow pipeline"
)
def train_pipeline(
    docker_image,
    run_execution_id,
    auth_token,
):
    # Create large persistant volume to store training data.
    vop = dsl.VolumeOp(
        name="train-pvc",
        resource_name="train-pvc",
        size=PVC_SIZE,
        modes=dsl.VOLUME_MODE_RWO
    )

    # USim Download
    download = dsl.ContainerOp(
        name="usim download",
        image=docker_image,
        command=[
            "python",
            "-m",
            "datasetinsights.scripts.usim_download",
        ],
        arguments=[
            f"--run-execution-id={run_execution_id}",
            f"--include-binary",
            f"--auth-token={auth_token}",
        ],
        pvolumes={"/data": vop.volume},
    )
    # Memory request/limit of download run
    download.set_memory_request(MEMORY_LIMIT)
    download.set_memory_limit(MEMORY_LIMIT)
    # Use GCP Service Accounts to allow access to GCP resources
    download.apply(gcp.use_gcp_secret("user-gcp-sa"))

    # Train
    train = dsl.ContainerOp(
        name="train",
        image=docker_image,
        command=[
            "python",
            "-m",
            "datasetinsights.scripts.mprof_captures",
        ],
        arguments=[
            f"--run-execution-id={run_execution_id}",
        ],
        # Refer to pvloume in previous step to explicitly call out dependency
        pvolumes={"/data": download.pvolumes["/data"]},
    )
    # Memory request/limit of this run
    train.set_memory_request(MEMORY_LIMIT)
    train.set_memory_limit(MEMORY_LIMIT)
    # Use GCP Service Accounts to allow access to GCP resources
    train.apply(gcp.use_gcp_secret("user-gcp-sa"))


def run_pipeline(args):
    run_name = args.run_name
    del args.run_name
    exp_name = args.exp_name
    del args.exp_name

    client = kfp.Client()
    client.create_run_from_pipeline_func(
        train_pipeline,
        arguments=vars(args),
        run_name=run_name,
        experiment_name=exp_name
    )


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
