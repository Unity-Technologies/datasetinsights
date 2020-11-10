import tempfile
from yacs.config import CfgNode as CN


from datasetinsights.io.gcs import GCSClient
from datasetinsights.io.download import download_file


def load_config(config):
    if config.startswith("gs://"):
        client = GCSClient()
        with tempfile.NamedTemporaryFile(suffix=".yaml") as tmp_file:
            bucket, key = client._parse(config)
            bucket = client.client.get_bucket(bucket)
            blob = bucket.get_blob(key)
            blob.download_to_filename(tmp_file.name)
            return CN.load_cfg(open(tmp_file.name, "r"))
    elif config.startswith(("https://", "http://")):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_file_name = "config.yaml"
            dest_path = download_file(source_uri=config, dest_path=tmp_dir, file_name=config_file_name)
            return CN.load_cfg(open(dest_path, "rb"))
    else:
        return CN.load_cfg(open(config, "r"))


def override_config(config, override):
    list_of_overrides = override.split(" ")
    overrides = list()
    for override in list_of_overrides:
        if "=" not in override:
            raise ValueError("Override parameter must contain = sign")
        for value in override.split("="):
            overrides.append(value)
    config.merge_from_list(overrides)
    return config
