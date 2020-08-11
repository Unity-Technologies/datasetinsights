import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

METRICS_DIR = os.path.join(PROJECT_ROOT, "metrics")
METRICS_FILE_NAME = "mlpipeline-metrics.json"

GCS_BUCKET = "thea-dev"
GCS_BASE_STR = "gs://"
HTTP_URL_BASE_STR = "http://"
HTTPS_URL_BASE_STR = "https://"

# This is a hack on yacs config system, as it does not allow null values
# in configs. They are working on supporting null values in config
# https://github.com/rbgirshick/yacs/pull/18.
NULL_STRING = "None"

# Root directory of all datasets
# We assume the datasets are stored in the following structure:
# data_root/
#   cityscapes/
#   kitti/
#   nuscenes/
#   synthetic/
#   ...
DEFAULT_DATA_ROOT = "/data"
SYNTHETIC_SUBFOLDER = "synthetic"

# Default Unity Project ID where USim jobs was executed
DEFAULT_PROJECT_ID = "474ba200-4dcc-4976-818e-0efd28efed30"
USIM_API_ENDPOINT = "https://api.simulation.unity3d.com"

# Default Timing text for codetiming.Timer decorator
TIMING_TEXT = "[{name}] elapsed time: {:0.4f} seconds."
