import os
from datetime import datetime

TIMESTAMP_SUFFIX = datetime.now().strftime("%Y%m%d-%H%M%S")
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

GCS_BASE_STR = "gs://"
HTTP_URL_BASE_STR = "http://"
HTTPS_URL_BASE_STR = "https://"
LOCAL_FILE_BASE_STR = "file://"

NULL_STRING = "None"

DEFAULT_DATA_ROOT = "/data"
SYNTHETIC_SUBFOLDER = "synthetic"

# Default Unity Project ID where USim jobs was executed
DEFAULT_PROJECT_ID = "474ba200-4dcc-4976-818e-0efd28efed30"
USIM_API_ENDPOINT = "https://api.simulation.unity3d.com"

# Default Timing text for codetiming.Timer decorator
TIMING_TEXT = "[{name}] elapsed time: {:0.4f} seconds."

# Click CLI context settings
CONTEXT_SETTINGS = {
    "help_option_names": ["-h", "--help"],
    "show_default": True,
    "ignore_unknown_options": True,
    "allow_extra_args": True,
}
DEFAULT_DATASET_VERSION = "latest"
