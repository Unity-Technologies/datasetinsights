DEFAULT_DATA_ROOT = "/data"

# Default perception schema version
DEFAULT_PERCEPTION_VERSION = "0.0.1"

# Unity Simulation API endpoints
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

COCO_KEYPOINTS = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

COCO_SKELETON = [
    [16, 14],
    [14, 12],
    [17, 15],
    [15, 13],
    [12, 13],
    [6, 12],
    [7, 13],
    [6, 7],
    [6, 8],
    [7, 9],
    [8, 10],
    [9, 11],
    [2, 3],
    [1, 2],
    [1, 3],
    [2, 4],
    [3, 5],
    [4, 6],
    [5, 7],
]
