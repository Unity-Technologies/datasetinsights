LABEL_RANGE = {
    "car": 50,
    "truck": 50,
    "bus": 50,
    "trailer": 50,
    "construction_vehicle": 50,
    "pedestrian": 40,
    "motorcycle": 40,
    "bicycle": 40,
    "traffic_cone": 30,
    "barrier": 30,
}
DIST_FCN = "center_distance"
DIST_THS = [0.5, 1.0, 2.0, 4.0]
MIN_RECALL = 0.1
MIN_PRECISION = 0.1
MAX_BOXES_PER_SAMPLE = 500
MEAN_AP_WEIGHT = 5  # constant used in NuScenes Detection Score (NDS) to scale
# mAP
