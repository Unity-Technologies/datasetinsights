import datasetinsights.visualization.object_detection as object_detection


def test_generate_scale_data():
    captures = [
        {
            "id": "4521949a-2a71-4c03-beb0-4f6362676639",
            "sequence_id": "e96b97cd-8130-4ab4-a105-1b911a6d912b",
            "sensor": {
                "sensor_id": 1,
                "ego_id": 1,
                "modality": "camera",
                "translation": [0.2, 1.1, 0.3],
                "scale": 1.0,
            },
            "filename": "captures/camera_001.png",
            "format": "PNG",
        },
        {
            "id": "4b35a47a-3f63-4af3-b0e8-e68cb384ad75",
            "sequence_id": "e96b97cd-8130-4ab4-a105-1b911a6d912b",
            "sensor": {
                "sensor_id": 2,
                "ego_id": 1,
                "modality": "lidar",
                "translation": [0.0, 0.0, 0.0],
                "rotation": [0.0, 0.0, 0.0, 0.0],
                "scale": 1.0,
            },
            "ego": {
                "ego_id": 1,
                "translation": [0.12, 0.1, 0.0],
                "rotation": [0.0, 0.15, 0.24, 0.0],
                "velocity": [0.0, 0.0, 0.0],
            },
            "filename": "captures/lidar_000.pcd",
            "format": "PCD",
        },
    ]
    actual_scale = object_detection.ScaleFactor.generate_scale_data(captures)
    expected_scale = [1.0, 1.0]
    assert expected_scale == actual_scale
