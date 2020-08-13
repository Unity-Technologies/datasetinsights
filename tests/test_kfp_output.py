import os

from datasetinsights.io.kfp_output import KubeflowPipelineWriter


def test_kfp_writer_save_and_overwrite_metric():
    expected_final_dict = {"mAP": 0.667, "mAR": 0.787}
    writer_obj = KubeflowPipelineWriter()
    writer_obj.add_metric(name="mAP", val=0.587)
    writer_obj.add_metric(name="mAR", val=0.787)
    writer_obj.add_metric(name="mAP", val=0.667)
    assert writer_obj.data_dict == expected_final_dict


def test_serialize_and_write_metrics():
    file_name = "metrics.json"
    file_path = os.path.dirname(os.getcwd())
    expected_data = {
        "metrics": [
            {"name": "mAR", "numberValue": 0.787, "format": "RAW"},
            {"name": "mAP", "numberValue": 0.667, "format": "RAW"},
        ]
    }
    writer_obj = KubeflowPipelineWriter(filename=file_name, filepath=file_path)
    writer_obj.add_metric(name="mAR", val=0.787)
    writer_obj.add_metric(name="mAP", val=0.667)
    writer_obj.write_metric()
    assert writer_obj.data == expected_data
    assert os.path.exists(os.path.join(file_name, file_path))
