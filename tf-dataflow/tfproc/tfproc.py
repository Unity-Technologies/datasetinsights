#!/usr/bin/env python
import hashlib
import json
import logging
import os
import random

import apache_beam as beam
import click
import tensorflow as tf
from apache_beam import PTransform
from apache_beam.options.pipeline_options import PipelineOptions

_record_size_max_mb = 10.0
_bytes_in_mb = 1000000.0

logging.getLogger().setLevel(logging.INFO)


def _annotations_to_bb_normalized(values, width, height):
    x_mins = [v["x"] / width for v in values]
    x_maxes = [x + v["width"] / width for x, v in zip(x_mins, values)]
    y_mins = [v["y"] / height for v in values]
    y_maxes = [y + v["height"] / height for y, v in zip(y_mins, values)]
    return x_mins, x_maxes, y_mins, y_maxes


def _convert_to_example(filename, image_buffer, height, width, labels, bboxes):
    colorspace = b"RGB"
    channels = 3
    image_format = b"PNG"
    xmins, xmaxes, ymins, ymaxes = bboxes

    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                "image/height": _int64_feature(height),
                "image/width": _int64_feature(width),
                "image/colorspace": _bytes_feature(colorspace),
                "image/channels": _int64_feature(channels),
                "image/object/class/label": _int64_list_feature(
                    labels
                ),  # model expects 1-based
                "image/object/bbox/xmin": _float_list_feature(xmins),
                "image/object/bbox/xmax": _float_list_feature(xmaxes),
                "image/object/bbox/ymin": _float_list_feature(ymins),
                "image/object/bbox/ymax": _float_list_feature(ymaxes),
                "image/format": _bytes_feature(image_format),
                "image/filename": _bytes_feature(
                    bytes(os.path.basename(filename), "utf-8")
                ),
                "image/encoded": _bytes_feature(image_buffer),
            }
        )
    )
    return example


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = (
            value.numpy()
        )  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return _float_list_feature([value])


def _float_list_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return _int64_list_feature([value])


def _int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


class AppParamLister(PTransform):
    def __init__(self, execution_source):
        super().__init__()
        self.execution_source = execution_source

    def expand(self, pcollection):
        app_params_pattern = [
            os.path.join(self.execution_source, "urn:app_params:*/*")
        ]
        logging.info("App params pattern %s", app_params_pattern)

        app_params_coll = [f for f in tf.io.gfile.glob(app_params_pattern[0])]
        logging.info(
            "App param collection %s, length %d",
            app_params_coll,
            len(app_params_coll),
        )

        return pcollection | "Create instances" >> beam.Create(app_params_coll)


class ProcessInstance(beam.DoFn):
    def __init__(self, output_dir, eval_pct, label=None):
        super().__init__(label)
        self.output_dir = output_dir
        self.eval_pct = eval_pct

    def process(self, element, *args, **kwargs):
        annotations_by_img = {}
        captures_pattern = os.path.join(
            element, "attempt:*/Dataset*/captures_*.json"
        )
        for f in tf.io.gfile.glob(captures_pattern):
            capture = tf.io.gfile.GFile(f)
            capture_contents = capture.read()
            annotation_json = json.loads(capture_contents)
            capture_data = annotation_json["captures"]

            # Collect all annotation captures per image file.
            for capture in capture_data:
                filename = os.path.basename(capture["filename"])
                if filename not in annotations_by_img:
                    annotations_by_img[filename] = capture["annotations"]
                else:
                    annotations_by_img[filename] = (
                        annotations_by_img[filename] + capture["annotations"]
                    )

        imgs_pattern = os.path.join(element, "attempt:*/RGB*/rgb_*.png")
        logging.info("img path %s", imgs_pattern)
        counter = 0
        path_hash = hashlib.sha1(element.encode("utf-8")).hexdigest()
        train_dataset_dir = os.path.join(self.output_dir, "train")
        eval_dataset_dir = os.path.join(self.output_dir, "eval")
        tf.io.gfile.makedirs(train_dataset_dir)
        tf.io.gfile.makedirs(eval_dataset_dir)
        train_path = os.path.join(train_dataset_dir, path_hash + ".tfrecord")
        train_writer = tf.io.TFRecordWriter(train_path)
        eval_path = os.path.join(eval_dataset_dir, path_hash + ".tfrecord")
        eval_writer = tf.io.TFRecordWriter(eval_path)

        for img_file in tf.io.gfile.glob(imgs_pattern):
            img_basename = os.path.basename(img_file)
            if img_basename not in annotations_by_img:
                continue

            annotations = annotations_by_img[img_basename]
            with tf.io.gfile.GFile(img_file, mode="rb") as img_f:
                img = img_f.read()
                image = tf.io.decode_png(img)

                height, width = image.shape[0], image.shape[1]
                labels = [v["label_id"] for v in annotations[0]["values"]]
                bboxes = _annotations_to_bb_normalized(
                    annotations[0]["values"], width, height
                )
                example = _convert_to_example(
                    img_file, img, height, width, labels, bboxes
                )
                counter += 1

                if random.uniform(0, 100) < self.eval_pct:
                    eval_writer.write(example.SerializeToString())
                else:
                    train_writer.write(example.SerializeToString())

        logging.info("Completed %d images", counter)
        return [
            dict(
                element=element,
                num_records=counter,
                train_path=train_path,
                eval_path=eval_path,
            )
        ]


def run(source: str, eval_pct: int, beam_options):
    output_dir = os.path.join(source, "output")
    with beam.Pipeline(options=beam_options) as p:
        dataset = (
            p
            | "App param lister" >> AppParamLister(source)
            | "Process app params"
            >> beam.ParDo(ProcessInstance(output_dir, eval_pct))
        )

        dataset | beam.io.WriteToText(os.path.join(output_dir, "result_log"))


@click.command(
    context_settings=dict(ignore_unknown_options=True, allow_extra_args=True)
)
@click.pass_context
@click.option("--source", type=str)
@click.option("--eval-pct", type=int, required=True)
def main(ctx, source, eval_pct):

    # the source is a path to a run execution of the form:
    # gs://xxxxxxxx/projects/xxxxxxxx/run_executions/urn:run_definitions:xxxxxxx/urn:run_executions:xxxxxxx
    extra_args = {
        ctx.args[i]: ctx.args[i + 1] for i in range(0, len(ctx.args) - 1, 2)
    }

    logging.info("source = %s, extra = %s", source, extra_args)

    beam_options = PipelineOptions(ctx.args, save_main_session=True)
    run(source=source, eval_pct=eval_pct, beam_options=beam_options)
