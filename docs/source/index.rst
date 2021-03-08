.. Thea documentation master file, created by
   sphinx-quickstart on Mon Apr 27 17:25:16 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Dataset Insights
================

Unity Dataset Insights is a python package for downloading, parsing and analyzing synthetic datasets generated using the Unity `Perception SDK <https://github.com/Unity-Technologies/com.unity.perception>`_.

Installation
------------

Dataset Insights maintains a pip package for easy installation. It can work in any standard Python environment using :code:`pip install datasetinsights` command. We support Python 3 (3.7 and 3.8).

Getting Started
---------------

Dataset Statistics
~~~~~~~~~~~~~~~~~~
We provide a sample `notebook <https://github.com/Unity-Technologies/datasetinsights/blob/master/notebooks/Perception_Statistics.ipynb>`_  to help you load synthetic datasets generated using `Perception package <https://github.com/Unity-Technologies/com.unity.perception>`_  and visualize dataset statistics. We plan to support other sample Unity projects in the future.

Dataset Download
~~~~~~~~~~~~~~~~~~

You can download the datasets from HTTP(s), GCS, and Unity simulation projects using the download command from `CLI` or `API`.

`CLI <https://datasetinsights.readthedocs.io/en/latest/datasetinsights.commands.html#datasetinsights-commands-download>`_

.. code-block:: bash

   datasetinsights download \
      --source-uri=<xxx> \
      --output=$HOME/data

`API <https://datasetinsights.readthedocs.io/en/latest/datasetinsights.io.downloader.html#module-datasetinsights.io.downloader.gcs_downloader>`_

.. code-block:: python3

   from datasetinsights.io.downloader import (UnitySimulationDownloader,
   GCSDatasetDownloader, HTTPDatasetDownloader)

   downloader = UnitySimulationDownloader(access_token=access_token)
   downloader.download(source_uri=source_uri, output=data_root)

   downloader = GCSDatasetDownloader()
   downloader.download(source_uri=source_uri, output=data_root)

   downloader = HTTPDatasetDownloader()
   downloader.download(source_uri=source_uri, output=data_root)

Dataset Explore
~~~~~~~~~~~~~~~~~~

You can explore the downloaded dataset `schema <https://datasetinsights.readthedocs.io/en/latest/Synthetic_Dataset_Schema.html#synthetic-dataset-schema>`_ by using following API:

`Unity Perception <https://datasetinsights.readthedocs.io/en/latest/datasetinsights.datasets.unity_perception.html#datasetinsights-datasets-unity-perception>`_

.. code-block:: python3

   from datasetinsights.datasets.unity_perception import (AnnotationDefinitions,
   MetricDefinitions, Captures, Metrics, Egos, Sensors)

   captures = Captures(data_root="/data", version="my_schema_version")
   captures_df = captures.filter(def_id="my_definition_id")

   metrics = Metrics(data_root="/data", version="my_schema_version")
   metrics_df = metrics.filter_metrics(def_id="my_definition_id")

   annotation_def = AnnotationDefinitions(data_root="/data", version="my_schema_version")
   definition_dict = annotation_def.get_definition(def_id="my_definition_id")

   metric_def = MetricDefinitions(data_root="/data", version="my_schema_version")
   definition_dict = metric_def.get_definition(def_id="my_definition_id")

   egos = Egos(data_root="/data", version="my_schema_version")
   egos_df = egos.load_egos(data_root="/data", version="my_schema_version")

   sensors = Sensors(data_root="/data", version="my_schema_version")
   sensors_df = sensors.load_sensors(data_root="/data", version="my_schema_version")

Contents
========

.. toctree::
   :maxdepth: 3

   modules


.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Getting Started

   SynthDet Guide <https://github.com/Unity-Technologies/SynthDet/blob/master/docs/Readme.md>


.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Synthetic Dataset

   Synthetic_Dataset_Schema


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Citation
==================
If you find this package useful, consider citing it using:

::

   @misc{datasetinsights2020,
       title={Unity {D}ataset {I}nsights Package},
       author={{Unity Technologies}},
       howpublished={\url{https://github.com/Unity-Technologies/datasetinsights}},
       year={2020}
   }
