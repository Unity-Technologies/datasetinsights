.. Thea documentation master file, created by
   sphinx-quickstart on Mon Apr 27 17:25:16 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Dataset Insights
================

Unity Dataset Insights is a python package for understanding synthetic datasets.
This package enables users to analyze synthetic datasets generated using the `Perception SDK <https://github.com/Unity-Technologies/com.unity.perception>`_.

Installation
------------

Dataset Insights maintains a pip package for easy installation. It can work in any standard Python environment using :code:`pip install datasetinsights` command. We support Python 3 (>= 3.7).

Getting Started
---------------

Dataset Statistics
~~~~~~~~~~~~~~~~~~

We provide a sample `notebook <https://github.com/Unity-Technologies/datasetinsights/blob/master/notebooks/SynthDet_Statistics.ipynb>`_ to help you get started with dataset statistics for the `SynthDet <https://github.com/Unity-Technologies/SynthDet>`_ project. We plan to support other sample Unity projects in the future.

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

   from datasetinsights.io.downloader import UnitySimulationDownloader,
   GCSDatasetDownloader, HTTPDatasetDownloader

   downloader = UnitySimulationDownloader(access_token=access_token)
   downloader.download(source_uri=source_uri, output=data_root)

   downloader = GCSDatasetDownloader()
   downloader.download(source_uri=source_uri, output=data_root)

   downloader = HTTPDatasetDownloader()
   downloader.download(source_uri=source_uri, output=data_root)

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
