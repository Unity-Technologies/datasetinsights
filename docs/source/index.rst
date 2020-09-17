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

Dataset Evaluation
~~~~~~~~~~~~~~~~~~

Dataset evaluation provides tools to train and evaluate ML models for different datasets. You can run :code:`download`, :code:`train` and :code:`evaluate` commands:

`Download Dataset <https://datasetinsights.readthedocs.io/en/latest/datasetinsights.commands.html#datasetinsights-commands-download>`_

.. code-block:: bash

   datasetinsights download \
      --source-uri=<xxx> \
      --output=$HOME/data

`Train <https://datasetinsights.readthedocs.io/en/latest/datasetinsights.commands.html#datasetinsights-commands-train>`_

.. code-block:: bash

   datasetinsights train \
      --config=datasetinsights/configs/faster_rcnn.yaml \
      --train-data=$HOME/data

`Evaluate <https://datasetinsights.readthedocs.io/en/latest/datasetinsights.commands.html#datasetinsights-commands-evaluate>`_

.. code-block:: bash

   datasetinsights evaluate \
      --config=datasetinsights/configs/faster_rcnn.yaml \
      --test-data=$HOME/data

To learn more, see this `tutorial <https://datasetinsights.readthedocs.io/en/latest/Evaluation_Tutorial.html>`_.


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
   Evaluation_Tutorial


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
