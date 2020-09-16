.. Thea documentation master file, created by
   sphinx-quickstart on Mon Apr 27 17:25:16 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the Unity Dataset Insights documentation!
====================================================
Unity Dataset Insights is a python package for understanding synthetic datasets. This package enables users to analyze synthetic datasets generated using the `Perception SDK <https://github.com/Unity-Technologies/com.unity.perception>`_
for the `Unity game engine <https://unity.com/>`_ and, optionally, at scale using `Unity Simulations <https://unity.com/products/simulation>`_.
Dataset Insights exposes the metrics collected when the dataset was created e.g. object count, label distribution, etc. To use our sample notebooks pull our docker image `unitytechnologies/datasetinsights <https://hub.docker.com/r/unitytechnologies/datasetinsights>`_.
It can also train and evaluate your model.


Getting Started
===============
To get started using a sample project to generate synthetic data and to explore your dataset using this package please follow the `SynthDet Documentation <https://github.com/Unity-Technologies/SynthDet/blob/master/docs/Readme.md>`_


Dataset Evaluation
------------------
To use the pre-compiled pipelines that allow you to evaluate the quality of synthetic dataset, you can follow the :doc:`Evaluation_Tutorial` Documentation.


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
