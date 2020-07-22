.. Thea documentation master file, created by
   sphinx-quickstart on Mon Apr 27 17:25:16 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the Unity Dataset Insights documentation!
====================================================
Unity Dataset Insights is a python package for understanding synthetic datasets.
This package enables users to analyze synthetic datasets generated using the `Perception SDK <https://github.com/Unity-Technologies/com.unity.perception>`_
for the `Unity game engine <https://unity.com/>`_ and, optionally, at scale using
`Unity Simulations <https://unity.com/products/simulation>`_. Dataset Insights exposes the metrics collected when the dataset
was created e.g. object count, label distribution, etc. To use our sample notebooks pull our docker image
`unitytechnologies/datasetinsights <https://hub.docker.com/r/unitytechnologies/datasetinsights>`_



Workflow
========

Download From Unity Simulations
-------------------------------
If you used `Unity Simulations <https://unity.com/products/simulation>`_ to generate your dataset,
then you can use :meth:`~datasetinsights.data.simulation.download.download_manifest` method and the :class:`~datasetinsights.data.simulation.download.Downloader` object to download your data.

.. code-block:: python

    manifest_file = os.path.join(data_volume, f"{run_execution_id}.csv")
    download_manifest(run_execution_id, manifest_file, auth_token)
    dl = Downloader(manifest_file, data_root, use_cache=True)
The first line of code defines your local path for the manifest file.
The second line downloads the manifest from Unity Simulations to the local path.
The third line, initializes the :class:`~datasetinsights.data.simulation.download.Downloader` object.
Once initialized, the downloader class can :meth:`~datasetinsights.data.simulation.download.Downloader.download_all`,
to download all files associated with that dataset. Or use any of the following methods:
:meth:`~datasetinsights.data.simulation.download.Downloader.download_captures`,
:meth:`~datasetinsights.data.simulation.download.Downloader.download_metrics`,
to download the :ref:`captures` or :ref:`metrics` respectively.


Load Dataset Metadata
---------------------
Once the dataset's metadata is downloaded, it can be loaded for statistics using the :mod:`~datasetinsights.data.simulation` module.
Annotation and metric definitions are loaded into pandas dataframes using :class:`~datasetinsights.data.simulation.references.AnnotationDefinitions` and :class:`~datasetinsights.data.simulation.references.MetricDefinitions` respectively.
These definition ids correspond to the different kinds of metrics and annotations that are present in the dataset.
You can then use these ids to filter the statistics to correspond to the specific metric or annotation you are interested in.
For instance, for the `SynthDet <https://github.com/Unity-Technologies/SynthDet>`_ dataset, you can access the dataframe corresponding to foreground placement
by filtering on that id.
The code snippet below shows how you could filter the metrics to just display information about the foreground object placement and then filter again to just show the rotations of the foreground objects present in the SynthDet dataset.

.. code-block:: python

   foreground_placement_info_definition_id = "061e08cc-4428-4926-9933-a6732524b52b"
   columns = ("x_rot", "y_rot", "z_rot")
   filtered_metrics = metrics.filter_metrics(foreground_placement_info_definition_id)
   rotation_df = pd.DataFrame(filtered_metrics["rotation"].to_list(), columns=columns)



SynthDet Quick Start
===========
To get started using a sample project to generate synthetic data and to explore your dataset using this package please follow the  `SynthDet Started Guide <https://github.com/Unity-Technologies/SynthDet/blob/master/docs/Readme.md>`_


.. toctree::
   :maxdepth: 3
   :caption: Package Contents:

   data

.. toctree::
   :titlesonly:
   :caption: Additional docs:

   Synthetic_Dataset_Schema



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
