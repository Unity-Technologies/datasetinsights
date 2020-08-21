Table of contents
=================

- [Contributing to datasetinsights](#contributing-to-datasetinsights)
- [Developing datasetinsights](#developing-datasetinsights)
    - [Add new dependencies](#add-new-dependencies)
- [Codebase structure](#codebase-structure)
- [Train Model](#train-model)
- [Unit testing](#unit-testing)
- [Style Guide](#style-guide)
- [Writing documentation](#writing-documentation)
    - [Building documentation](#building-documentation)

## Contributing to datasetinsights

If you are interested in contributing to datasetinsights, your contributions will fall into two categories:

1. You want to propose a new models/datasets/evaluation metrics and implement it.
2. You want to implement a feature or bug-fix for an outstanding issue.

## Developing datasetinsights

Here are some steps to setup datasetinsights virtual environment with on your machine:

1. Install [poetry](https://python-poetry.org/), [git](https://git-scm.com/) and [pre-commit](https://pre-commit.com/)

2. Create a virtual environment. We recommend using [miniconda](https://docs.conda.io/en/latest/miniconda.html)

```bash
conda create -n dins-dev python=3.7
conda activate dins-dev
```

3. Clone a copy of datasetinsights from source:

```bash
git clone https://github.com/Unity-Technologies/datasetinsights.git
cd datasetinsights
```

Note: clone the repo from `git@gitlab.internal.unity3d.com:machine-learning/thea.git` before datasetinsights source are available on public github.

4. Install datasetinsights in `develop` mode:

```bash
poetry install
```

This will symlink the Python files from the current local source tree into the installed virtual environment install.
The `develop` mode also includes Python packages such as [pytest](https://docs.pytest.org/en/latest/) and [black](https://black.readthedocs.io/en/stable/).

5. Install pre-commit [hook](https://pre-commit.com/#3-install-the-git-hook-scripts) to `.git` folder.

```bash
pre-commit install
# pre-commit installed at .git/hooks/pre-commit
```

### Add new dependencies

Adding new Python dependencies to datasetinsights environment using poetry like:

```
poetry add numpy@^1.18.4
```

Make sure you only add the desired packages instead of adding all dependencies.
Let package management system resolve for dependencies.
See [poetry add](https://python-poetry.org/docs/cli/#add) for detail instructions.

## Codebase structure
The datasetinsights contains the following modules.

datasetinsights
*    [commands](datasetinsights/commands)
        This module contains the cli commands.

*    [configs](datasetinsights/configs)
        This module contains estimator configuration files.

*    [datasets](datasetinsights/datasets)
        This module contains different datasets.
        The dataset classes contain knowledge on how the
        dataset should be loaded into memory.

*    [estimators](datasetinsights/estimators)
        This module contain estimatos are used for
        training and evaluating models on the datasets.

*    [evaluation_metrics](datasetinsights/evaluation_metrics)
        This module contains metrics used by the different
        estimators and are specific in the estimator config file.

*    [io](datasetinsights/io)
        This module contains functionality that relates to
        writing/downloading/uploading to/from different sources.

*    [stats](datasetinsights/stats)
        This module contains code for visualizing and gathering
        statistics on the dataset

## Train Model

```
datasetinsights train \
 --config=datasetinsights/configs/faster_rcnn.yaml \
 --train-data=path_to_data
```

## Unit testing

We use [pytest](https://docs.pytest.org/en/latest/) to run tests located under `tests/`. Run the entire test suite with

```bash
pytest
```

or run individual test files, like:

```bash
pytest tests/test_visual.py
```

for individual test suites.

## Style Guide

We follow Black code [style](https://black.readthedocs.io/en/stable/the_black_code_style.html) for this repository.
The max line length is set at 80.
We enforce this code style using [Black](https://black.readthedocs.io/en/stable/) to format Python code.
In addition to Black, we use [isort](https://github.com/timothycrosley/isort) to sort Python imports.

Before submitting a pull request, run:

```
black .
isort -rc .
flake8
```

Fix all issues that were highlighted by flake8. If you want to skip exceptions such as long url lines in docstring, add `# noqa: E501 <describe reason>` for the specific line violation. See [this](https://flake8.pycqa.org/en/3.1.1/user/ignoring-errors.html) to learn more about how to ignore flake8 errors.

Some editors support automatically formatting on save. For example, in [vscode](https://code.visualstudio.com/docs/python/editing#_formatting)

## Writing documentation

Datasetinsights uses [Google style](http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) for formatting docstrings.
Length of line inside docstrings block must be limited to 80 characters with exceptions such as long urls or tables.

### Building documentation

Follow instructions [here](docs/README.md).
