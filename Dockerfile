FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04

RUN apt-get update \
    && apt-get install -y \
        build-essential \
        curl \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgl1-mesa-dev \
        libffi-dev \
        libzmq3-dev \
        python3.8-dev \
        python3-pip \
    && ln -s /usr/bin/python3.8 /usr/local/bin/python

RUN python -m pip install --upgrade pip
RUN python -m pip install setuptools==60.2.0 cryptography==36.0.1 poetry==1.1.12 notebook==6.4.8

# Add Tini
ENV TINI_VERSION v0.18.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/local/bin/tini
RUN chmod +x /usr/local/bin/tini

WORKDIR /datasetinsights
VOLUME /data /root/.config

COPY poetry.lock pyproject.toml ./
RUN poetry config virtualenvs.create false \
    && poetry install --no-root

COPY . ./
# Run poetry install again to install datasetinsights
RUN poetry config virtualenvs.create false \
    && poetry install

# Use -g to ensure all child process received SIGKILL
ENTRYPOINT ["tini", "-g", "--"]

CMD sh -c "jupyter notebook --notebook-dir=/ --ip=0.0.0.0 --no-browser --allow-root --port=8888 --NotebookApp.token='' --NotebookApp.password='' --NotebookApp.allow_origin='*' --NotebookApp.base_url=${NB_PREFIX}"
