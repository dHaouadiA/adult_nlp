FROM ubuntu:18.04

ENV PYTHON_VERSION 3.7
ENV CONDA_ENV_NAME python_processing
ENV LANG C.UTF-8

RUN apt-get update && apt-get install -y \
    apt-utils \
    ca-certificates \
    apt-transport-https \
    gnupg2 \
    wget \
    unzip \
    curl \
    bzip2 \
    git \
    sudo \
    fonts-liberation

# Installation miniconda3
RUN curl -sSL http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -bfp /usr/local && \
    rm -rf /tmp/miniconda.sh

# Set up conda environment
RUN conda update -y conda && \
    conda create -n ${CONDA_ENV_NAME} python=${PYTHON_VERSION}
ENV PATH /opt/conda/envs/${CONDA_ENV_NAME}/bin:$PATH
RUN echo "source activate ${CONDA_ENV_NAME}" > ~/.bashrc

RUN apt-get update && apt-get install -yq --no-install-recommends \
    build-essential \
    emacs \
    inkscape \
    jed \
    libsm6 \
    libxext-dev \
    libxrender1 \
    lmodern \
    nano \
    netcat \
    pandoc \
    python-dev \
    ffmpeg \
    libgtk2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev

RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/*
RUN conda clean --all --yes
RUN mkdir -p /processing 
WORKDIR /processing
# Expose port & cmd
EXPOSE 15200

COPY . .
RUN pip3 install -r requirements.txt
#RUN pip install pysftp
RUN python -m spacy download en_core_web_sm
#RUN python -m nltk.downloader stopwords

