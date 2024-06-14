# Étape 1: Utiliser une image de base NVIDIA qui supporte Ubuntu 22.04
FROM nvidia/cuda:12.2.0-base-ubuntu22.04

# Définir l'argument de version de Python pour faciliter les mises à jour
ARG PYTHON_VERSION=3.11
ARG DEBIAN_FRONTEND=noninteractive
ENV GDAL_CONFIG=/usr/bin/gdal-config
ENV PYTHONDONTWRITEBYTECODE=1

# Étape 2: Mise à jour des paquets et installation des dépendances nécessaires
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    build-essential \
    libssl-dev \
    libffi-dev \
    wget

# Étape 3: Installation de Python
RUN add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y python${PYTHON_VERSION} \
    && update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 \
#    && ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Étape 4: Installation de Python3 et des dépendances
RUN apt-get update && apt-get install -y python3-venv python3-dev \
    ffmpeg  \
    tesseract-ocr \
    python3-setuptools \
    && apt-get clean 

# Étape 5: Installation de GDAL & TeX
RUN apt-get update && \
    apt-get upgrade --yes && \
    apt-get install -y \
        libcurl4-openssl-dev \
        libssl-dev \
        libxml2-dev \
        zlib1g-dev \
        libfontconfig1-dev \
        libfreetype6-dev \
        libpng-dev \
        libtiff5-dev \
        libjpeg-dev \
        libharfbuzz-dev \
        libfribidi-dev \
        libpq-dev \
        && apt-get clean

# Étape 6: Installation de pandoc et LaTeX
RUN apt-get update && apt-get install -y \
    libgit2-dev \
    libgl1-mesa-glx \
    gdal-bin \
    libgdal-dev \
    libudunits2-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home spt
USER spt
WORKDIR /home/spt

ENV VIRTUALENV=/home/spt/venv
RUN python3 -m venv $VIRTUALENV
ENV PATH="$VIRTUALENV/bin:$PATH"

RUN pip3 --no-cache-dir install torch torchvision torchaudio -f https://download.pytorch.org/whl/cu111/torch_stable.html

COPY --chown=spt pyproject.toml constraints.txt requirements.txt *.sh ./
RUN python -m pip install --no-cache-dir --upgrade pip setuptools

COPY --chown=spt src/ src/
COPY --chown=spt models/ models/

RUN python -m pip install --no-cache-dir . -c constraints.txt && \
    python -m pip install --no-cache-dir -r requirements.txt

EXPOSE 8999 8501
WORKDIR /home/spt/src

CMD ["flask", "--app", "api", "run", \
     "--host", "0.0.0.0", "--port", "8999"]