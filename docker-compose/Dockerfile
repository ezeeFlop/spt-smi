
FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04 AS build

RUN apt-get update && apt-get install -y build-essential git cmake

WORKDIR /sd.cpp

COPY stable-diffusion.cpp .

RUN mkdir build && cd build && cmake .. -DSD_CUBLAS=ON && cmake --build . --config Release

# Étape 1: Utiliser une image de base NVIDIA qui supporte Ubuntu 22.04
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04 AS runtime

COPY --from=build /sd.cpp/build/bin/sd /sd

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
    wget \
    git \
    python3-venv python3-dev \
    ffmpeg \
    tesseract-ocr \
    python3-setuptools \
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
    libgit2-dev \
    libgl1-mesa-glx \
    gdal-bin \
    libgdal-dev \
    libudunits2-dev \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y python${PYTHON_VERSION} \
    && update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Ajouter un utilisateur non-root pour des raisons de sécurité et le mettre dans le groupe docker
RUN groupadd -g 999 docker && \
    useradd --create-home -u 1000 -g docker spt && \
    usermod -aG docker spt
USER spt
WORKDIR /home/spt

# Création et activation d'un environnement virtuel Python
ENV VIRTUALENV=/home/spt/venv
RUN python3 -m venv $VIRTUALENV
ENV PATH="$VIRTUALENV/bin:$PATH"

# Installer les paquets Python de base
RUN pip3 --no-cache-dir install --upgrade pip setuptools

# Installer PyTorch
RUN pip3 --no-cache-dir install torch torchvision torchaudio -f https://download.pytorch.org/whl/cu111/torch_stable.html

# Copier les fichiers de dépendances avant de les installer
COPY --chown=spt pyproject.toml constraints.txt requirements.txt ./
RUN python -m pip install --no-cache-dir . -c constraints.txt && \
    python -m pip install --no-cache-dir -r requirements.txt

# Copier le code source et les modèles
COPY --chown=spt src/ src/
COPY --chown=spt models/ models/
COPY --chown=spt models/nltk/ nltk_data
# Exposer les ports nécessaires
EXPOSE 8999 8501

# Définir le répertoire de travail
WORKDIR /home/spt/src

# Commande pour démarrer l'application
CMD ["flask", "--app", "api", "run", \
    "--host", "0.0.0.0", "--port", "8999"]