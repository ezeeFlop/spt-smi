#!/bin/bash
#python -m pip freeze --exclude-editable > constraints.txt

docker login -u verdier1706@gmail.com -p ${DOCKER_TOKEN} docker.io && \
echo "Building docker image..." && \
docker build -t spt-smi:$(git rev-parse --short HEAD) --platform linux/amd64 -f ./docker-compose/Dockerfile . && \
echo "Pushing docker image..." && \
docker tag spt-smi:$(git rev-parse --short HEAD) spongetheory/spt-smi:latest && \
docker push spongetheory/spt-smi:latest && \
echo "Deploying to portainer..." && \
curl -X POST ${HOOK_URL}