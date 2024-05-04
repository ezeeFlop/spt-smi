#!/bin/bash
#python -m pip freeze --exclude-editable > constraints.txt

docker login -u verdier1706@gmail.com -p 71662F0e32 docker.io && \
echo "Building docker image..." && \
docker build -t spt-smi:$(git rev-parse --short HEAD) --progress=plain --platform linux/amd64 -f ./docker-compose/Dockerfile . && \
echo "Pushing docker image..." && \
docker tag spt-smi:$(git rev-parse --short HEAD) spongetheory/spt-smi:latest && \
docker push spongetheory/spt-smi:latest && \
echo "Deploying to portainer..." && \
curl -X POST https://portainer.sponge-theory.dev/api/stacks/webhooks/69d44efc-508d-4d54-9f5f-011dbb76ca7b