#!/bin/bash

docker login -u verdier1706@gmail.com -p 71662F0e32 docker.io && \
echo "Building docker image..." && \
docker build -t spt-smi:$(git rev-parse --short HEAD) --platform linux/amd64 . && \
echo "Pushing docker image..." && \
docker tag spt-smi:$(git rev-parse --short HEAD) spongetheory/spt-smi:latest && \
docker push spongetheory/spt-smi:latest && \
echo "Deploying to portainer..." && \
curl -X POST https://portainer.sponge-theory.dev/api/stacks/webhooks/3014ae70-cfe4-4d5d-b6ab-e81b1e73b327