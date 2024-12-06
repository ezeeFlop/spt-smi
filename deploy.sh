#!/bin/bash
#python -m pip freeze --exclude-editable > constraints.txt
HOOK_URL=https://portainer.sponge-theory.dev/api/stacks/webhooks/5c447230-9a3f-44e9-af63-38bf187dee8a
echo "Building docker image..." && \
docker build -t spt-smi:$(git rev-parse --short HEAD) --platform linux/amd64 -f ./docker-compose/Dockerfile . && \
echo "Pushing docker image..." && \
docker tag spt-smi:$(git rev-parse --short HEAD) registry.sponge-theory.dev/spt-smi:latest && \
docker push registry.sponge-theory.dev/spt-smi:latest && \
echo "Deploying to portainer..." && \
curl -X POST ${HOOK_URL}