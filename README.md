# spt-smi : Sponge Theory Scalable Models Inferences


This framework allow a scalable model to be used for inference on various computations resources (GPU)

# Set up


## RabbitMQ

docker run -d --hostname smi_rabbit --name rabbit -p 15672:15672 -p 5672:5672 -e RABBITMQ_DEFAULT_USER=root -e RABBITMQ_DEFAULT_PASS=jskdljflskdjflkjsqkjflkjqsldf564654 rabbitmq:3-management

## Redis 
docker run -d --name redis-stack-server -p 6379:6379 redis/redis-stack-server:latest


```
 python -m venv .venv
 source .venv/bin/activate
 pip  install -r requirements.txt
```

# api
flask --app src/api run --port 8999 --debug

# Setup project 

python -m pip install --editable .
python -m pip freeze --exclude-editable > constraints.txt

# Install project

python -m pip install -c constraints.txt .

# build Docker Image

## Build docker image for local Docker server 
docker build -t spt-smi:$(git rev-parse --short HEAD) .


## Build docker image for amd64 (intel) useful if building on Mx Mac and distributing to Docker Intel servers
docker build -t spt-smi:$(git rev-parse --short HEAD) --platform linux/amd64 .

## Tag and upload image
docker login -u verdier1706@gmail.com

docker tag spt-smi:$(git rev-parse --short HEAD) spongetheory/spt-smi:$(git rev-parse --short HEAD)
docker tag spt-smi:$(git rev-parse --short HEAD) spongetheory/spt-smi:latest

docker push spt-smi:$(git rev-parse --short HEAD)
docker push spongetheory/spt-smi:latest

# rebuild container
docker compose build
docker compose up --build

