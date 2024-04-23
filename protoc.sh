#!/bin/bash
python -m grpc_tools.protoc -I./proto --python_out=./src/ --grpc_python_out=./src/ ./proto/imagegeneration.proto
python -m grpc_tools.protoc -I./proto --python_out=./src/ --grpc_python_out=./src/ ./proto/llmgeneration.proto
python -m grpc_tools.protoc -I./proto --python_out=./src/ --grpc_python_out=./src/ ./proto/generic.proto
