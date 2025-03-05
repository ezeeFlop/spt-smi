#!/bin/bash
source .venv/bin/activate
(cd ./src && python -m spt.services.server --host localhost --port 55001 --type IMAGE_SERVICE)