#!/bin/bash
(cd ./src && python -m spt.services.generic.service --host localhost --port 55001 --type IMAGE_SERVICE)