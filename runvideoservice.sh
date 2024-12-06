#!/bin/bash
(cd ./src && python -m spt.services.server --host localhost --port 55005 --type VIDEO_SERVICE)