#!/bin/bash
(cd ./src && python -m spt.services.generic.service --host localhost --port 55002 --type LLM_SERVICE)