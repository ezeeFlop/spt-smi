#!/bin/bash
(cd ./src && python -m spt.services.server --host localhost --port 55002 --type LLM_SERVICE)