#!/bin/bash
source .venv/bin/activate
(cd ./src && python -m spt.jobs)
#(cd ./src && watchmedo shell-command --patterns="*.py" --command='python -m spt.jobs' .)
