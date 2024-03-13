#!/bin/bash
(cd ./src && python -m spt.jobs)
#(cd ./src && watchmedo shell-command --patterns="*.py" --command='python -m spt.jobs' .)
