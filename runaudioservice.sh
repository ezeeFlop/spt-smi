#!/bin/bash
(cd ./src && python -m spt.services.server --host localhost --port 55003 --type AUDIO_SERVICE)