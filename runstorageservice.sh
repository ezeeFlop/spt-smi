#!/bin/bash
source .venv/bin/activate
uvicorn spt.services.storage.api:app --reload --app-dir ./src --host 0.0.0.0 --port 8909  