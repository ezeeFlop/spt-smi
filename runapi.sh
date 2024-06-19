
export STREAMING_PORTS_RANGE=7000-7005
uvicorn spt.api.app:app --reload --app-dir ./src --host 0.0.0.0 --port 8999  