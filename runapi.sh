
export STREAMING_PORTS_RANGE=7000-7005
uvicorn api:app --reload --app-dir ./src --host 0.0.0.0 --port 8999  