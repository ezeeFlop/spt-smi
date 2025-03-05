
export STREAMING_PORTS_RANGE=7000-7005
source .venv/bin/activate
uvicorn spt.api.app:app --reload --app-dir ./src --host 0.0.0.0 --port 8999  