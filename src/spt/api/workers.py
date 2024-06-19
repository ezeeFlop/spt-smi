
from fastapi import HTTPException, Request
from spt.models.workers import WorkerConfigs

workers_configurations = WorkerConfigs.get_configs()

async def validate_worker_exists(request: Request):
    data = await request.json()
    worker_id = data.get("worker_id")
    if worker_id not in workers_configurations.workers_configs:
        raise HTTPException(
            status_code=404, detail=f"Worker configuration for model {worker_id} not found")
    return worker_id