
from fastapi import Header
from fastapi import WebSocket
from keys import API_KEY
from spt.models.jobs import JobsTypes, JobStatuses, JobResponse, JobPriority, JobStorage
from spt.models.audio import SpeechToTextRequest, SpeechToTextResponse
from spt.models.remotecalls import class_to_string
from spt.models.workers import WorkerStreamManageRequest, WorkerStreamManageResponse, WorkerStreamType
from spt.jobs import Job, Jobs
from config import SERVICE_KEEP_ALIVE
import socket
from spt.utils import find_free_port, get_ip
from spt.api.app import app, logger
from spt.api.jobs import submit_job
from spt.api.stream import stream
from spt.api.workers import workers_configurations

async def speech_to_text(request_data: SpeechToTextRequest, worker_id: str, storage_key: str, api_key:str, priority_key: str, keep_alive_key: str, async_key: str):
    job = await Jobs.create_job(
        payload=request_data.model_dump_json(),  # Assuming you serialize to JSON if needed
        type=JobsTypes.audio_generation,
        worker_id=worker_id,
        request_model_class=SpeechToTextRequest,
        response_model_class=SpeechToTextResponse,
        storage=storage_key,
        keep_alive=keep_alive_key
    )

    return await submit_job(job, async_key, priority_key)

async def speech_to_text_job(job_id: str, accept:Header, api_key:str):
    job = Job(id=job_id, type=JobsTypes.audio_generation,
              response_model_class=class_to_string(SpeechToTextResponse))
    status = await app.state.jobs[JobsTypes.audio_generation].get_job_status(job)

    if status.status == JobStatuses.completed:
        result = await app.state.jobs[JobsTypes.audio_generation].get_job_result(job)
        return result
    return JobResponse(id=job.id, status=status.status, type=status.type, message=status.message)

async def speech_to_text_stream(websocket:WebSocket, worker_id: str, timeout:int):

    api_key = websocket.headers.get("x-smi-key")
    logger.info(
        f"WebSocket connection attempt worker_id: {worker_id}")

    if api_key != API_KEY:
        await websocket.close(code=1008, reason="API key invalid")
        return

    if worker_id not in workers_configurations.workers_configs:
        await websocket.close(code=404, reason=f"Worker configuration for model {worker_id} not found")
        return

    logger.info(f"Websocket connection with worker_id: {worker_id}")

    hostname = socket.gethostname()

    # Get the IP address associated with the hostname
    ip_address = get_ip()

    request = WorkerStreamManageRequest(action="start", worker_id=worker_id,
                                        intype=WorkerStreamType.bytes, 
                                        outtype=WorkerStreamType.json,
                                        ip_address=ip_address,
                                        hostname=hostname,
                                        port=find_free_port(),
                                        timeout=timeout)
    
    job = await Jobs.create_job(payload=request.model_dump_json(),
                                type=JobsTypes.audio_generation,
                                worker_id=worker_id,
                                request_model_class=WorkerStreamManageRequest,
                                response_model_class=WorkerStreamManageResponse,
                                remote_method="stream",
                                storage=JobStorage.local,
                                keep_alive=SERVICE_KEEP_ALIVE)

    response: WorkerStreamManageResponse = await submit_job(job, 'False', JobPriority.high)
    await stream(websocket, request=request, response=response)