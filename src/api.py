from fastapi import FastAPI, HTTPException, Depends, Security, Request, Response, Header, UploadFile, Form, File
from fastapi.security.api_key import APIKeyHeader
from fastapi.responses import StreamingResponse
from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from keys import API_KEY
from spt.models.jobs import JobsTypes, JobStatuses, JobResponse, JobPriority, JobStorage
from spt.models.image import TextToImageRequest, TextToImageResponse 
from spt.models.workers import WorkerConfigs
from spt.models.llm import ChatRequest, ChatResponse, EmbeddingsRequest, EmbeddingsResponse
from spt.models.audio import SpeechToTextRequest, SpeechToTextResponse, TextToSpeechRequest, TextToSpeechResponse
from spt.models.remotecalls import class_to_string, string_to_class, GPUsInfo, FunctionCallError, MethodCallError
from spt.models.workers import WorkerStreamManageRequest, WorkerStreamManageResponse, WorkerStreamType
from spt.jobs import Job, Jobs
import time
from config import POLLING_TIMEOUT, SERVICE_KEEP_ALIVE
import asyncio
from rich.logging import RichHandler
from rich.console import Console
import logging
import base64
from typing import Type, Any, Optional, Union
import requests
import zmq
import traceback
from zmq.asyncio import Context, Poller
import socket
from spt.utils import find_free_port

console = Console()

logging.basicConfig(
    level="INFO",
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(
        console=console, rich_tracebacks=True, show_time=False)]
)

logger = logging.getLogger("API")

jobs = None
dispatcher = None
workers_configurations = WorkerConfigs.get_configs()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global jobs
    global dispatcher
    if jobs is None:
        jobs_types = [JobsTypes.image_generation, JobsTypes.llm_generation,
                    JobsTypes.audio_generation, JobsTypes.video_generation]
        jobs = {job_type: Jobs(job_type) for job_type in jobs_types}
    if dispatcher is None:
        from spt.dispatcher import Dispatcher
        dispatcher = Dispatcher()
    yield
    # Clean up the ML models and release the resources
    for job in jobs:
        job.stop()

app = FastAPI(
    lifespan=lifespan,
    title="spt-smi",
    version="0.0.5",
    description="""
                spt-smi API 🚀

                # Scalable Models Inferences

                You can request inferences from several models using this API.
                The models are hosted on the Sponge Theory infrastructure, so you can request
                inferences from several models at the same time using a hidden queue mecanism.

                This API can be deployed on a docker container for your own use.
                It does include the following stacks : 
                    - RabbitMQ for message broker
                    - Redis for caching
                    - FastAPI for the API
                    - Minio for the storage
                It support dymanic scaling and load balancing, GRPC distrubuted remote services with workers
                for each IA models.
                Websocket streaming is also supported (ie. STT)
                """,
    contact={
        "name": "Sponge Theory",
        "url": "https://sponge-theory.ai",
        "email": "contact@sponge-theory.io",
    })

"""
This initializes a FastAPI instance with title, version, 
description, and contact info. It also configures logging.

FastAPI provides a clean way to write REST APIs in Python.
"""
api_key_header = APIKeyHeader(name="x-smi-key", auto_error=False)

async def get_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="API key invalid")
    return api_key

"""
    if x-smi-async is set, it tells to the underlying service to use async mode.
"""
async_key_header = APIKeyHeader(name="x-smi-async", auto_error=False)
async def get_async_key(async_key: str = Security(async_key_header)):
    return async_key

"""
    if x-smi-keep-alive is set, it tells to the underlying service to keep the models in memory for x seconds.
"""
keep_alive_key_header = APIKeyHeader(name="x-smi-keep-alive", auto_error=False)
async def get_keep_alive_key(keep_alive_key: int = Security(keep_alive_key_header)):
    if keep_alive_key is None or keep_alive_key == 0:
        return SERVICE_KEEP_ALIVE
    return int(keep_alive_key)

"""
    if x-smi-storage is "S3", the underlying service is allowed to access the storage on Minio instance
"""
storage_key_header = APIKeyHeader(name="x-smi-storage", auto_error=False)
async def get_storage_key(storage_key: str = Security(storage_key_header)):
    if storage_key is not None and storage_key not in [JobStorage.local, JobStorage.s3]:
        raise HTTPException(status_code=401, detail="Storage key invalid value")
    if storage_key is None:
        storage_key = JobStorage.local
    return storage_key

"""
    if x-smi-priority is "high", the underlying service is allowed to use high priority jobs.
    it bypass the hidden queue and directly execute the job.
"""
priority_key_header = APIKeyHeader(name="x-smi-priority", auto_error=False)
async def get_priority_key(priority_key: str = Security(priority_key_header)):
    if priority_key is None:
        priority_key = JobPriority.low
    if priority_key not in [JobPriority.low, JobPriority.normal, JobPriority.high]:
        raise HTTPException(status_code=401, detail="Priority key invalid value")
    return priority_key

async def validate_worker_exists(request: Request):
    data = await request.json()
    worker_id = data.get("worker_id")
    if worker_id not in workers_configurations.workers_configs:
        raise HTTPException(
            status_code=404, detail=f"Worker configuration for model {worker_id} not found")
    return worker_id

"""Logs requests to the logger.

This middleware logs each request, the response time, 
and the request details to the logger.
"""
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(
        f"Request: {request.method} {request.url} - process time: {process_time} seconds")
    return response

@app.post("/v1/text-to-image", response_model=Union[JobResponse, TextToImageResponse ], status_code=201, tags=["Text To Image Generation"])
async def text_to_image(request_data: TextToImageRequest, 
                       accept=Header(None), 
                        worker_id: str = Depends(validate_worker_exists),
                       api_key: str = Depends(get_api_key), 
                       async_key: str = Depends(get_async_key), 
                       keep_alive_key: int = Depends(get_keep_alive_key), 
                       storage_key: str = Depends(get_storage_key),
                       priority_key: str = Depends(get_priority_key)):
    
    logger.info(f"Text To Image with worker_id: {worker_id}")

    job = await Jobs.create_job(payload=request_data.model_dump_json(), 
                        type=JobsTypes.image_generation, 
                                worker_id=worker_id,
                        #request_model_class=TextToImageRequest, 
                        #response_model_class=TextToImageResponse,
                        storage=storage_key,
                        keep_alive=keep_alive_key)
    
    result = await submit_job(job, async_key, priority_key)

    if async_key:
        return result
    
    if accept == "image/png" and isinstance(result, TextToImageResponse):
        image_data = None
        if storage_key == JobStorage.local:
            image_data = base64.b64decode(result.artifacts[0].base64)
        elif storage_key == JobStorage.s3:
            image_data = requests.get(result.artifacts[0].url).content
        return Response(content=image_data, media_type="image/png")
    
    return result


@app.get("/v1/text-to-image/{job_id}", response_model=Union[JobResponse, TextToImageResponse], tags=["Text To Image Generation"])
async def text_to_image(job_id: str, request_data: TextToImageRequest, 
                        accept=Header(None), api_key: str = Depends(get_api_key)):
    logger.info(f"Text To Image job retrieval: {job_id}")
    job = Job(id=job_id, type=JobsTypes.image_generation, response_model_class=class_to_string(TextToImageResponse))
    status = await jobs[JobsTypes.image_generation].get_job_status(job)

    if status.status == JobStatuses.completed:
        result = await jobs[JobsTypes.image_generation].get_job_result(job)
        if accept == "image/png":
            image_data = None
            if result.artifacts[0].base64 is not None and result.artifacts[0].base64 != "":
                image_data = base64.b64decode(result.artifacts[0].base64)
            else:
                image_data = requests.get(result.artifacts[0].url).content

            return Response(content=image_data, media_type="image/png")
        return result
    return JobResponse(id=job.id, status=status.status, type=status.type, message=status.message)


@app.get("/v1/workers/list", response_model=WorkerConfigs)
async def list_worker_configurations(api_key: str = Depends(get_api_key)):
    logger.info(f"List available workers configurations")
    return workers_configurations

@app.get("/v1/gpu/info", response_model=Union[GPUsInfo|FunctionCallError])
async def gpu_infos(api_key: str = Depends(get_api_key)):
    logger.info(f"Get GPUs Infos")
    return await dispatcher.call_remote_function(JobsTypes.llm_generation, "spt.services.gpu","gpu_infos", {}, GPUsInfo)

# New endpoints calling the OLLAMA API

@app.post("/v1/text-to-text", response_model=Union[ChatResponse, JobResponse], tags=["Text To Text Generation"])
async def text_to_text(request_data: ChatRequest,
                       worker_id: str = Depends(validate_worker_exists),
                        api_key: str = Depends(get_api_key), 
                        async_key: str = Depends(get_async_key), 
                        keep_alive_key: int = Depends(get_keep_alive_key), 
                        storage_key: str = Depends(get_storage_key), priority_key: str = Depends(get_priority_key)):
    
    job = await Jobs.create_job(payload=request_data.model_dump_json(), 
                        type=JobsTypes.llm_generation, 
                                worker_id=worker_id,
                        request_model_class=ChatRequest,
                        response_model_class=ChatResponse,
                        storage=storage_key,
                        keep_alive=keep_alive_key)

    return await submit_job(job, async_key, priority_key)


@app.post("/v1/image-to-text", response_model=Union[ChatResponse, JobResponse], tags=["Image To Text Generation"])
async def image_to_text(request_data: ChatRequest,
                        worker_id: str = Depends(validate_worker_exists),
                       api_key: str = Depends(get_api_key),
                       async_key: str = Depends(get_async_key),
                       keep_alive_key: int = Depends(get_keep_alive_key),
                       storage_key: str = Depends(get_storage_key), priority_key: str = Depends(get_priority_key)):

    job = await Jobs.create_job(payload=request_data.model_dump_json(),
                                type=JobsTypes.llm_generation,
                                worker_id=worker_id,
                                request_model_class=ChatRequest,
                                response_model_class=ChatResponse,
                                storage=storage_key,
                                keep_alive=keep_alive_key)

    return await submit_job(job, async_key, priority_key)


@app.get("/v1/text-to-text/{job_id}", response_model=Union[JobResponse, ChatResponse], tags=["Text To Text Generation"])
async def text_to_text(job_id: str, accept=Header(None), api_key: str = Depends(get_api_key)):
    logger.info(f"LLM Chat generation job retrieval: {job_id}")
    job = Job(id=job_id, type=JobsTypes.llm_generation,
              response_model_class=class_to_string(ChatResponse))
    status = await jobs[JobsTypes.llm_generation].get_job_status(job)

    if status.status == JobStatuses.completed:
        result = await jobs[JobsTypes.llm_generation].get_job_result(job)
        return result
    return JobResponse(id=job.id, status=status.status, type=status.type, message=status.message)


@app.post("/v1/text-to-embeddings", response_model=Union[JobResponse, EmbeddingsResponse], tags=["Text ToEmbeddings Generation"])
async def text_to_embeddings(request_data: EmbeddingsRequest, 
                             worker_id: str = Depends(validate_worker_exists),
                              api_key: str = Depends(get_api_key), 
                              async_key: str = Depends(get_async_key), 
                              keep_alive_key: int = Depends(get_keep_alive_key), 
                              storage_key: str = Depends(get_storage_key), priority_key: str = Depends(get_priority_key)):
    job = await Jobs.create_job(payload=request_data.model_dump_json(), type=JobsTypes.llm_generation, 
                                worker_id=worker_id,
                        request_model_class=EmbeddingsRequest, 
                        response_model_class=EmbeddingsResponse,
                        storage=storage_key,
                        keep_alive=keep_alive_key)

    return await submit_job(job, async_key, priority_key)

@app.websocket("/ws/v1/speech-to-text")
async def speech_to_text_stream(websocket: WebSocket, 
                           worker_id: str, timeout: int = 30):
  
    api_key = websocket.headers.get("x-smi-key")
    logger.info(
        f"WebSocket connection attempt with API key: {api_key} and worker_id: {worker_id}")

    if api_key != API_KEY:
        await websocket.close(code=1008, reason="API key invalid")
        return

    if worker_id not in workers_configurations.workers_configs:
        await websocket.close(code=404, reason=f"Worker configuration for model {worker_id} not found")
        return

    logger.info(f"Websocket connection with worker_id: {worker_id}")

    hostname = socket.gethostname()

    # Get the IP address associated with the hostname
    ip_address = socket.gethostbyname(hostname)

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


@app.post("/v1/speech-to-text", response_model=Union[JobResponse, SpeechToTextResponse], tags=["Speech To Text Generation"])
async def speech_to_text(
    file: UploadFile = File(...),
    worker_id: str = Form(...),
    language: Optional[str] = Form(None),
    temperature: Optional[float] = Form(0.0),
    prompt: Optional[str] = Form(None),
    keep_alive: Optional[str] = Form(0),
    api_key: str = Depends(get_api_key),
    async_key: str = Depends(get_async_key),
    keep_alive_key: int = Depends(get_keep_alive_key),
    storage_key: str = Depends(get_storage_key),
    priority_key: str = Depends(get_priority_key)
):
    logger.info(f"Speech to text generation: {worker_id}")
    file_content = await file.read()
    request_data = SpeechToTextRequest(
        worker_id=worker_id,
        file=file_content,
        language=language,
        temperature=temperature,
        prompt=prompt
    )
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

@app.post("/v1/text-to-speech", response_model=Union[JobResponse, TextToSpeechResponse], tags=["Text To Speech Generation"])
async def text_to_speech(request_data: TextToSpeechRequest, accept=Header(None),
                         worker_id: str = Depends(validate_worker_exists),
                              api_key: str = Depends(get_api_key), 
                              async_key: str = Depends(get_async_key), 
                              keep_alive_key: int = Depends(get_keep_alive_key), 
                              storage_key: str = Depends(get_storage_key), priority_key: str = Depends(get_priority_key)):

    job = await Jobs.create_job(
        payload=request_data.model_dump_json(),  # Assuming you serialize to JSON if needed
        type=JobsTypes.audio_generation,
        worker_id=worker_id,
        request_model_class=TextToSpeechRequest,
        response_model_class=TextToSpeechResponse,
        storage=storage_key,
        keep_alive=keep_alive_key
    )
    result: Union[JobResponse, TextToSpeechResponse] = await submit_job(job, async_key, priority_key)

    if async_key:
        return result

    if accept == "audio/wav":
        data = None
        if result.base64 is not None and result.base64 != "":
            data = base64.b64decode(result.base64)
        else:
            data = requests.get(result.url).content
        return Response(content=data, media_type="audio/wav")

    return result

async def submit_job(job: Job, async_key: str, priority_key: str) -> Type[BaseModel] | JobResponse:
    job_result = None
    if priority_key == JobPriority.high:
        job_result = await dispatcher.execute_job(job)
    else:
        await jobs[job.type].add_job(job)
        job_result = await get_job_result(job, async_key)

    if isinstance(job_result, JobResponse) and job_result.status == JobStatuses.failed:
        raise HTTPException(status_code=503, detail=job_result.message)
    
    return job_result

async def get_job_result(job: Job, async_key: str) -> Type[BaseModel] | JobResponse:
    if async_key:
        logger.info(f"Waiting for async job {job.id} {job.status} {job.type} {job.message} to complete")
        return JobResponse(id=job.id, status=job.status, type=job.type, message=job.message)

    for _ in range(POLLING_TIMEOUT):
        await asyncio.sleep(1)  # Utilise une pause asynchrone
        status = await jobs[JobsTypes.llm_generation].get_job_status(job)
        if status.status == JobStatuses.completed:
            result = await jobs[JobsTypes.llm_generation].get_job_result(job)
            logger.info(
                f"Job {job.id} completed with result: {result}")
            return result
        if status.status == JobStatuses.failed:
            return JobResponse(id=job.id, status=status.status, type=status.type, message=status.message)
    raise HTTPException(status_code=408, detail="Job timeout")


async def stream(websocket: WebSocket, request: WorkerStreamManageRequest, response: WorkerStreamManageResponse):
    await websocket.accept()

    logger.info(f"Websocket {response}")

    context = Context()
    sender = context.socket(zmq.PUSH)
    sender.bind(f"tcp://*:{request.port}")

    receiver = context.socket(zmq.PULL)
    receiver.connect(f"tcp://{response.ip_address}:{response.port}")

    poller = Poller()
    poller.register(receiver, zmq.POLLIN)

    input_funcs = {
        WorkerStreamType.text: receiver.recv_string,
        WorkerStreamType.bytes: receiver.recv,
        WorkerStreamType.json: receiver.recv_json
    }
    output_funcs = {
        WorkerStreamType.text: sender.send_string,
        WorkerStreamType.bytes: sender.send,
        WorkerStreamType.json: sender.send_json
    }

    output_ws_funcs = {
        WorkerStreamType.text: websocket.send_text,
        WorkerStreamType.bytes: websocket.send_bytes,
        WorkerStreamType.json: websocket.send_json
    }
    input_ws_funcs = {
        WorkerStreamType.text: websocket.receive_text,
        WorkerStreamType.bytes: websocket.receive_bytes,
        WorkerStreamType.json: websocket.receive_json
    }

    input_func = input_funcs[request.outtype]
    output_func = output_funcs[request.intype]

    input_ws_func = input_ws_funcs[request.intype]
    output_ws_func = output_ws_funcs[request.outtype]

    async def receive_from_ws():
        try:
            while True:
                try:
                    data = await asyncio.wait_for(input_ws_func(), timeout=request.timeout)
                    logger.debug(f"Received data from WebSocket: {data}")
                    await output_func(data)
                except asyncio.TimeoutError:
                    logger.info("WebSocket timed out due to inactivity")
                    break
        except WebSocketDisconnect as e:
            logger.info(f"WebSocket disconnected: {e.code} - {e.reason}")
        except asyncio.CancelledError:
            logger.info("Task receive_from_ws cancelled")
        except Exception as e:
            logger.error(
                f"Error in receive_from_ws: {e} {traceback.format_exc()}")
        finally:
            try:
                receiver.close()
                sender.close()
                context.term()
                if websocket.state == WebSocketState.CONNECTED:
                    await websocket.close()
            except Exception as e:
                logger.error(
                    f"Error during cleanup in receive_from_ws: {e} {traceback.format_exc()}")

    async def send_to_ws():
        try:
            while True:
                events = await poller.poll(timeout=1000)
                if receiver in dict(events):
                    message = await input_func()
                    logger.debug(f"Received message from ZeroMQ: {message}")
                    await output_ws_func(message)
        except WebSocketDisconnect as e:
            logger.info(f"WebSocket disconnected: {e.code} - {e.reason}")
        except asyncio.CancelledError:
            logger.info("Task send_to_ws cancelled")
        except Exception as e:
            logger.error(f"Error in send_to_ws: {e} {traceback.format_exc()}")
        finally:
            try:
                receiver.close()
                sender.close()
                context.term()
                if websocket.state == WebSocketState.CONNECTED:
                    await websocket.close()
            except Exception as e:
                logger.error(
                    f"Error during cleanup in send_to_ws: {e} {traceback.format_exc()}")

    receive_task = asyncio.create_task(
        receive_from_ws(), name="receive_from_ws")
    send_task = asyncio.create_task(send_to_ws(), name="send_to_ws")

    try:
        await asyncio.gather(send_task, receive_task)
    except asyncio.CancelledError:
        logger.info("Main gather task cancelled")
    finally:
        receive_task.cancel()
        send_task.cancel()
        await asyncio.gather(receive_task, send_task, return_exceptions=True)
