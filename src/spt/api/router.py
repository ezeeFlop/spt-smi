
from fastapi import FastAPI, HTTPException, Depends, Security, Request, Response, Header, UploadFile, Form, File, WebSocket, WebSocketDisconnect
from fastapi.security.api_key import APIKeyHeader
from keys import API_KEY
from spt.models.jobs import JobsTypes, JobStatuses, JobResponse, JobPriority, JobStorage
from spt.models.image import TextToImageRequest, TextToImageResponse 
from spt.models.workers import WorkerConfigs
from spt.models.llm import ChatRequest, ChatResponse, EmbeddingsRequest, EmbeddingsResponse
from spt.models.audio import SpeechToTextRequest, SpeechToTextResponse, TextToSpeechRequest, TextToSpeechResponse
from spt.models.remotecalls import class_to_string, string_to_class, GPUsInfo, FunctionCallError
import time
from config import POLLING_TIMEOUT, SERVICE_KEEP_ALIVE
from typing import Type, Any, Optional, Union
from spt.api.app import app, logger, dispatcher
import spt.api.controllers as controllers
from spt.api.workers import validate_worker_exists, workers_configurations

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
    return await controllers.text_to_image(request_data=request_data,
                                            worker_id=worker_id, 
                                            accept=accept, 
                                            api_key=api_key,
                                            async_key = async_key, 
                                            keep_alive_key = keep_alive_key,
                                            storage_key = storage_key, 
                                            priority_key= priority_key)

@app.get("/v1/text-to-image/{job_id}", response_model=Union[JobResponse, TextToImageResponse], tags=["Text To Image Generation"])
async def text_to_image(job_id: str, request_data: TextToImageRequest, 
                        accept=Header(None), api_key: str = Depends(get_api_key)):
    return await controllers.text_to_image_job(job_id=job_id, accept=accept, api_key=api_key)

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
    return await controllers.text_to_text(request_data=request_data, 
                                          worker_id=worker_id, 
                                          api_key=api_key, 
                                          async_key=async_key, 
                                          keep_alive_key=keep_alive_key, 
                                          storage_key=storage_key, 
                                          priority_key=priority_key)

@app.get("/v1/text-to-text/{job_id}", response_model=Union[JobResponse, ChatResponse], tags=["Text To Text Generation"])
async def text_to_text(job_id: str, accept=Header(None), api_key: str = Depends(get_api_key)):
    return controllers.text_to_text_job(job_id=job_id, accept=accept, api_key=api_key)

@app.post("/v1/image-to-text", response_model=Union[ChatResponse, JobResponse], tags=["Image To Text Generation"])
async def image_to_text(request_data: ChatRequest,
                        worker_id: str = Depends(validate_worker_exists),
                       api_key: str = Depends(get_api_key),
                       async_key: str = Depends(get_async_key),
                       keep_alive_key: int = Depends(get_keep_alive_key),
                       storage_key: str = Depends(get_storage_key), priority_key: str = Depends(get_priority_key)):
    return await controllers.image_to_text(request_data=request_data, 
                                          worker_id=worker_id, 
                                          api_key=api_key, 
                                          async_key=async_key, 
                                          keep_alive_key=keep_alive_key, 
                                          storage_key=storage_key, 
                                          priority_key=priority_key)

@app.get("/v1/image-to-text/{job_id}", response_model=Union[JobResponse, ChatResponse], tags=["Image To Text Generation"])
async def image_to_text(job_id: str, accept=Header(None), api_key: str = Depends(get_api_key)):
    return controllers.image_to_text_job(job_id=job_id, accept=accept, api_key=api_key)

@app.post("/v1/text-to-embeddings", response_model=Union[JobResponse, EmbeddingsResponse], tags=["Text ToEmbeddings Generation"])
async def text_to_embeddings(request_data: EmbeddingsRequest, 
                             worker_id: str = Depends(validate_worker_exists),
                              api_key: str = Depends(get_api_key), 
                              async_key: str = Depends(get_async_key), 
                              keep_alive_key: int = Depends(get_keep_alive_key), 
                              storage_key: str = Depends(get_storage_key), priority_key: str = Depends(get_priority_key)):
    return await controllers.text_to_embeddings(request_data=request_data, 
                                          worker_id=worker_id, 
                                          api_key=api_key, 
                                          async_key=async_key, 
                                          keep_alive_key=keep_alive_key, 
                                          storage_key=storage_key, 
                                          priority_key=priority_key)

@app.get("/v1/text-to-embeddings/{job_id}", response_model=Union[JobResponse, EmbeddingsResponse], tags=["Text ToEmbeddings Generation"])
async def image_to_text(job_id: str, accept=Header(None), api_key: str = Depends(get_api_key)):
    return controllers.text_to_embeddings_job(job_id=job_id, accept=accept, api_key=api_key)

@app.websocket("/ws/v1/speech-to-text")
async def speech_to_text_stream(websocket: WebSocket, 
                           worker_id: str, timeout: int = 30):
    return await controllers.speech_to_text_stream(websocket, worker_id, timeout)

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
    return await controllers.speech_to_text(request_data=request_data, 
                                          worker_id=worker_id, 
                                          api_key=api_key, 
                                          async_key=async_key, 
                                          keep_alive_key=keep_alive_key, 
                                          storage_key=storage_key, 
                                          priority_key=priority_key)

@app.post("/v1/text-to-speech", response_model=Union[JobResponse, TextToSpeechResponse], tags=["Text To Speech Generation"])
async def text_to_speech(request_data: TextToSpeechRequest, accept=Header(None),
                         worker_id: str = Depends(validate_worker_exists),
                              api_key: str = Depends(get_api_key), 
                              async_key: str = Depends(get_async_key), 
                              keep_alive_key: int = Depends(get_keep_alive_key), 
                              storage_key: str = Depends(get_storage_key), priority_key: str = Depends(get_priority_key)):

    return await controllers.text_to_speech(request_data=request_data, 
                                          worker_id=worker_id, 
                                          accept=accept,
                                          api_key=api_key, 
                                          async_key=async_key, 
                                          keep_alive_key=keep_alive_key, 
                                          storage_key=storage_key, 
                                          priority_key=priority_key)
