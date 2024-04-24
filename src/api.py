from fastapi import FastAPI, HTTPException, Depends, Security, Request, Response, Header
from fastapi.security.api_key import APIKeyHeader
from fastapi.responses import StreamingResponse

from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from keys import API_KEY
from spt.models.jobs import JobsTypes, JobStatuses, JobResponse
from spt.models.txt2img import TextToImageRequest, EnginesList, ArtifactsList 
from spt.models.llm import GenerateRequest, GenerateResponse, ChatRequest, ChatResponse, EmbeddingsRequest, EmbeddingsResponse
from spt.models.remotecalls import class_to_string, string_to_class
from spt.jobs import Job, Jobs
import time
from config import POLLING_TIMEOUT
from typing import Union
import asyncio
from rich.logging import RichHandler
from rich.console import Console
import logging
import base64
from pydantic import BaseModel
from typing import Type, Any

console = Console()

logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(
        console=console, rich_tracebacks=True, show_time=False)]
)

logger = logging.getLogger(__name__)

jobs = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global jobs
    if jobs is None:
        jobs_types = [JobsTypes.image_generation, JobsTypes.llm_generation,
                    JobsTypes.audio_generation, JobsTypes.video_generation]
        jobs = {job_type: Jobs(job_type) for job_type in jobs_types}
# jobs = Jobs(JobsTypes.image_generation)
    yield
    # Clean up the ML models and release the resources
    for job in jobs:
        job.stop()

app = FastAPI(
    lifespan=lifespan,
    title="spt-smi",
    version="0.0.1",
    description="""
                spt-smi API 🚀

                # Scalable Models Inferences

                You can request inferences from several models using this API.
                The models are hosted on the Sponge Theory infrastructure, so you can request
                inferences from several models at the same time using a hidden queue mecanism.

                This API can be deployed on a docker container for your own use.

                """,
    contact={
        "name": "Sponge Theory",
        "url": "https://sponge-theory.io",
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

async_key_header = APIKeyHeader(name="x-smi-async", auto_error=False)
async def get_async_key(async_key: str = Security(async_key_header)):
    return async_key

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
        f"Requête: {request.method} {request.url} - Temps de traitement: {process_time} secondes")
    return response


@app.post("/v1/generation/{engine_id}/text-to-image", response_model=Union[JobResponse, ArtifactsList], status_code=201, tags=["Image Generation"])
async def create_image(engine_id: str, request_data: TextToImageRequest, accept=Header(None), api_key: str = Depends(get_api_key), async_key: str = Depends(get_async_key)):
    logger.info(f"Text To Image with engine_id: {engine_id}")
    request_data.model = engine_id
    job = await add_job(request_data.model_dump_json(), JobsTypes.image_generation, engine_id, "generate_images", "spt.services.image_generation.service.DiffusionModels", TextToImageRequest, ArtifactsList)
    result = await get_job_result(job, async_key)

    if async_key:
        return result
    
    if accept == "image/png":
        # Suppose que le premier artifact contient l'image que tu veux retourner
        image_data = base64.b64decode(result.artifacts[0].base64)
        return Response(content=image_data, media_type="image/png")
    
    return result

@app.get("/v1/generation/text-to-image/{job_id}", response_model=Union[JobResponse, ArtifactsList])
async def create_image(job_id: str, request_data: TextToImageRequest, accept=Header(None), api_key: str = Depends(get_api_key)):
    logger.info(f"Text To Image job retreival: {job_id}")
    job = Job(id=job_id, type=JobsTypes.image_generation, response_model_class=class_to_string(ArtifactsList))
    status = await jobs[JobsTypes.image_generation].get_job_status(job)

    if status.status == JobStatuses.completed:
        result = await jobs[JobsTypes.image_generation].get_job_result(job)
        if accept == "image/png":
            # Suppose que le premier artifact contient l'image que tu veux retourner
            image_data = base64.b64decode(result.artifacts[0].base64)
            return Response(content=image_data, media_type="image/png")
        return result
    return JobResponse(id=job.id, status=status.status, type=status.type, message=status.message)

@app.get("/v1/engines/list", response_model=EnginesList)
async def list_engines(api_key: str = Depends(get_api_key)):
    logger.info(f"List engines")

    engines = EnginesList.get_engines()

    return EnginesList(engines=engines)

# New endpoints calling the OLLAMA API

@app.post("/v1/generate/completion", response_model=GenerateResponse, tags=["Text Generation"])
async def generate_completion(request_data: GenerateRequest, api_key: str = Depends(get_api_key)):
    pass

@app.post("/v1/generate/chat", response_model=Union[ChatResponse, JobResponse], tags=["Chat Generation"])
async def generate_chat(request_data: ChatRequest, api_key: str = Depends(get_api_key), async_key: str = Depends(get_async_key)):
    job = await add_job(request_data.model_dump_json(), JobsTypes.llm_generation, request_data.model, "generate_chat", "spt.services.llm_generation.service.LLMModels", ChatRequest, ChatResponse)
    return await get_job_result(job, async_key)


@app.get("/v1/generate/chat/{job_id}", response_model=Union[JobResponse, ChatResponse])
async def generate_chat(job_id: str, accept=Header(None), api_key: str = Depends(get_api_key)):
    logger.info(f"LLM Chat generation job retreival: {job_id}")
    job = Job(id=job_id, type=JobsTypes.llm_generation,
              response_model_class=class_to_string(ChatResponse))
    status = await jobs[JobsTypes.llm_generation].get_job_status(job)

    if status.status == JobStatuses.completed:
        result = await jobs[JobsTypes.llm_generation].get_job_result(job)
        return result
    return JobResponse(id=job.id, status=status.status, type=status.type, message=status.message)


@app.post("/v1/generate/embeddings", response_model=EmbeddingsResponse, tags=["Embeddings Generation"])
async def generate_embeddings(request_data: EmbeddingsRequest, api_key: str = Depends(get_api_key)):
    pass

async def add_job(payload: str, type: JobsTypes, model_id: str, remote_method: str, remote_class: str, request_model_class: Type[BaseModel], response_model_class: Type[BaseModel]) -> Job:
   job = Job(payload=payload,
             type=type, 
             model_id=model_id, 
             remote_method=remote_method, 
             remote_class=remote_class, 
             request_model_class=class_to_string(request_model_class), 
             response_model_class=class_to_string(response_model_class))
   await jobs[type].add_job(job)
   return job

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
