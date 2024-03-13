from fastapi import FastAPI, HTTPException, Depends, Security, Request
from fastapi.security.api_key import APIKeyHeader
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from keys import API_KEY
from spt.models import TextToImageRequest, EnginesList, ArtifactsList, JobResponse, JobsTypes, JobStatuses
from spt.jobs import Job, Jobs
import time
import json
from config import POLLING_TIMEOUT

from rich.logging import RichHandler
from rich.console import Console
import logging

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
        jobs = Jobs(JobsTypes.image_generation)
    yield
    # Clean up the ML models and release the resources
    jobs.stop()

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


@app.post("/v1/generation/{engine_id}/text-to-image", response_model=JobResponse|ArtifactsList, status_code=201, tags=["Image Generation"])
async def create_image(engine_id:str, request_data: TextToImageRequest, api_key: str = Depends(get_api_key), async_key: str = Depends(get_async_key)):
    logger.info(f"Text To Image with engine_id: {engine_id}")
    job = Job(payload=request_data.model_dump_json(), type=JobsTypes.image_generation, model_id=engine_id)
    await jobs.add_job(job)
    
    if async_key:
        return JobResponse(id=job.id, status=job.status, type=job.type, message=job.message)
    
    for _ in range(POLLING_TIMEOUT):
        time.sleep(1)
        status = jobs.get_job_status(job)
        if status.status == JobStatuses.completed:
            artifact = await jobs.get_job_result(job)
            return ArtifactsList(id=job.id, status=status.status, message=status.message, type=status.type, artifacts=[artifact])
        if status.status == JobStatuses.failed:
            return JobResponse(id=job.id, status=status.status, type=status.type, message=status.message)
    raise HTTPException(status_code=408, detail="Job timeout")


@app.get("/v1/generation/text-to-image/{job_id}", response_model=ArtifactsList|JobResponse)
async def create_image(job_id: str, request_data: TextToImageRequest, api_key: str = Depends(get_api_key)):
    logger.info(f"Text To Image job retreival: {job_id}")
    job = Job(id=job_id)
    status = jobs.get_job_status(job)
    if status.status == JobStatuses.completed:
        artifact = await jobs.get_job_result(job)
        return ArtifactsList(id=job.id, status=status.status, message=status.message, type=status.type, artifacts=[artifact])
    return JobResponse(id=job.id, status=status.status, type=status.type, message=status.message)

@app.get("/v1/engines/list", response_model=EnginesList)
async def list_engines(api_key: str = Depends(get_api_key)):
    logger.info(f"List engines")

    engines = EnginesList.get_engines()

    return EnginesList(engines=engines)
