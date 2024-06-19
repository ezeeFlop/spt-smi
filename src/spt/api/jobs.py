

from fastapi import FastAPI, HTTPException, Depends, Security, Request, Response, Header, UploadFile, Form, File
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from keys import API_KEY
from spt.models.jobs import JobsTypes, JobStatuses, JobResponse, JobPriority
from spt.jobs import Job
from config import POLLING_TIMEOUT, SERVICE_KEEP_ALIVE
import asyncio
from typing import Type, Any
from spt.utils import find_free_port, get_ip
from spt.api.workers import validate_worker_exists
from spt.api.app import app, logger

async def submit_job(job: Job, async_key: str, priority_key: str) -> Type[BaseModel] | JobResponse:
    job_result = None
    if priority_key == JobPriority.high:
        job_result = await app.state.dispatcher.execute_job(job)
    else:
        await app.state.jobs[job.type].add_job(job)
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
        status = await app.state.jobs[JobsTypes.llm_generation].get_job_status(job)
        if status.status == JobStatuses.completed:
            result = await app.state.jobs[JobsTypes.llm_generation].get_job_result(job)
            logger.info(
                f"Job {job.id} completed with result: {result}")
            return result
        if status.status == JobStatuses.failed:
            return JobResponse(id=job.id, status=status.status, type=status.type, message=status.message)
    raise HTTPException(status_code=408, detail="Job timeout")
