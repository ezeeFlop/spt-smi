
from fastapi import Header
from spt.models.jobs import JobsTypes, JobStatuses, JobResponse
from spt.models.llm import ChatRequest, ChatResponse
from spt.models.remotecalls import class_to_string
from spt.jobs import Job, Jobs
from spt.api.app import  app
from spt.api.jobs import submit_job

async def text_to_text(request_data: ChatRequest, worker_id: str, api_key:str, storage_key: str, async_key: str, priority_key: str, keep_alive_key: int):
    job = await Jobs.create_job(payload=request_data.model_dump_json(), 
                        type=JobsTypes.llm_generation, 
                                worker_id=worker_id,
                        request_model_class=ChatRequest,
                        response_model_class=ChatResponse,
                        storage=storage_key,
                        keep_alive=keep_alive_key)

    return await submit_job(job, async_key, priority_key)

async def text_to_text_job(job_id: str, accept:Header, api_key:str):
    job = Job(id=job_id, type=JobsTypes.llm_generation,
              response_model_class=class_to_string(ChatResponse))
    status = await app.state.jobs[JobsTypes.llm_generation].get_job_status(job)

    if status.status == JobStatuses.completed:
        result = await app.state.jobs[JobsTypes.llm_generation].get_job_result(job)
        return result
    return JobResponse(id=job.id, status=status.status, type=status.type, message=status.message)