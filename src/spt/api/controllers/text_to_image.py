
from fastapi import Response, Header
from keys import API_KEY
from spt.models.jobs import JobsTypes, JobStatuses, JobResponse, JobStorage
from spt.models.image import TextToImageRequest, TextToImageResponse 
from spt.models.remotecalls import class_to_string
from spt.jobs import Job, Jobs
import base64
import requests
from spt.api.app import app
from spt.api.jobs import submit_job

async def text_to_image(request_data: TextToImageRequest, accept:Header, worker_id: str , api_key: str, async_key: str ,keep_alive_key: int, storage_key: str ,priority_key: str):
    job = await Jobs.create_job(payload=request_data.model_dump_json(), 
                        type=JobsTypes.image_generation, 
                        worker_id=worker_id,
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

async def text_to_image_job(job_id: str, accept:Header, api_key:str):

    job = Job(id=job_id, type=JobsTypes.image_generation, response_model_class=class_to_string(TextToImageResponse))
    status = await app.state.jobs[JobsTypes.image_generation].get_job_status(job)

    if status.status == JobStatuses.completed:
        result = await app.state.jobs[JobsTypes.image_generation].get_job_result(job)
        if accept == "image/png":
            image_data = None
            if result.artifacts[0].base64 is not None and result.artifacts[0].base64 != "":
                image_data = base64.b64decode(result.artifacts[0].base64)
            else:
                image_data = requests.get(result.artifacts[0].url).content

            return Response(content=image_data, media_type="image/png")
        return result
    return JobResponse(id=job.id, status=status.status, type=status.type, message=status.message)