
from fastapi import Header, Response
from spt.models.jobs import JobsTypes, JobStatuses, JobResponse
from spt.models.llm import ChatResponse
from spt.models.audio import TextToSpeechRequest, TextToSpeechResponse
from spt.models.remotecalls import class_to_string
from spt.jobs import Job, Jobs
import base64
from typing import Union
import requests
from spt.api.app import app
from spt.api.jobs import submit_job

async def text_to_speech(request_data: TextToSpeechRequest, accept:Header, worker_id: str, api_key:str, storage_key: str, async_key: str, priority_key: str, keep_alive_key: int):
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

async def text_to_speech_job(job_id: str, accept:Header, api_key:str):
    job = Job(id=job_id, type=JobsTypes.audio_generation,
              response_model_class=class_to_string(ChatResponse))
    status = await app.state.jobs[JobsTypes.audio_generation].get_job_status(job)

    if status.status == JobStatuses.completed:
        result = await app.state.jobs[JobsTypes.audio_generation].get_job_result(job)
        return result
    return JobResponse(id=job.id, status=status.status, type=status.type, message=status.message)