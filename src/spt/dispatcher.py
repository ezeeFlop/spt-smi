from spt.jobs import Job
from spt.services.generic.client import GenericClient
from spt.models.jobs import JobStatuses, JobsTypes, JobResponse
from spt.models.remotecalls import MethodCallError, class_to_string, string_to_class, FunctionCallError
import logging
from config import IMAGE_GENERATION, IMAGE_PROCESSING, VIDEO_GENERATION, LLM_GENERATION, AUDIO_GENERATION
from spt.jobs import Jobs
from google.protobuf.json_format import MessageToJson
import traceback
import json
from pydantic import BaseModel, ValidationError
from typing import Type, Any, Union

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Dispatcher:
    def __init__(self) -> None:
        self.jobs = Jobs()
        logger.info("Initializing dispatcher")
        self.clients: dict[JobsTypes, GenericClient] = {}
        configs = {
            JobsTypes.image_generation: IMAGE_GENERATION,
            JobsTypes.llm_generation: LLM_GENERATION,
            JobsTypes.audio_generation: AUDIO_GENERATION,
            JobsTypes.video_generation: VIDEO_GENERATION,
            JobsTypes.image_processing: IMAGE_PROCESSING
        }

        for job_type in [JobsTypes.image_generation, JobsTypes.llm_generation, JobsTypes.audio_generation, JobsTypes.video_generation, JobsTypes.image_processing]:
            logger.info(f"Initializing client for job type {job_type}")
            try:
                self.clients[job_type] = GenericClient(configs[job_type])
            except Exception as e:
                logger.error(f"Failed to initialize client for job type {job_type}: {e} stack trace: {traceback.format_exc()}")

    async def call_remote_function(self, jobs_type: JobsTypes, remote_module: str, remote_function: str, payload: dict, response_model_class:Type[BaseModel]) -> Union[BaseModel|FunctionCallError]:
        logger.info(f"Calling remote function {remote_function} with payload: {payload}")
        try:
            response = self.clients[jobs_type].call_remote_function(
                remote_module, remote_function, payload, class_to_string(response_model_class))
            logger.info(f"Response: {response}")
            return response
        except Exception as e:
            logger.error(
                f"Failed to run remote function {remote_function}: {e} stack trace: {traceback.format_exc()}")
            return FunctionCallError(message=str(e))

    async def execute_job(self, job: Job) -> Union[BaseModel|MethodCallError]:
        logger.info(f"Executing job {job.id} {job.type}")
        try:
            job.payload = json.loads(job.payload)

            response = self.clients[job.type].process_data(job)
            
            payload = response.json_payload.decode('utf-8')

            if "status" in payload:
                logger.error(f"Job {job.id} failed: {payload}")
                error = MethodCallError(**json.loads(payload))
                if error.status == JobStatuses.failed:
                    return JobResponse(id=job.id, status=error.status, type=JobsTypes.unknown, message=error.message)
            else:
                response_model_class = string_to_class(job.response_model_class)
                result = response_model_class.model_validate_json(payload)
                return result
        except Exception as e:
            logger.error(f"Failed to execute job {job.id}: {e} stack trace: {traceback.format_exc()}")
            return MethodCallError(message=str(e))

    async def dispatch_job(self, job: Job):
        logger.info(f"Dispatching job {job.id} {job.type} with payload: {job.payload} keep alive {job.keep_alive} storage {job.storage}")
        try:
            await self.jobs.set_job_status(job, JobStatuses.in_progress)
            response = self.clients[job.type].process_data(job)
           
            payload = response.json_payload.decode('utf-8')

            if "status" in payload:
                logger.error(f"Job {job.id} failed: {payload}")
                error = MethodCallError(**json.loads(payload))
                if error.status == JobStatuses.failed:
                    await self.jobs.set_job_result(job, {
                        "payload": {},
                    })
                    await self.jobs.set_job_status(job, JobStatuses.failed, message=error.message)
            else:
                await self.jobs.set_job_result(job, {
                    "payload": payload,
                })
                await self.jobs.set_job_status(job, JobStatuses.completed)

            logger.info(f"Job {job.id} completed {payload}")
   
        except Exception as e:
            logger.error(f"Failed to dispatch job {job.id}: {e} stack trace: {traceback.format_exc()}")
            await self.jobs.set_job_result(job, {
                "payload": {},
            })
            await self.jobs.set_job_status(job, JobStatuses.failed, message=f"Failed to dispatch job: {str(e)}: {traceback.format_exc()}")
  