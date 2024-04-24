from spt.jobs import Job
from spt.services.generic.client import GenericClient
from spt.models.jobs import JobStatuses, JobsTypes
from spt.models.remotecalls import MethodCallError, class_to_string
import logging
from config import IMAGE_GENERATION, IMAGE_PROCESSING, VIDEO_GENERATION, LLM_GENERATION, AUDIO_GENERATION
from spt.jobs import Jobs
from google.protobuf.json_format import MessageToJson
import traceback
import json
from pydantic import BaseModel, ValidationError, validator
from typing import Type, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Dispatcher:
    def __init__(self) -> None:
        self.jobs = Jobs()
        logger.info("Initializing dispatcher")
        self.clients = {}
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

    async def call_remote_function(self, jobs_type: JobsTypes, remote_module: str, remote_function: str, payload: dict, response_model_class:Type[BaseModel]) -> BaseModel:
        logger.info(f"Calling remote function {remote_function} with payload: {payload}")
        try:
            response = self.clients[jobs_type].call_remote_function(
                remote_module, remote_function, payload, class_to_string(response_model_class))
            logger.info(f"Response: {response}")
            return response
        except Exception as e:
            logger.error(
                f"Failed to run remote function {remote_function}: {e} stack trace: {traceback.format_exc()}")

    async def dispatch_job(self, job: Job):
        logger.info(f"Dispatching job {job.id} {job.type}")
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
  