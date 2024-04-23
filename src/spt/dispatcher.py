from spt.jobs import Job
from spt.services.generic.client import GenericClient
from spt.models.jobs import JobStatuses, JobsTypes
import logging
from config import IMAGEGENERATION_SERVICE_PORT, IMAGEGENERATION_SERVICE_HOST, LLM_SERVICE_HOST, LLM_SERVICE_PORT
from spt.jobs import Jobs
from google.protobuf.json_format import MessageToJson
import traceback
import json
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Dispatcher:
    def __init__(self) -> None:
        self.jobs = Jobs()
        logger.info("Initializing dispatcher")

        logger.info(f"Initializing text to image client with host {IMAGEGENERATION_SERVICE_HOST} and port {IMAGEGENERATION_SERVICE_PORT}")
        self.textToImageClient = GenericClient(
            IMAGEGENERATION_SERVICE_HOST, IMAGEGENERATION_SERVICE_PORT)
        
        logger.info(
            f"Initializing text to image client with host {IMAGEGENERATION_SERVICE_HOST} and port {IMAGEGENERATION_SERVICE_PORT}")
        self.llmClient = GenericClient(
            LLM_SERVICE_HOST, LLM_SERVICE_PORT)

    async def dispatch_job(self, job: Job):
        logger.info(f"Dispatching job {job.id} {job.type}")
        try:
            await self.jobs.set_job_status(job, JobStatuses.in_progress)
            response = None
            if job.type == JobsTypes.image_generation:
                response = self.textToImageClient.process_data(job)
            elif job.type == JobsTypes.llm_generation:
                response = self.llmClient.process_data(job)

            message = response.json_payload.decode('utf-8')

            await self.jobs.set_job_result(job, {
                "payload": message,
            })
            await self.jobs.set_job_status(job, JobStatuses.completed)
            logger.info(f"Job {job.id} completed {message}")
   
        except Exception as e:
            logger.error(f"Failed to dispatch job {job.id}: {e} stack trace: {traceback.format_exc()}")
            await self.jobs.set_job_status(job, JobStatuses.failed, message=str(e))
  