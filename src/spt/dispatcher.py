from spt.jobs import Job
from spt.services.image_generation.client import TextToImageClient
from spt.models import JobsTypes, JobStatuses
import logging
from config import IMAGEGENERATION_SERVICE_PORT, IMAGEGENERATION_SERVICE_HOST
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
        self.textToImageClient = TextToImageClient(
            IMAGEGENERATION_SERVICE_HOST, IMAGEGENERATION_SERVICE_PORT)
    
    async def dispatch_job(self, job: Job):
        logger.info(f"Dispatching job {job.id} {job.type}")
        try:
            if job.type == JobsTypes.image_generation:
                await self.jobs.set_job_status(job, JobStatuses.in_progress)
                status = self.textToImageClient.status()
                logger.info(f"Job {job.id} in progress {status}")
                response = self.textToImageClient.generate_image(job)
                #logger.info(f"Job {job.id} in progress {response}")
                
                message = MessageToJson(response)
                #logger.info(f"Job {job.id} done : {message}")
                message = json.loads(message)
                await self.jobs.set_job_result(job, {
                    "images": message["images"],
                    "finishReason": message["finishReason"],
                })
                await self.jobs.set_job_status(job, JobStatuses.completed)
                #logger.info(f"Job {job.id} completed {response}")
            else:
                logger.error(f"Unknown job type {job.type}")
        except Exception as e:
            logger.error(f"Failed to dispatch job {job.id}: {e} stack trace: {traceback.format_exc()}")
            await self.jobs.set_job_status(job, JobStatuses.failed, message=str(e))
  