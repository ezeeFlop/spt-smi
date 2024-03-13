from spt.jobs import Job
from spt.services.image_generation.client import TextToImageClient
from spt.models import JobsTypes, JobStatuses
import logging
from config import IMAGEGENERATION_SERVICE_PORT
from spt.jobs import Jobs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Dispatcher:
    def __init__(self) -> None:
        self.jobs = Jobs()
        self.textToImageClient = TextToImageClient(
            "localhost", IMAGEGENERATION_SERVICE_PORT)
    
    async def dispatch_job(self, job: Job):
        logger.info(f"Dispatching job {job.id} {job.type}")
        try:
            if job.type == JobsTypes.image_generation:
                await self.jobs.set_job_status(job, JobStatuses.in_progress)
                status = self.textToImageClient.status()
                logger.info(f"Job {job.id} in progress {status}")
                response = self.textToImageClient.generate_image(job)
                logger.info(f"Job {job.id} in progress {response}")
                await self.jobs.set_job_result(job, {
                    "base64": response.base64,
                    "finishReason": response.finishReason,
                    "seed": response.seed
                })
                await self.jobs.set_job_status(job, JobStatuses.completed)
                logger.info(f"Job {job.id} completed {response}")
            else:
                logger.error(f"Unknown job type {job.type}")
        except Exception as e:
            logger.error(f"Failed to dispatch job {job.id}: {e}")
            await self.jobs.set_job_status(job, JobStatuses.failed, message=str(e))
  