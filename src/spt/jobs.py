#import aioredis
from config import REDIS_HOST
import redis
import uuid
import json 
from spt.queue import QueueMessageSender, Headers, Priority, QueueMessageReceiver, sync
from typing import List, Optional, Union
from spt.models import JobStatuses, JobsTypes, TextToImageRequest, JobResponse, Artifact
from rich.logging import RichHandler
from rich.console import Console
import logging
import concurrent.futures
import asyncio
import time


console = Console()

logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(
        console=console, rich_tracebacks=True, show_time=False)]
)

logger = logging.getLogger(__name__)


class Job:
    def __init__(self, payload: Optional[TextToImageRequest] = None , type: Optional[JobsTypes] = None, id: Optional[str] = None, model_id: Optional[str] = None) -> None:
        self.id = uuid.uuid4().hex if id is None else id
        self.payload = payload
        self.status = JobStatuses.pending
        self.message = ""
        self.type = type
        self.model_id = model_id

class Jobs:
    def __init__(self, type: JobsTypes = JobsTypes.unknown):
        self.redis = self._redis_connect()
        self.publisher = None
        self.consumer = None
        self.routing_key = f"{type.value}"

    def _redis_connect(self):
        return redis.Redis(
            host=REDIS_HOST, port=6379, db=0)

    def check_redis_connection(self):
        try: 
            response = self.redis.ping()
        except redis.ConnectionError:
            logger.error("Redis is not connected")
            self.redis = self._redis_connect()
    
    async def set_job_result(self, job: Job, result: Union[str, bytes]):
        self.check_redis_connection()

        toStore = json.dumps(result)
        self.redis.set(f"{job.id}:result", toStore)

    async def get_job_result(self, job: Job) -> Artifact|JobResponse:
        self.check_redis_connection()

        result = self.redis.get(f"{job.id}:result")
        if result is None:
            return JobResponse(id=job.id, status=JobStatuses.unknown, message="Job not found", type=JobsTypes.unknown)
        result = json.loads(result.decode('utf-8'))
        logger.info(f"Job {job.id} result: {result}")
        return Artifact(base64=result['base64'], finishReason=result['finishReason'], seed=result['seed'])

    async def set_job_status(self, job: Job, status: JobStatuses, message:str = ""):
        self.check_redis_connection()

        nextStatus = json.dumps({"status": status.value, "message": message, "type": job.type.value})
        self.redis.set(f"{job.id}:status", nextStatus)

    def get_job_status(self, job: Job) -> JobResponse:
        self.check_redis_connection()

        status = self.redis.get(f"{job.id}:status")
        if status is None:
            return JobResponse(id=job.id, status=JobStatuses.unknown, message="Job not found", type=JobsTypes.unknown)
        status = json.loads(status.decode('utf-8'))
        logger.info(f"Job {job.id} status: {status}")
        return JobResponse(id=job.id, status=JobStatuses(status['status']), message=status['message'], type=status['type'])

    async def add_job(self, job: Job):
        await self.set_job_status(job, JobStatuses.pending)
        
        try:
            self._send_job(job)
        except Exception as e:
            logger.error(f"Failed to add job {job.id} to queue: {e}")
            self.set_job_status(job, JobStatuses.failed, message=str(e))
            return
        
        await self.set_job_status(job, JobStatuses.queued, )

        logger.info(f"Job {job.id} added to queue {self.routing_key}")

    def _send_job(self, job: Job):
        if self.publisher is None:
            self.publisher = QueueMessageSender()

        self.publisher.declare_exchange(exchange_name="spt", exchange_type="direct")
        self.publisher.declare_queue(queue_name="smi-requests")
        self.publisher.send_message(
            exchange_name="spt",
            routing_key=self.routing_key,
            body=job.payload,
            priority=Priority.NORMAL,
            headers=Headers(job_id=job.id, job_type=job.type,
                            job_model_id=job.model_id)
        )

    @sync
    async def receive_job(self, channel, method, properties, body):
        body = self.consumer.decode_message(body=body)
        
        logger.info(f"{channel}, {method} {properties} Body received {body}")
        logger.info(
            f"----> JOB ID {properties.headers['job_id']} TYPE {properties.headers['job_type']} MODEL ID {properties.headers['job_model_id']}")
        
        job = Job(json.loads(body), type=JobsTypes(properties.headers['job_type']), id=properties.headers['job_id'], model_id=properties.headers['job_model_id'])
        
        await self.set_job_status(job, JobStatuses.in_progress)
        
        from spt.dispatcher import Dispatcher

        await Dispatcher().dispatch_job(job)


    def start_jobs_receiver_thread(self):
        logger.info(f"Starting jobs  thread for queue {self.routing_key}")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            asyncio.ensure_future(self.start_jobs_receiver())
            loop.run_forever()
        except Exception as e:
            logger.error(
                f"Failed to start jobs receiver for queue {self.routing_key}: {e}")
        finally:
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()
            self.stop()

    def start_jobs_receiver(self):
        logger.info(f"Starting jobs receiver for queue {self.routing_key}")
        self.consumer = QueueMessageReceiver()
        self.consumer.declare_queue(queue_name="smi-requests")
        self.consumer.declare_exchange(exchange_name="spt")
        self.consumer.bind_queue(
            exchange_name="spt", queue_name="smi-requests", routing_key=self.routing_key
        )
        self.consumer.consume_messages(
            queue="smi-requests", callback=self.receive_job)
    
    def stop(self):
        self.publisher.close()
        self.redis.close()


if __name__ == "__main__":
    logger.info("Starting jobs receivers...")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        jobs_types = [JobsTypes.image_generation, JobsTypes.llm_generation, JobsTypes.audio_generation, JobsTypes.video_generation]
        jobs = [Jobs(job_type) for job_type in jobs_types]
        futures = [executor.submit(job.start_jobs_receiver_thread)
                   for job in jobs]
        logger.info("Started jobs receivers")
        while True:
            time.sleep(10)
            logger.info("Checking jobs receivers...")
      