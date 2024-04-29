#import aioredis
from config import REDIS_HOST, SERVICE_KEEP_ALIVE
import redis
import uuid
import json 
from spt.queue import QueueMessageSender, Headers, Priority, QueueMessageReceiver, sync
from typing import List, Optional, Union
from spt.models.jobs import JobStatuses, JobsTypes, JobResponse
from spt.models.remotecalls import class_to_string, string_to_class
from spt.models.task import FunctionTask, MethodTask

from spt.scheduler import Scheduler
from rich.logging import RichHandler
from rich.console import Console
import logging
import concurrent.futures
import asyncio
import time
import msgpack
import threading
from pydantic import BaseModel
from typing import Type, Any
import traceback

console = Console()

logging.basicConfig(
    level="INFO",
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(
        console=console, rich_tracebacks=True, show_time=False)]
)

logger = logging.getLogger("Jobs")
dispatcher = None

class Job:
    def __init__(self, payload: Optional[str] = None, 
                 type: Optional[JobsTypes] = None, 
                 id: Optional[str] = None, 
                 model_id: Optional[str] = None, 
                 remote_class: Optional[str] = None, 
                 remote_method: Optional[str] = None,
                 request_model_class: Optional[str] = None, 
                 response_model_class: Optional[str] = None,
                 storage: Optional[str] = None,
                 keep_alive: Optional[int] = SERVICE_KEEP_ALIVE) -> None:
        self.id = uuid.uuid4().hex if id is None else id
        self.payload = payload
        self.status = JobStatuses.pending
        self.message = ""
        self.type = type
        self.model_id = model_id
        self.remote_class = remote_class
        self.remote_method = remote_method
        self.request_model_class = request_model_class
        self.response_model_class = response_model_class
        self.keep_alive = keep_alive
        self.storage = storage
        self.thread = None

class Jobs:
    def __init__(self, type: JobsTypes = JobsTypes.unknown):
        self.redis = self._redis_connect()
        self.publisher = None
        self.consumer = None
        self.routing_key = f"{type.value}"
        self.thread = None
        self.dispatcher = None

    def _redis_connect(self):
        return redis.Redis(
            host=REDIS_HOST, port=6379, db=0)

    def check_redis_connection(self):
        try: 
            response = self.redis.ping()
        except redis.ConnectionError:
            logger.error("Redis is not connected")
            self.redis = self._redis_connect()
    
    async def delete_job(self, job: Job):
        self.check_redis_connection()

        self.redis.delete(f"{job.id}:status")
        self.redis.delete(f"{job.id}:result")

    async def set_job_result(self, job: Job, result: Union[str, bytes]):
        self.check_redis_connection()

        toStore = json.dumps(result)
        toStore = msgpack.packb(toStore)
        self.redis.set(f"{job.id}:result", toStore)

    async def get_job_result(self, job: Job) -> Type[BaseModel] | JobResponse:
        self.check_redis_connection()

        result = self.redis.get(f"{job.id}:result")
        if result is None:
            return JobResponse(id=job.id, status=JobStatuses.unknown, message="Job not found", type=JobsTypes.unknown)
        
        result = msgpack.unpackb(result)
        result = json.loads(result)
        
        await self.delete_job(job)
        logger.info(f"Job {job.id} result: {result}")
        response_model_class = string_to_class(job.response_model_class)
        arg = response_model_class.model_validate_json(result["payload"])

        return arg

    async def set_job_status(self, job: Job, status: JobStatuses, message:str = ""):
        self.check_redis_connection()

        nextStatus = json.dumps({"status": status.value, "message": message, "type": job.type.value})
        self.redis.set(f"{job.id}:status", nextStatus)

    async def get_job_status(self, job: Job) -> JobResponse:
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
            await self.set_job_status(job, JobStatuses.failed, message=str(e))
            return
        
        await self.set_job_status(job, JobStatuses.queued, )

        logger.info(f"Job {job.id} added to queue {self.routing_key}")


    @classmethod
    async def create_job(cls, payload: str,
                        type: JobsTypes,
                        model_id: str,
                        remote_method: str,
                        remote_class: str,
                        request_model_class: Type[BaseModel],
                        response_model_class: Type[BaseModel],
                        keep_alive: int,
                        storage: str) -> Job:
        job = Job(payload=payload,
                    type=type,
                    model_id=model_id,
                    remote_method=remote_method,
                    remote_class=remote_class,
                    keep_alive=keep_alive,
                    storage=storage,
                    request_model_class=class_to_string(request_model_class),
                    response_model_class=class_to_string(response_model_class))
        return job

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
                            job_model_id=job.model_id, 
                            job_remote_class=job.remote_class, 
                            job_remote_method=job.remote_method, 
                            job_response_model_class=job.response_model_class, 
                            job_request_model_class=job.request_model_class,
                            job_storage=job.storage,
                            job_keep_alive=job.keep_alive)
        )

    @sync
    async def receive_job(self, channel, method, properties, body):
        global dispatcher

        body = self.consumer.decode_message(body=body)
        
        logger.info(f"---> Receive Job {channel}, {method} {properties}")
        logger.info(
            f"----> JOB ID {properties.headers['job_id']} TYPE {properties.headers['job_type']} MODEL ID {properties.headers['job_model_id']} CLASS {properties.headers['job_remote_class']} METHOD {properties.headers['job_remote_method']} Response Model Class {properties.headers['job_response_model_class']} Request Model Class {properties.headers['job_request_model_class']}")
        
        job = Job(json.loads(body), type=JobsTypes(properties.headers['job_type']), 
                  id=properties.headers['job_id'], 
                  model_id=properties.headers['job_model_id'], 
                  remote_class=properties.headers['job_remote_class'], 
                  remote_method=properties.headers['job_remote_method'],
                  response_model_class=properties.headers['job_response_model_class'],
                  request_model_class=properties.headers['job_request_model_class'],
                  keep_alive=properties.headers['job_keep_alive'],
                  storage=properties.headers['job_storage'])
        
        await self.set_job_status(job, JobStatuses.in_progress)
        
        if dispatcher is None:
            from spt.dispatcher import Dispatcher
            dispatcher = Dispatcher()

        await dispatcher.dispatch_job(job)

    def start_jobs_receiver_thread(self):
        """
        Démarre le thread de réception des jobs et initialise le membre thread.
        Cette méthode crée également une nouvelle boucle d'événements pour asyncio
        dans le nouveau thread, permettant d'exécuter des tâches asynchrones.
        """
        if self.thread is None or not self.thread.is_alive():
            # Création et démarrage du nouveau thread
            self.thread = threading.Thread(
                target=self._run_async_jobs_receiver, daemon=True)
            self.thread.start()
            logger.info(f"Thread démarré pour la queue {self.routing_key}.")

    def _run_async_jobs_receiver(self):
        """
        Crée une nouvelle boucle d'événements asyncio, la démarre, et exécute
        la réception des jobs dans cette boucle.
        """
        try:
            # Configuration d'une nouvelle boucle d'événements pour ce thread
            asyncio.set_event_loop(asyncio.new_event_loop())
            loop = asyncio.get_event_loop()
            # Démarrage de la réception des jobs dans la boucle d'événements
            loop.run_until_complete(self.start_jobs_receiver())
        except Exception as e:
            logger.error(
                f"Erreur lors de l'exécution de la réception des jobs: {e}")
            traceback.print_exc()
        finally:
            loop.close()

    def start_jobs_receiver_threadv1(self):
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
        if self.publisher is not None:
            self.publisher.close()
        self.redis.close()


def monitor_and_restart_jobs(jobs: List[Jobs], executor: concurrent.futures.ThreadPoolExecutor):
    """
    Surveille et relance les threads de réception des jobs si nécessaire.
    """
    while True:
        for job in jobs:
            # Vérifie si le thread est vivant.
            if not job.thread.is_alive():
                logger.error(
                    f"Le thread pour la queue {job.routing_key} est mort. Tentative de redémarrage...")
                # Relance le thread via ThreadPoolExecutor.
                future = executor.submit(job.start_jobs_receiver_thread)
                logger.info(
                    f"Thread pour la queue {job.routing_key} redémarré.")
        time.sleep(10)


if __name__ == "__main__":
    logger.info("Starting scheduler...")

    minioFilesPruner = MethodTask(method="prune_bucket", className="spt.storage.Storage", payload={
                                        "bucket": JobsTypes.image_generation, "day_to_keep": 7})
    scheduler = Scheduler()
    scheduler.add_job_method(minioFilesPruner, "* * * * *")

    logger.info("Starting jobs receiver...")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        jobs_types = [JobsTypes.image_generation, JobsTypes.llm_generation, JobsTypes.audio_generation, JobsTypes.video_generation]
        jobs = [Jobs(job_type) for job_type in jobs_types]
        futures = [executor.submit(job.start_jobs_receiver_thread)
                   for job in jobs]
        executor.submit(scheduler.start)
        logger.info("Started jobs receivers")
        monitor_thread = executor.submit(monitor_and_restart_jobs, jobs, executor)
        logger.info("Récepteurs de jobs démarrés et surveillance active.")


        # Attente (optionnelle) pour la démonstration; dans la pratique, tu pourrais vouloir
        # avoir une condition d'arrêt ou intégrer ceci dans ton système de manière différente.
        monitor_thread.result()