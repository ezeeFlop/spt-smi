from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.jobstores.redis import RedisJobStore
import apscheduler.events
from apscheduler.triggers.cron import CronTrigger
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR
from config import REDIS_HOST, REDIS_PORT
from spt.models.task import MethodTask, FunctionTask
from spt.models.remotecalls import class_to_string, string_to_class, string_to_module
from rich.logging import RichHandler
from rich.console import Console
from typing import Optional, List
import logging
console = Console()

logging.basicConfig(
    level="INFO",
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(
        console=console, rich_tracebacks=True, show_time=False)]
)

logger = logging.getLogger('Scheduler')

class Scheduler:

    def __init__(self) -> None:
        jobstores = {
            'default': RedisJobStore(
                jobs_key='apscheduler.jobs',
                run_times_key='apscheduler.run_times', 
                host=REDIS_HOST, port=REDIS_PORT
            )
        }
        self.scheduler = BackgroundScheduler(jobstores=jobstores)
    
    def start(self):
        logger.info("Starting scheduler")
        self.scheduler.add_listener(
            self.job_events_listener, EVENT_JOB_EXECUTED | EVENT_JOB_ERROR)
        self.scheduler.start()

    def job_events_listener(self, event: apscheduler.events.JobExecutionEvent) -> None:
        """
        Listener function for APScheduler events.
        Logs exceptions raised by jobs and events themselves.

        Parameters
        ----------
        event : apscheduler.events.JobEvent
            APScheduler event object.

        Returns
        -------
        None
        """
        if event.exception:
            logger.error(event.exception)
        else:
            logger.info(event)

    def shutdown(self):
        self.scheduler.shutdown(wait=False)

    def job_id_exists(self, id: str) -> bool:
        return self.scheduler.get_job(id) is not None
    
    def add_job_local_method(self, method, cron: str, id: str = None):
        if self.job_id_exists(id):
            self.scheduler.remove_job(id)
        logger.info(f"Adding job cron: {cron}")
        self.scheduler.add_job(method,
                               trigger=CronTrigger.from_crontab(cron), id=id)

    def add_job_method(self, task: MethodTask, cron: str, id: str = None):
        if self.job_id_exists(id):
            self.scheduler.remove_job(id)
        logger.info(f"Adding job: {task.className}.{task.method} cron: {cron}")
        class_ = string_to_class(task.className)
        instance = class_()
        method = getattr(instance, task.method)
        self.scheduler.add_job(method, args=task.payload,
                               trigger=CronTrigger.from_crontab(cron), id=id)

    def add_job_function(self, task: FunctionTask, cron: str, id: str = None):
        if self.job_id_exists(id):
            self.scheduler.remove_job(id)
            
        logger.info(f"Adding job: {task.module}.{task.function} cron: {cron}")
        module = string_to_module(task.module)
        func = getattr(module, task.function)
        self.scheduler.add_job(func, args=task.payload,
                               trigger=CronTrigger.from_crontab(cron), id=id)

    def del_jobs(self, id: str = None):
        if id is None:
            logger.info("Deleting all jobs")
            self.scheduler.remove_all_jobs()
        elif self.job_id_exists(id):
            logger.info(f"Deleting jobs with tag: {id}")
            self.scheduler.remove_job(id)
    
    def get_jobs(self) -> List[str]:
        return self.scheduler.get_jobs()
