from fastapi import FastAPI
from contextlib import asynccontextmanager
from spt.models.jobs import JobsTypes
from spt.jobs import Jobs
from rich.logging import RichHandler
from rich.console import Console
import logging

console = Console()

logging.basicConfig(
    level="INFO",
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(
        console=console, rich_tracebacks=True, show_time=False)]
)

logger = logging.getLogger("API")

jobs = None
dispatcher = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global jobs
    global dispatcher
    if jobs is None:
        jobs_types = [JobsTypes.image_generation, JobsTypes.llm_generation,
                    JobsTypes.audio_generation, JobsTypes.video_generation]
        jobs = {job_type: Jobs(job_type) for job_type in jobs_types}
        app.state.jobs = jobs
    if dispatcher is None:
        from spt.dispatcher import Dispatcher
        dispatcher = Dispatcher()
        app.state.dispatcher = dispatcher
    yield
    # Clean up the ML models and release the resources
    for job in jobs:
        job.stop()

app = FastAPI(
    lifespan=lifespan,
    title="spt-smi",
    version="0.0.5",
    description="""
                spt-smi API 🚀

                # Scalable Models Inferences

                You can request inferences from several models using this API.
                The models are hosted on the Sponge Theory infrastructure, so you can request
                inferences from several models at the same time using a hidden queue mecanism.

                This API can be deployed on a docker container for your own use.
                It does include the following stacks : 
                    - RabbitMQ for message broker
                    - Redis for caching
                    - FastAPI for the API
                    - Minio for the storage
                It support dymanic scaling and load balancing, GRPC distrubuted remote services with workers
                for each IA models.
                Websocket streaming is also supported (ie. STT)
                """,
    contact={
        "name": "Sponge Theory",
        "url": "https://sponge-theory.ai",
        "email": "contact@sponge-theory.io",
    })

import spt.api.router
