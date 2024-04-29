import grpc
import logging
import generic_pb2
import generic_pb2_grpc
from spt.jobs import Job
import json
from rich.logging import RichHandler
from rich.console import Console

console = Console()

logging.basicConfig(
    level="INFO",
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(
        console=console, rich_tracebacks=True, show_time=False)]
)

logger = logging.getLogger(__name__)

class GenericClient:
    def __init__(self, host) -> None:

        self.channel = grpc.insecure_channel(f'{host}')
        self.stub = generic_pb2_grpc.GenericServiceStub(self.channel)

    def call_remote_function(self, remote_module: str, remote_function: str, payload: dict, response_model_class:str)-> generic_pb2.GenericResponse:
        logger.info(
            f"Execute remote function {remote_function} with payload: {payload}")
        request = generic_pb2.GenericRequest(
            json_payload=json.dumps(payload).encode('utf-8'), 
            remote_function=remote_function,
            remote_module = remote_module,
            response_model_class=response_model_class
        )
        response = self.stub.ProcessData(request)
        logger.info(f"Response with payload: {response.json_payload}")
        return json.loads(response.json_payload)

    def process_data(self, job: Job) -> generic_pb2.GenericResponse:
        string_payload = json.dumps(job.payload)
        json_payload = string_payload.encode('utf-8')
        logger.info(
            f"Execute service request Class {job.remote_class} Method {job.remote_method} ")
        request = generic_pb2.GenericRequest(
            json_payload=json_payload, 
            remote_class=job.remote_class, 
            remote_method=job.remote_method, 
            request_model_class=job.request_model_class, 
            response_model_class=job.response_model_class,
            storage=job.storage,
            keep_alive=job.keep_alive)
        
        response = self.stub.ProcessData(request)
        logger.info(f"Service response with payload: {response.json_payload}")
        return response
