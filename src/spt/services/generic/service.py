import grpc
from concurrent import futures
import generic_pb2
import generic_pb2_grpc
import logging
from spt.models.remotecalls import MethodCallRequest, string_to_class, MethodCallError, string_to_module
from spt.models.jobs import JobStatuses
import json 
from typing import Type
import argparse
import traceback
from spt.services.gpu import gpu_infos
from rich.logging import RichHandler
from rich.console import Console
from spt.scheduler import Scheduler
from spt.jobs import JobsTypes
import gc
from pydantic import BaseModel, Field, validator

console = Console()

logging.basicConfig(
    level="INFO",
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(
        console=console, rich_tracebacks=True, show_time=False)]
)

logger = logging.getLogger(__name__)

class GenericServiceServicer(generic_pb2_grpc.GenericServiceServicer):
    def __init__(self, type: JobsTypes) -> None:
        super().__init__()
        logger.info("Init Servicer")
        self.current_class = None
        self.current_instance  = None
        self.type = type

    def cleanup(self):
        logger.info("Servicer Cleanup")
        
        if self.current_instance is not None:
            if self.current_instance.should_cleanup():
                scheduler.del_jobs(id=f"{self.current_class}_cleanup")
                self.current_instance.cleanup()
                del self.current_instance
                gc.collect()
                self.current_instance = self.current_class = None

    def ProcessData(self, request, context):
        payload = json.loads(request.json_payload.decode('utf-8'))
        remote_class = request.remote_class
        remote_method = request.remote_method
        request_model_class = request.request_model_class
        response_model_class = request.response_model_class
        remote_function = request.remote_function
        remote_module = request.remote_module
        keep_alive = request.keep_alive
        storage = request.storage
        payload = {
            'remote_class': remote_class,
            'remote_method': remote_method,
            'request_model_class': request_model_class,
            'response_model_class': response_model_class,
            'remote_function': remote_function,
            'remote_module': remote_module,
            'keep_alive': keep_alive,
            'storage': storage,
            'payload': payload
        }
        response = b'{}'
        logger.info(f"Received request with storage: {storage} keep_alive: {keep_alive}")  
        if remote_function is not None and len(remote_function) > 0:
            try:
                logger.info(f"Try to call {remote_function} ")
                request = MethodCallRequest.model_validate(payload)
                response = self.execute_function(request)

                logger.info(response)

            except Exception as e:
                response = MethodCallError(
                    status=JobStatuses.failed, message=f"Failed to call remote function {remote_function}: {str(e)}: {traceback.format_exc()}")
                logger.error(traceback.format_exc())
        else:
            try:
                logger.info(f"Try to call {remote_class}.{remote_method}")
                request = MethodCallRequest.model_validate(payload)
                response = self.execute_method(request=request, storage=storage, keep_alive=keep_alive)
                logger.info(response)
            except Exception as e:
                response = MethodCallError(status=JobStatuses.failed, message=f"Failed to dispatch job: {str(e)}: {traceback.format_exc()}")
                logger.error(logger.error(traceback.format_exc()))

        response = response.model_dump_json().encode('utf-8')

        return generic_pb2.GenericResponse(json_payload=response)
    
    def execute_function(self, request:MethodCallRequest) -> Type[BaseModel]:
        module = string_to_module(request.remote_module)
        request_model_class = string_to_class(request.response_model_class)

        func = getattr(module, request.remote_function)
        result = func()
        result = request_model_class.model_validate(result)

        return result

    def execute_method(self, request:MethodCallRequest, storage:str, keep_alive:int) -> Type[BaseModel]:
        if self.current_class is None:

            self.current_class = request.request_model_class
            class_ = string_to_class(request.remote_class)
            self.current_instance = class_(self)
            scheduler.add_job_local_method(
                cleanup, cron='* * * * *', id=f"{self.current_class}_cleanup")

        elif self.current_class != request.request_model_class:
            self.current_instance.cleanup()
            del self.current_instance
            gc.collect()
            self.current_class = request.request_model_class
            class_ = string_to_class(request.remote_class)
            self.current_instance = class_(self)
            scheduler.add_job_local_method(
                cleanup, cron='* * * * *', id=f"{self.current_class}_cleanup")

        method = getattr(self.current_instance, request.remote_method)

        self.current_instance.set_storage(storage)
        self.current_instance.set_keep_alive(keep_alive)
        request_model_class = string_to_class(request.request_model_class)
        arg = request_model_class.model_validate(request.payload)

        result = method(arg)

        return result

scheduler = None
service = None

def cleanup():
    service.cleanup()

def serve(max_workers=1, host="localhost", port=50052, type="generic"):
    global scheduler, service

    server = grpc.server(futures.ThreadPoolExecutor(
        max_workers=max_workers), compression=grpc.Compression.Gzip)
    
    service = GenericServiceServicer(type)
    scheduler = Scheduler()
    scheduler.start()
    scheduler.del_jobs(id=f"{type}_cleanup")
    scheduler.del_jobs()

    generic_pb2_grpc.add_GenericServiceServicer_to_server(
        service, server)
    server.add_insecure_port(f"{host}:{port}")
    server.start()

    logger.info("Service started. Listening on port %d", port)
    server.wait_for_termination()


if __name__ == '__main__':
   parser = argparse.ArgumentParser(description="Start the Generic service.")
   parser.add_argument('--host', type=str, default='localhost',
                       help='Host where the service will run')
   parser.add_argument('--port', type=int, default=50051,
                       help='Port on which the service will listen')
   parser.add_argument('--type', type=str, default="generic",
                       help='Type of service')

   # Parse les arguments de la ligne de commande
   args = parser.parse_args()

   gpu_infos(display=True)

   # Affichage des informations et démarrage du service
   logger.info(
       f"Starting {args.type} service on host {args.host} port {args.port} type {args.type}")
   gpu_infos(display=True)
   serve(max_workers=1, host=args.host, port=args.port, type=args.type)
