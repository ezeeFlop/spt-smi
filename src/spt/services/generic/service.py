import grpc
from concurrent import futures
import generic_pb2
import generic_pb2_grpc
import logging
from spt.models.remotecalls import MethodCallRequest, string_to_class, MethodCallError, string_to_module
from spt.models.jobs import JobStatuses
from pydantic import BaseModel, ValidationError, validator
import json 
from typing import Type, Any
import argparse
import traceback
from pynvml import nvmlInit, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex, nvmlDeviceGetName, nvmlDeviceGetMemoryInfo, nvmlDeviceGetUtilizationRates, NVMLError, nvmlShutdown
import spt.services.gpu
from rich.logging import RichHandler
from rich.console import Console

console = Console()

logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(
        console=console, rich_tracebacks=True, show_time=False)]
)
logger = logging.getLogger('gRPC-Server')

class GenericServiceServicer(generic_pb2_grpc.GenericServiceServicer):
    def ProcessData(self, request, context):
        payload = json.loads(request.json_payload.decode('utf-8'))
        remote_class = request.remote_class
        remote_method = request.remote_method
        request_model_class = request.request_model_class
        response_model_class = request.response_model_class
        remote_function = request.remote_function
        remote_module = request.remote_module
        payload = {
            'remote_class': remote_class,
            'remote_method': remote_method,
            'request_model_class': request_model_class,
            'response_model_class': response_model_class,
            'remote_function': remote_function,
            'remote_module': remote_module,
            'payload': payload
        }
        response = b'{}'
        logger.info(f"Received request with payload: {payload}")
        if remote_function is not None and len(remote_function) > 0:
            try:
                logger.info(f"Try to call {remote_function} with payload: {payload}")
                request = MethodCallRequest.model_validate(payload)
                response = self.execute_function(request)

                logger.info(response)

            except Exception as e:
                response = MethodCallError(
                    status=JobStatuses.failed, message=f"Failed to call remote function {remote_function}: {str(e)}: {traceback.format_exc()}")
                logger.error(traceback.format_exc())
        else:
            try:
                logger.info(f"Try to call {remote_class}.{remote_method} with payload: {payload}")
                request = MethodCallRequest.model_validate(payload)
                response = self.execute_method(request)
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

    def execute_method(self, request:MethodCallRequest) -> Type[BaseModel]:
        class_ = string_to_class(request.remote_class)
        instance = class_(self)

        method = getattr(instance, request.remote_method)
        request_model_class = string_to_class(request.request_model_class)
        arg = request_model_class.model_validate(request.payload)
        result = method(arg)

        return result


def gpu_infos(display=False):
    try:
        nvmlInit()
        device_count = nvmlDeviceGetCount()
        gpus = []

        for i in range(device_count):
            gpu = {}
            handle = nvmlDeviceGetHandleByIndex(i)
            name = nvmlDeviceGetName(handle)
            memory_info = nvmlDeviceGetMemoryInfo(handle)
            utilization = nvmlDeviceGetUtilizationRates(handle)

            gpu['name'] = name
            gpu['memory_total_gb'] = memory_info.total / (1024 ** 3)
            gpu['memory_used_gb'] = memory_info.used / (1024 ** 3)
            gpu['memory_free_gb'] = memory_info.free / (1024 ** 3)
            gpu['utilization_gpu_percent'] = utilization.gpu
            gpu['utilization_memory_percent'] = utilization.memory
            gpus.append(gpu)

            if display:
                logger.info(f"GPU {i}: {name}")
                logger.info(
                    f"  Mémoire Totale: {gpu['memory_total_gb']:.2f} GB")
                logger.info(
                    f"  Mémoire Utilisée: {gpu['memory_used_gb']:.2f} GB")
                logger.info(f"  Mémoire Libre: {gpu['memory_free_gb']:.2f} GB")
                logger.info(
                    f"  Utilisation GPU: {gpu['utilization_gpu_percent']}%")
                logger.info(
                    f"  Utilisation Mémoire: {gpu['utilization_memory_percent']}%")

        return json.dumps(gpus)

    except NVMLError as e:
        logger.error(f"Erreur NVML: {e}")
        return json.dumps({'error': str(e)})

    finally:
        try:
            nvmlShutdown()
        except NVMLError as e:
            logger.error(f"Erreur lors de la fermeture de NVML: {e}")


def serve(max_workers=10, host="localhost", port=50052):
    server = grpc.server(futures.ThreadPoolExecutor(
        max_workers=max_workers), compression=grpc.Compression.Gzip)
    generic_pb2_grpc.add_GenericServiceServicer_to_server(
        GenericServiceServicer(), server)
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

   # Parse les arguments de la ligne de commande
   args = parser.parse_args()

   # Affichage des informations et démarrage du service
   logger.info(
       f"Starting Generic service on host {args.host} port {args.port}")
   gpu_infos(display=True)
   serve(max_workers=10, host=args.host, port=args.port)
