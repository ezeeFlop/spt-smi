import grpc
from concurrent import futures
import generic_pb2
import generic_pb2_grpc
import logging
from config import LLM_SERVICE_PORT, LLM_SERVICE_HOST
from spt.models.remotecalls import MethodCallRequest, string_to_class, MethodCallError
from spt.models.jobs import JobStatuses
from pydantic import BaseModel, ValidationError, validator
import json 
from typing import Type, Any
import argparse
import traceback
from pynvml import nvmlInit, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex, nvmlDeviceGetName, nvmlDeviceGetMemoryInfo, nvmlDeviceGetUtilizationRates, NVMLError, nvmlShutdown

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('grpc-server')

class GenericServiceServicer(generic_pb2_grpc.GenericServiceServicer):
    def ProcessData(self, request, context):
        payload = json.loads(request.json_payload.decode('utf-8'))
        remote_class = request.remote_class
        remote_method = request.remote_method
        request_model_class = request.request_model_class
        response_model_class = request.response_model_class

        payload = {
            'remote_class': remote_class,
            'remote_method': remote_method,
            'request_model_class': request_model_class,
            'response_model_class': response_model_class,
            'payload': payload
        }
        response = {}
        try:
            logger.info(f"Try to call {remote_class}.{remote_method} with payload: {payload}")
            request = MethodCallRequest.model_validate(payload)
            response = self.execute_method(request)
            logger.info(response)
        except Exception as e:
            response = MethodCallError(status=JobStatuses.failed, message=f"Failed to dispatch job: {str(e)}: {traceback.format_exc()}")
            logger.error(response)

        response = response.model_dump_json().encode('utf-8')

        return generic_pb2.GenericResponse(json_payload=response)

    def execute_method(self, request:MethodCallRequest) -> Type[BaseModel]:
        class_ = string_to_class(request.remote_class)
        instance = class_()

        method = getattr(instance, request.remote_method)
        request_model_class = string_to_class(request.request_model_class)
        arg = request_model_class.model_validate(request.payload)
        result = method(arg)

        return result

def gpu_infos():
    try:
        # Tentative d'initialisation de NVML
        nvmlInit()
    except NVMLError as e:
        logger.error(f"NVML ne peut pas être initialisé : {e}")
        return

    try:
        # Compter le nombre de GPU disponibles
        device_count = nvmlDeviceGetCount()
        logger.info(f"Nombre de GPU détectés : {device_count}")

        # Itérer sur chaque GPU et afficher les informations
        for i in range(device_count):
            handle = nvmlDeviceGetHandleByIndex(i)
            name = nvmlDeviceGetName(handle)
            memory_info = nvmlDeviceGetMemoryInfo(handle)
            utilization = nvmlDeviceGetUtilizationRates(handle)
            
            logger.info(f"GPU {i}: {name}")
            logger.info(f"  Mémoire Totale: {memory_info.total / (1024 ** 3):.2f} GB")
            logger.info(f"  Mémoire Utilisée: {memory_info.used / (1024 ** 3):.2f} GB")
            logger.info(f"  Mémoire Libre: {memory_info.free / (1024 ** 3):.2f} GB")
            logger.info(f"  Utilisation GPU: {utilization.gpu}%")
            logger.info(f"  Utilisation Mémoire: {utilization.memory}%")
    
    except NVMLError as e:
        logger.error(f"Erreur NVML lors de l'accès aux informations GPU : {e}")

    finally:
        # Assurez-vous de fermer NVML proprement
        try:
            nvmlShutdown()
        except NVMLError as e:
            logger.error(f"Erreur lors de la fermeture de NVML : {e}")

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
   gpu_infos()
   serve(max_workers=10, host=args.host, port=args.port)
