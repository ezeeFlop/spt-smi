import grpc
from concurrent import futures
import generic_pb2
import generic_pb2_grpc
import logging
from config import LLM_SERVICE_PORT, LLM_SERVICE_HOST
from spt.models.remotecalls import MethodCallRequest, string_to_class
from pydantic import BaseModel, ValidationError, validator
import json 
from typing import Type, Any
import argparse
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
            response = response.model_dump_json().encode('utf-8')
        except ValidationError as e:
            print(f"Validation Error: {str(e)}")
        return generic_pb2.GenericResponse(json_payload=response)

    def execute_method(self, request:MethodCallRequest) -> Type[BaseModel]:
        try:
            # Importation dynamique des classes
            #module = importlib.import_module('spt.services.models')
            #class_ = getattr(module, request.remote_class)
            class_ = string_to_class(request.remote_class)
            instance = class_()

            # Obtention et exécution de la méthode
            method = getattr(instance, request.remote_method)
            request_model_class = string_to_class(request.request_model_class)
            arg = request_model_class.model_validate(request.payload)
            result = method(arg)
            return result

        except ImportError as e:
            # Gestion des erreurs d'importation
            return f"Error importing module or class: {str(e)}"
        except AttributeError as e:
            # Gestion des erreurs liées à l'absence de méthodes ou de classes
            return f"Method or class not found: {str(e)}"
        except TypeError as e:
            # Gestion des erreurs de paramètres incorrects ou manquants
            return f"Error in method arguments: {str(e)}"
        except Exception as e:
            # Gestion de toutes les autres erreurs possibles
            return f"An error occurred: {str(e)}"

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
   serve(max_workers=10, host=args.host, port=args.port)
