import grpc
from concurrent.futures import ThreadPoolExecutor
import generic_pb2
import generic_pb2_grpc
import logging
import json
import argparse
import traceback
import gc
from typing import Type
from pydantic import BaseModel

import asyncio
import torch

from rich.logging import RichHandler
from rich.console import Console

console = Console()
logging.basicConfig(
    level="INFO",
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True, show_time=False)]
)
logger = logging.getLogger(__name__)

from spt.models.remotecalls import MethodCallRequest, string_to_class, MethodCallError, string_to_module
from spt.models.jobs import JobStatuses
from spt.jobs import JobsTypes

class GenericServiceServicer(generic_pb2_grpc.GenericServiceServicer):
    def __init__(self, type: JobsTypes) -> None:
        super().__init__()
        self.type = type
        self.instances = {}  # Stores instances with their expiration time
        logger.info("Initialized Stateless Servicer")

    async def ProcessData(self, request, context):
        await asyncio.sleep(0)  # Yield control to support concurrency
        try:
            payload = json.loads(request.json_payload.decode('utf-8'))

            remote_class = request.remote_class
            remote_method = request.remote_method
            request_model_class = request.request_model_class
            response_model_class = request.response_model_class
            remote_function = request.remote_function
            remote_module = request.remote_module
            storage = request.storage
            keep_alive = request.keep_alive
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

            instance_key = (payload['remote_class'], storage)
            logger.info(f"Received request with storage: {storage} keep_alive: {keep_alive} instance_key: {instance_key} remote_class: {payload['remote_class']} remote_function: {payload['remote_function']} remote_method: {payload['remote_method']} payload: {payload['payload']}")
            
            if instance_key not in self.instances or self.instances[instance_key][1] < asyncio.get_event_loop().time():
                class_ = string_to_class(payload['remote_class'])
                instance = class_(self)
                self.instances[instance_key] = (instance, asyncio.get_event_loop().time() + keep_alive)
                if hasattr(instance, 'set_storage'):
                    instance.set_storage(storage)
                if hasattr(instance, 'set_keep_alive'):
                    instance.set_keep_alive(keep_alive)
            else:
                instance, expiration = self.instances[instance_key]
                self.instances[instance_key] = (instance, asyncio.get_event_loop().time() + keep_alive)  # Reset expiration

            if 'remote_function' in payload and payload['remote_function']:
                response = await self.execute_function(instance, payload)
            else:
                response = await self.execute_method(instance, payload)

            #response_model_class = string_to_class(payload['response_model_class'])
            #response = response_model_class.model_validate(**result)
            response = response.model_dump_json().encode('utf-8')
            return generic_pb2.GenericResponse(json_payload=response)

        except Exception as e:
            logger.error(f"Error processing data: {traceback.format_exc()}")
            error_response = {
                'status': JobStatuses.failed,
                'message': f"Failed to process request due to: {str(e)}"
            }
            return generic_pb2.GenericResponse(json_payload=json.dumps(error_response).encode('utf-8'))

    async def execute_function(self, instance, payload):
        module = string_to_module(payload['remote_module'])
        func = getattr(module, payload['remote_function'])
        result = func(instance)  # Assume func() takes the instance as an argument
        request_model_class = string_to_class(payload['request_model_class'])
        return request_model_class.model_validate(result)

    async def execute_method(self, instance, payload):
        method = getattr(instance, payload['remote_method'])
        request_model_class = string_to_class(payload['request_model_class'])
        arg = request_model_class.model_validate(payload['payload'])
        result = method(arg)
        return result

    async def cleanup_expired_instances(self):
        """ Cleanup expired instances periodically """
        while True:
            current_time = asyncio.get_event_loop().time()
            expired_keys = [key for key, (inst, exp) in self.instances.items() if exp < current_time]
            for key in expired_keys:
                inst, _ = self.instances.pop(key)
                if hasattr(inst, 'cleanup'):
                    inst.cleanup()
            await asyncio.sleep(60)  # Cleanup every minute

async def serve(max_workers=10, host="localhost", port=50051, type=JobsTypes.unknown):
    server = grpc.aio.server(ThreadPoolExecutor(max_workers=max_workers))
    service = GenericServiceServicer(type)
    generic_pb2_grpc.add_GenericServiceServicer_to_server(service, server)
    server.add_insecure_port(f"{host}:{port}")
    await server.start()
    logger.info(f"Service started. Listening on {host}:{port}")
    asyncio.create_task(service.cleanup_expired_instances())  # Start cleanup task for expired instances
    await server.wait_for_termination()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Start the Stateless Generic service.")
    parser.add_argument('--host', type=str, default='localhost', help='Host where the service will run')
    parser.add_argument('--port', type=int, default=50051, help='Port on which the service will listen')
    parser.add_argument('--type', type=str, default="generic", help='Type of service to start')

    args = parser.parse_args()
    asyncio.run(serve(max_workers=10, host=args.host, port=args.port, type=args.type))
