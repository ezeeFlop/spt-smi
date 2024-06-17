from spt.models.remotecalls import MethodCallRequest, string_to_class, MethodCallError, string_to_module
from spt.models.jobs import JobStatuses
from spt.jobs import JobsTypes
import asyncio
from pydantic import BaseModel, ValidationError
import grpc
from concurrent.futures import ThreadPoolExecutor
import generic_pb2
import generic_pb2_grpc
import logging
import json
import argparse
import traceback
from rich.logging import RichHandler
from rich.console import Console
from typing import Dict, Tuple, Any

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
        self.type: JobsTypes = type
        # Stores instances with their expiration time
        self.instances: Dict[Tuple[str, str, str], Tuple[Any, float]] = {}
        logger.info("[*] Initialized Stateless Servicer")

    async def ProcessData(self, request: generic_pb2.GenericRequest, context: grpc.aio.ServicerContext) -> generic_pb2.GenericResponse:
        await asyncio.sleep(0)  # Yield control to support concurrency
        try:
            payload: dict = json.loads(request.json_payload.decode('utf-8'))
            remote_class: str = request.remote_class
            remote_method: str = request.remote_method
            request_model_class: str = request.request_model_class
            response_model_class: str = request.response_model_class
            remote_function: str = request.remote_function
            remote_module: str = request.remote_module
            storage: str = request.storage
            keep_alive: int = request.keep_alive
            worker_id: str = request.worker_id
            payload = {
                'payload': payload,
                'remote_class': remote_class,
                'remote_method': remote_method,
                'request_model_class': request_model_class,
                'response_model_class': response_model_class,
                'remote_function': remote_function,
                'remote_module': remote_module,
                'keep_alive': keep_alive,
                'storage': storage
            }

            instance_key: Tuple[str, str, str] = (
                payload['remote_class'], payload['remote_method'], storage)

            logger.info(f"[*] Received request with worker_id {worker_id} storage: {storage} keep_alive: {keep_alive} instance_key: {instance_key} remote_class: {payload['remote_class']} remote_function: {payload['remote_function']} remote_method: {payload['remote_method']} response_model_class: {payload['response_model_class']}")

            if 'remote_function' in payload and payload['remote_function']:
                response = await self.execute_function(payload)
            else:
                if instance_key not in self.instances or self.instances[instance_key][1] < asyncio.get_event_loop().time():
                    class_ = string_to_class(payload['remote_class'])
                    instance = class_(self)
                    self.instances[instance_key] = (
                        instance, asyncio.get_event_loop().time() + keep_alive * 60)
                    if hasattr(instance, 'set_storage'):
                        instance.set_storage(storage)
                    if hasattr(instance, 'set_keep_alive'):
                        instance.set_keep_alive(keep_alive)
                    if hasattr(instance, 'set_logger'):
                        instance.set_logger(logger)
                else:
                    instance, expiration = self.instances[instance_key]
                    self.instances[instance_key] = (
                        instance, asyncio.get_event_loop().time() + keep_alive * 60)  # Reset expiration
                    if hasattr(instance, 'set_keep_alive'):
                        instance.set_keep_alive(keep_alive)

                response = await self.execute_method(instance, payload)

            response = response.model_dump_json().encode('utf-8')
            return generic_pb2.GenericResponse(json_payload=response, response_model_class=payload['response_model_class'])

        except (ValidationError, ValueError) as e:
            logger.error(f"Validation error processing data: {str(e)}")
            error = MethodCallError(message=f"Failed to process request due to validation error: {str(e)}", status=JobStatuses.failed, error=traceback.format_exc())
            return generic_pb2.GenericResponse(json_payload=error.model_dump_json().encode("utf-8"), response_model_class="MethodCallError")
        except Exception as e:
            logger.error(f"Error processing data: {traceback.format_exc()}")
            error = MethodCallError(
                message=f"Failed to process request due to: {str(e)}", status=JobStatuses.failed, error=traceback.format_exc())
            return generic_pb2.GenericResponse(json_payload=error.model_dump_json().encode("utf-8"), response_model_class="MethodCallError")

    async def execute_function(self, payload: dict) -> BaseModel:
        module = string_to_module(payload['remote_module'])
        func = getattr(module, payload['remote_function'])
        result = await func()if asyncio.iscoroutinefunction(func) else func()
        response_model_class = string_to_class(payload['response_model_class'])
        return response_model_class.model_validate(result)

    async def execute_method(self, instance: Any, payload: dict) -> BaseModel:
        method = getattr(instance, payload['remote_method'])
        request_model_class = string_to_class(payload['request_model_class'])
        arg = request_model_class.model_validate(payload['payload'])
        result = await method(arg) if asyncio.iscoroutinefunction(method) else method(arg)
        return result

    async def cleanup_expired_instances(self) -> None:
        """ Cleanup expired instances periodically """
        while True:
            logger.info("  [*] Cleaning up expired instances...")
            current_time = asyncio.get_event_loop().time()
            expired_keys = [
                key for key, (inst, exp) in self.instances.items() if exp < current_time]
            logger.info(f"  [*] Found {len(expired_keys)} expired instances:")
            for key in expired_keys:
                inst, _ = self.instances.pop(key)
                logger.info(f"  [*] Cleaning up expired instance: {key}")
                if hasattr(inst, 'cleanup'):
                    inst.cleanup()

            running_keys = [
                key for key, (inst, exp) in self.instances.items() if exp > current_time]
            logger.info(f"  [*] Found {len(running_keys)} running instances:")

            for key in running_keys:
                inst, _ = self.instances[key]
                if hasattr(inst, 'decrease_keep_alive'):
                    inst.decrease_keep_alive()
                if hasattr(inst, 'check_workers'):
                    inst.check_workers()
            await asyncio.sleep(60)  # Cleanup every minute

async def serve(max_workers: int = 10, host: str = "localhost", port: int = 50051, type: JobsTypes = JobsTypes.unknown):
    server = grpc.aio.server(ThreadPoolExecutor(max_workers=max_workers))
    service = GenericServiceServicer(type)
    generic_pb2_grpc.add_GenericServiceServicer_to_server(service, server)
    server.add_insecure_port(f"{host}:{port}")
    await server.start()
    logger.info(f"Service started. Listening on {host}:{port} type {type}")
    # Start cleanup task for expired instances
    asyncio.create_task(service.cleanup_expired_instances())
    await server.wait_for_termination()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Start the Stateless Generic service.")
    parser.add_argument('--host', type=str, default='localhost',
                        help='Host where the service will run')
    parser.add_argument('--port', type=int, default=50051,
                        help='Port on which the service will listen')
    parser.add_argument('--type', type=str, default="generic",
                        help='Type of service to start')

    args = parser.parse_args()
    asyncio.run(serve(max_workers=10, host=args.host,
                port=args.port, type=args.type))
