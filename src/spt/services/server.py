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
from typing import Dict, Tuple, Any, Optional
import os
from dataclasses import dataclass
from datetime import datetime, timedelta

from spt.models.workers import WorkerState

console = Console()

# Configure logging before any gRPC operations
logging.basicConfig(
    level="INFO",
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(
        console=console, rich_tracebacks=True, show_time=False)]
)

# Configure gRPC logging - only show errors
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GRPC_TRACE'] = 'none'

# Set gRPC logger to warning level
logging.getLogger('grpc').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

@dataclass
class InstanceInfo:
    instance: Any
    expiration: float
    last_activity: datetime
    keep_alive: int

class TaskManager:
    def __init__(self, interval: int = 60):
        self._tasks: Dict[str, asyncio.Task] = {}
        self._shutdown_events: Dict[str, asyncio.Event] = {}
        self.interval = interval
        self.logger = logger.getChild('TaskManager')
        self._task_running: Dict[str, bool] = {}

    async def start_task(self, name: str, coro, *args, **kwargs):
        # Check if task exists and is done or failed
        if name in self._tasks:
            task = self._tasks[name]
            if task.done():
                try:
                    # Check if task failed with exception
                    exc = task.exception()
                    if exc:
                        self.logger.error(f"Task {name} failed with error: {exc}")
                        # Clean up the failed task
                        self._tasks.pop(name, None)
                        self._shutdown_events.pop(name, None)
                        self._task_running[name] = False
                except asyncio.CancelledError:
                    self.logger.warning(f"Task {name} was cancelled")
            elif not self._task_running.get(name, False):
                self.logger.warning(f"Task {name} exists but not marked as running, restarting")
                await self.stop_task(name)
            else:
                self.logger.info(f"Task {name} is already running")
                return

        # Start new task
        self._shutdown_events[name] = asyncio.Event()
        self._task_running[name] = True
        self._tasks[name] = asyncio.create_task(self._run_task(name, coro, *args, **kwargs))
        self.logger.warning(f"Started task: {name}")

    async def stop_task(self, name: str, timeout: float = 5.0):
        if name in self._tasks and not self._tasks[name].done():
            self.logger.warning(f"Stopping task: {name}")
            self._shutdown_events[name].set()
            self._task_running[name] = False
            try:
                await asyncio.wait_for(self._tasks[name], timeout=timeout)
            except asyncio.TimeoutError:
                self.logger.warning(f"Task {name} did not stop gracefully, cancelling")
                self._tasks[name].cancel()
                try:
                    await self._tasks[name]
                except asyncio.CancelledError:
                    pass
            finally:
                self._tasks.pop(name, None)
                self._shutdown_events.pop(name, None)

    async def stop_all(self, timeout: float = 5.0):
        for name in list(self._tasks.keys()):
            await self.stop_task(name, timeout)

    async def _run_task(self, name: str, coro, *args, **kwargs):
        self.logger.warning(f"Starting task loop: {name}")
        retry_count = 0
        max_retries = 3
        
        while not self._shutdown_events[name].is_set():
            try:
                if not self._task_running.get(name, False):
                    self.logger.warning(f"Task {name} marked as not running, stopping")
                    break
                    
                self.logger.warning(f"Running task iteration: {name}")
                await coro(*args, **kwargs)
                self.logger.warning(f"Task iteration completed: {name}")
                retry_count = 0  # Reset retry count on successful execution
                
            except Exception as e:
                retry_count += 1
                self.logger.error(f"Error in task {name} (attempt {retry_count}/{max_retries}): {str(e)}\n{traceback.format_exc()}")
                if retry_count >= max_retries:
                    self.logger.error(f"Task {name} failed too many times, stopping")
                    self._task_running[name] = False
                    break
                await asyncio.sleep(1)  # Short delay before retry
                continue
            
            try:
                self.logger.warning(f"Task {name} waiting for {self.interval} seconds")
                await asyncio.sleep(self.interval)
            except asyncio.CancelledError:
                self.logger.warning(f"Task {name} was cancelled during sleep")
                break
            except Exception as e:
                self.logger.error(f"Error during task {name} sleep: {str(e)}")
                await asyncio.sleep(1)  # Fallback sleep on error
                
        self._task_running[name] = False
        self.logger.warning(f"Task loop ended: {name}")

    def is_task_running(self, name: str) -> bool:
        return self._task_running.get(name, False) and name in self._tasks and not self._tasks[name].done()

class GenericServiceServicer(generic_pb2_grpc.GenericServiceServicer):
    def __init__(self, type: JobsTypes) -> None:
        super().__init__()
        self.type: JobsTypes = type
        self.instances: Dict[Tuple[str, str, str], InstanceInfo] = {}
        self.task_manager = TaskManager(interval=30)  # Reduced interval for more frequent checks
        self._logger = logger  # Keep a protected reference to the logger
        self._logger.warning("[*] Initialized Stateless Servicer")

    @property
    def logger(self):
        return self._logger

    async def cleanup_instances(self):
        """Cleanup expired instances with better state management"""
        try:
            self.logger.warning("[*] Running cleanup task...")
            current_time = datetime.now()
            
            # First, let each service instance handle its worker cleanup
            instances_copy = self.instances.copy()
            for key, info in instances_copy.items():
                try:
                    # Let service handle its workers first
                    if hasattr(info.instance, 'check_workers'):
                        info.instance.check_workers()
                    
                    # Handle keep-alive for active instances
                    if hasattr(info.instance, 'decrease_keep_alive'):
                        info.instance.decrease_keep_alive()
                        # Get the updated keep_alive value after decrease
                        current_keep_alive = info.instance.get_keep_alive() if hasattr(info.instance, 'get_keep_alive') else info.keep_alive
                    else:
                        current_keep_alive = info.keep_alive

                    # Remove instance if keep_alive reached 0 or instance has expired
                    if current_keep_alive <= 0 or current_time.timestamp() > info.expiration:
                        self.logger.warning(f"[*] Instance {key} being removed - Keep alive: {current_keep_alive}, Expired: {current_time.timestamp() > info.expiration}")
                        if key in self.instances:
                            if hasattr(info.instance, 'cleanup'):
                                info.instance.cleanup()
                            self.instances.pop(key, None)
                    else:
                        # Only update expiration if the service still has active workers
                        if hasattr(info.instance, 'instances') and len(info.instance.instances) > 0:
                            self.instances[key] = InstanceInfo(
                                instance=info.instance,
                                expiration=current_time.timestamp() + (current_keep_alive * 60),
                                last_activity=current_time,
                                keep_alive=current_keep_alive
                            )
                            self.logger.warning(f"[*] Updated active instance {key} with {len(info.instance.instances)} workers, keep_alive: {current_keep_alive}")
                        else:
                            self.logger.warning(f"[*] Instance {key} has no active workers, will expire")
                            
                except Exception as e:
                    self.logger.error(f"Error processing instance {key}: {str(e)}\n{traceback.format_exc()}")
            
            self.logger.warning(f"[*] Cleanup completed. Active instances: {len(self.instances)}")
        except Exception as e:
            self.logger.error(f"Critical error in cleanup task: {str(e)}\n{traceback.format_exc()}")
            # Don't re-raise to keep the task alive

    async def ProcessData(self, request: generic_pb2.GenericRequest, context: grpc.aio.ServicerContext) -> generic_pb2.GenericResponse:
        # Ensure cleanup task is running with verification
        await self.task_manager.start_task('cleanup', self.cleanup_instances)
        if not self.task_manager.is_task_running('cleanup'):
            self.logger.error("Cleanup task failed to start or died, attempting restart")
            await self.task_manager.start_task('cleanup', self.cleanup_instances)
        
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

            self.logger.warning(f"[*] Received request with worker_id {worker_id} storage: {storage} keep_alive: {keep_alive} instance_key: {instance_key} remote_class: {payload['remote_class']} remote_function: {payload['remote_function']} remote_method: {payload['remote_method']} response_model_class: {payload['response_model_class']}")

            current_time = datetime.now()

            if 'remote_function' in payload and payload['remote_function']:
                response = await self.execute_function(payload)
            else:
                if instance_key not in self.instances:
                    class_ = string_to_class(payload['remote_class'])
                    instance = class_(self)
                    self.instances[instance_key] = InstanceInfo(
                        instance=instance,
                        expiration=current_time.timestamp() + (keep_alive * 60),
                        last_activity=current_time,
                        keep_alive=keep_alive
                    )
                    if hasattr(instance, 'set_storage'):
                        instance.set_storage(storage)
                    if hasattr(instance, 'set_keep_alive'):
                        instance.set_keep_alive(keep_alive)
                    if hasattr(instance, 'set_logger'):
                        instance.set_logger(self.logger)  # Use the protected logger reference
                else:
                    info = self.instances[instance_key]
                    info.last_activity = current_time
                    info.expiration = current_time.timestamp() + (keep_alive * 60)
                    info.keep_alive = keep_alive
                    instance = info.instance
                    if hasattr(instance, 'set_keep_alive'):
                        instance.set_keep_alive(keep_alive)

                response = await self.execute_method(instance, payload)

            response = response.model_dump_json().encode('utf-8')
            return generic_pb2.GenericResponse(json_payload=response, response_model_class=payload['response_model_class'])

        except (ValidationError, ValueError) as e:
            self.logger.error(f"Validation error processing data: {str(e)} stack trace: {traceback.format_exc()}")
            error = MethodCallError(message=f"Failed to process request due to validation error: {str(e)}", status=JobStatuses.failed, error=traceback.format_exc())
            return generic_pb2.GenericResponse(json_payload=error.model_dump_json().encode("utf-8"), response_model_class="MethodCallError")
        except Exception as e:
            self.logger.error(f"Error processing data: {traceback.format_exc()}")
            if hasattr(instance, 'status'):
                instance.status = WorkerState.idle
            if hasattr(instance, 'cleanup'):
                instance.cleanup()
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

async def serve(max_workers: int = 10, host: str = "localhost", port: int = 50051, type: JobsTypes = JobsTypes.unknown):
    server_options = [
        ('grpc.max_send_message_length', 100 * 1024 * 1024),
        ('grpc.max_receive_message_length', 100 * 1024 * 1024),
        ('grpc.so_reuseport', 1)
    ]
    
    server = grpc.aio.server(
        ThreadPoolExecutor(max_workers=max_workers),
        options=server_options
    )
    service = GenericServiceServicer(type)
    generic_pb2_grpc.add_GenericServiceServicer_to_server(service, server)
    server.add_insecure_port(f"{host}:{port}")
    await server.start()
    logger.warning(f"Service started. Listening on {host}:{port} type {type}")
    
    # Start cleanup task
    await service.task_manager.start_task('cleanup', service.cleanup_instances)
    
    try:
        await server.wait_for_termination()
    finally:
        await service.task_manager.stop_all()

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
    
    try:
        asyncio.run(serve(max_workers=10, host=args.host,
                    port=args.port, type=args.type))
    except KeyboardInterrupt:
        logger.warning("Server shutting down...")
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        raise
