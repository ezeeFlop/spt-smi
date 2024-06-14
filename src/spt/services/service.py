
import socket
from spt.models.jobs import JobStorage
from spt.models.workers import WorkerState, WorkerStreamManageRequest, WorkerStreamManageResponse, WorkerBaseRequest
from spt.storage import Storage
from spt.models.workers import WorkerConfigs, WorkerConfig
from spt.services.worker import Worker
from spt.services.server import GenericServiceServicer
import asyncio
from pydantic import BaseModel, ValidationError
import importlib
import logging
from typing import Dict, Any, Optional

class Service:
    def __init__(self, servicer: GenericServiceServicer, max_run_time: int = 600) -> None:
        self.servicer: GenericServiceServicer = servicer
        self.storage_type: Optional[str] = None
        self.keep_alive: int = 15
        self.storage: Optional[Storage] = None
        self.workers: Dict[str, Worker] = {}
        self.worker_configs: WorkerConfig = WorkerConfigs.get_configs().workers_configs
        self.max_run_time: int = max_run_time
        self.instances: Dict[str, Worker] = {}
        self.logger: logging.Logger = None
    
    def set_logger(self, logger: logging.Logger):
        self.logger = logger

    def check_workers(self):
        self.logger.info("Service Check Workers...")
        for key, worker in list(self.instances.items()):
            if worker.get_status() == WorkerState.working and worker.get_duration() > self.max_run_time:
                self.logger.info(f"Stopping worker {key} due to long run time")
                worker.stop()
                worker.cleanup()
                del self.instances[key]

    def cleanup(self):
        self.logger.info(f"Service Cleanup {self.keep_alive} minutes left")
        for key, worker in list(self.instances.items()):
            if worker.get_status() == WorkerState.idle:
                self.logger.info(f"Cleaning up IDLE worker: {key}")
                worker.cleanup()
                del self.instances[key]
            elif worker.get_duration() > self.max_run_time and (worker.get_status() == WorkerState.working or worker.get_status() == WorkerState.streaming):
                self.logger.info(f"Stopping worker {key} due to long run time")
                worker.stop()
                worker.cleanup()
                del self.instances[key]

    def chunked_request(self, request: Any):
        self.logger.info(f"Chunked request: {request}")

    async def get_worker(self, worker_id: str) -> Worker:
        try:
            if (worker_id not in self.worker_configs):
                raise ValueError(
                    f'Worker class for model {worker_id} not found')

            worker_info:WorkerConfig = self.worker_configs[worker_id]

            # Check if an instance already exists and is IDLE
            if (worker_id in self.instances):
                worker_instance = self.instances[worker_id]
                if worker_instance.get_status() == WorkerState.idle:
                    return worker_instance

            module_path, class_name = worker_info.worker.rsplit('.', 1)
            module = importlib.import_module(module_path)
            worker_class = getattr(module, class_name)

            worker = worker_class(
                name=class_name, service=self, model=worker_info.model, logger=self.logger)
            self.instances[worker_id] = worker
            return worker

        except Exception as e:
            self.logger.error(f"Error in get_worker: {e}")
            raise

    async def work(self, request: WorkerBaseRequest) -> BaseModel:
        worker = await self.get_worker(request.worker_id)
        result = await worker.work(request)
        worker.stop()
        return result

    def find_free_port(self) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            return int(s.getsockname()[1])

    async def stream(self, request: WorkerStreamManageRequest) -> WorkerStreamManageResponse:
            # Get the hostname of the current machine
            hostname = socket.gethostname()
            
            # Get the IP address associated with the hostname
            ip_address = socket.gethostbyname(hostname)
            
            # Retrieve the worker instance associated with the requested model
            worker = await self.get_worker(request.worker_id)
            
            # Find a free port for input
            input_port = self.find_free_port()
            
            # Find a free port for output
            output_port = self.find_free_port()

            # Start the stream in the background using asyncio
            worker.stream_task = asyncio.create_task(
                worker.start_stream(ip=ip_address, 
                                    input_port=input_port, 
                                    output_port=output_port,
                                    intype=request.intype,
                                    outtype=request.outtype,
                                    timeout=request.timeout)
                                    )
            
            # Return a response with the stream details
            return WorkerStreamManageResponse(host=hostname,
                                            state=WorkerState.streaming, 
                                            inport=input_port, 
                                            outport=output_port, 
                                            ip_address=ip_address)

    def set_storage(self, storage: str):
        self.storage_type = storage
        if self.storage_type == JobStorage.s3 and self.storage is None:
            self.storage = Storage()

    def should_store(self) -> bool:
        return self.storage_type == JobStorage.s3

    def store_bytes(self, bytes: bytes, name: str, extension: str) -> str:
        filename = self.storage.sanitize_filename(name, extension)
        self.storage.upload_from_bytes(self.servicer.type, filename, bytes)
        return self.storage.create_signed_url(self.servicer.type, filename)

    def set_keep_alive(self, keep_alive: int):
        self.keep_alive = keep_alive

    def decrease_keep_alive(self):
        self.keep_alive -= 1
        self.logger.info(f"Keep alive decreased to {self.keep_alive}")

    def get_keep_alive(self) -> int:
        return self.keep_alive