
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
from typing import Dict, Any, Optional, List
from spt.utils import find_free_port, get_ip

class Service:
    def __init__(self, servicer: GenericServiceServicer) -> None:
        self.servicer: GenericServiceServicer = servicer
        self.storage_type: Optional[str] = None
        self.keep_alive: int = 15
        self.storage: Optional[Storage] = None
        self.workers: Dict[str, Worker] = {}
        self.worker_configs: WorkerConfig = WorkerConfigs.get_configs().workers_configs
        self.instances: List[Worker] = []
        self.logger: logging.Logger = None
    
    def set_logger(self, logger: logging.Logger):
        self.logger = logger

    def check_workers(self):
        self.logger.info(f"  [-] Service Check {len(self.instances)}# Workers...")
        for worker in self.instances[:]:  # Iterate over a copy of the list
            if worker.get_status() == WorkerState.idle:
                self.logger.info(f"    [-] Garbaging worker idle {worker.id}")
                worker.stop()
                worker.cleanup()
                self.instances.remove(worker)
            else:
                self.logger.info(f"    [-] Worker {worker.id} with status {worker.get_status()} is still alive since {worker.get_duration()} seconds")
                if worker.get_duration() > self.keep_alive * 60:
                    self.logger.info(f"    [-] Stopping worker {worker.id} due to long run time")
                    worker.stop()
                    worker.cleanup()
                    self.instances.remove(worker)

    def cleanup(self):
        self.logger.info(f"  [-] Service Cleanup {self.keep_alive} minutes left")
        for worker in self.instances[:]:  # Iterate over a copy of the list
            if worker.get_status() == WorkerState.idle:
                self.logger.info(f"    [-] Cleaning up IDLE worker: {worker.id}")
                worker.cleanup()
                self.instances.remove(worker)
            elif worker.get_status() == WorkerState.working or worker.get_status() == WorkerState.streaming:
                self.logger.info(f"    [-] Stopping worker {worker.id} due to long run time")
                worker.stop()
                worker.cleanup()
                self.instances.remove(worker)

    def chunked_request(self, request: Any):
        self.logger.info(f"Chunked request: {request}")

    async def get_worker(self, worker_id: str) -> Worker:
        try:
            if (worker_id not in self.worker_configs):
                raise ValueError(
                    f'  [-] Worker class for model {worker_id} not found')

            worker_info:WorkerConfig = self.worker_configs[worker_id]

            # Check if an instance already exists and is IDLE
            for worker in self.instances:
                if worker.get_status() == WorkerState.idle and worker.id == worker_id:
                    self.logger.info(f"  [-] Reusing worker {worker_id}")
                    worker.stop()
                    return worker

            module_path, class_name = worker_info.worker.rsplit('.', 1)
            module = importlib.import_module(module_path)
            worker_class = getattr(module, class_name)
            self.logger.info(f"  [-] Creating new worker {worker_id}")
            worker = worker_class(id=worker_id,
                name=class_name, service=self, model=worker_info.model, logger=self.logger)
            self.instances.append(worker)

            return worker

        except Exception as e:
            self.logger.error(f"  [-] Error in get_worker: {e}")
            raise

    async def work(self, request: WorkerBaseRequest) -> BaseModel:
        worker = await self.get_worker(request.worker_id)
        result = await worker.work(request)
        worker.stop()
        return result

    async def stream(self, request: WorkerStreamManageRequest) -> WorkerStreamManageResponse:
            # Get the hostname of the current machine
            hostname = socket.gethostname()
            self.logger.info(f"Hostname: {hostname}")

            # Get the IP address associated with the hostname
            ip_address = get_ip()
            
            # Retrieve the worker instance associated with the requested model
            worker = await self.get_worker(request.worker_id)
            
            # Find a free port for input
            input_port = find_free_port()
            
            # get request port for output
            output_port = request.port

            # Start the stream in the background using asyncio
            worker.stream_task = asyncio.create_task(
                worker.start_stream(ip=request.ip_address, 
                                    input_port=input_port, 
                                    output_port=output_port,
                                    intype=request.intype,
                                    outtype=request.outtype,
                                    timeout=request.timeout)
                                    )
            
            # Return a response with the stream details
            return WorkerStreamManageResponse(hostname=hostname,
                                            state=WorkerState.streaming, 
                                            port=input_port, 
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
        self.logger.info(f"[-] Keep alive decreased to {self.keep_alive}")

    def get_keep_alive(self) -> int:
        return self.keep_alive