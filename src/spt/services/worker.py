from spt.models.workers import WorkerState, WorkerStreamType 
import asyncio
from pydantic import BaseModel, ValidationError
import time
import logging
import zmq
from zmq.asyncio import Context, Poller
from typing import Dict, Tuple, Any, Optional, Union
import traceback

class Worker:
    # type: ignore
    def __init__(self, name: str = "Worker", service: 'Service' = None, logger: logging.Logger = None, model: Optional[str] = None):
        self.name: str = name
        self.service: Optional['Service'] = service
        self.logger: logging.Logger = logger
        self.status: WorkerState = WorkerState.idle
        self.start_time: Optional[float] = None
        self.model: Optional[str] = model
        self.context: Context = Context()  # Utilisation de zmq.asyncio.Context
        self.stop_event: asyncio.Event = asyncio.Event()
        self.stream_task: Optional[asyncio.Task] = None

    def set_service(self, service: 'Service'):
        self.service = service

    async def work(self, data: BaseModel) -> BaseModel:
        self.status = WorkerState.working
        self.start_time = time.time()

    async def stream(self, data: Union[bytes | str | Dict[str, Any]]) -> Union[bytes | str | Dict[str, Any]]:
        return data

    async def start_stream(self, ip: str, input_port: int, output_port: int, timeout: int = 300, 
                           intype: WorkerStreamType = WorkerStreamType.text, 
                           outtype: WorkerStreamType = WorkerStreamType.text):
        
        self.status = WorkerState.streaming
        self.start_time = time.time()

        # Setting up ZeroMQ sockets

        sender = self.context.socket(zmq.PUSH)
        sender.bind(f"tcp://*:{input_port}")

        receiver = self.context.socket(zmq.PULL)
        receiver.connect(f"tcp://{ip}:{output_port}")

        self.logger.info(
            f"Worker {self.name} Listening data on port {input_port} and sending to port {output_port} on {ip} with timeout {timeout} and input type {intype} and output type {outtype}")

        poller = Poller()
        poller.register(receiver, zmq.POLLIN)

        input_funcs = {
            WorkerStreamType.text: receiver.recv_string,
            WorkerStreamType.bytes: receiver.recv,
            WorkerStreamType.json: receiver.recv_json
        }

        output_funcs = {
            WorkerStreamType.text: sender.send_string,
            WorkerStreamType.bytes: sender.send,
            WorkerStreamType.json: sender.send_json
        }

        input_func= input_funcs[intype]
        output_func = output_funcs[outtype]

        try:
            while not self.stop_event.is_set():
                socks = dict(await poller.poll(timeout * 1000))
                if receiver in socks and socks[receiver] == zmq.POLLIN:
                    
                    message = await input_func()
                    self.start_time = time.time()
                    print (message)
                    await output_func(await self.stream(message))
                else:
                    self.logger.info(
                        "No activity detected, stopping the stream.")
                    break
        except zmq.error.ZMQError as zmq_error:
            self.logger.error(f"ZeroMQ error in stream: {str(zmq_error)} {traceback.format_exc()}")
        except Exception as e:
            self.logger.error(
                f"Error in stream: {str(e)} {traceback.format_exc()}")
        finally:
            self.logger.info("Stream stopped")
            receiver.close()
            sender.close()
            self.context.term()

    def stop(self):
        self.status = WorkerState.idle
        self.stop_event.set()
        self.start_time = None
        if self.stream_task:
            self.stream_task.cancel()

    def cleanup(self):
        pass

    def set_status(self, status: WorkerState):
        self.status = status

    def get_status(self) -> WorkerState:
        return self.status

    def get_duration(self) -> float:
        if self.start_time:
            return time.time() - self.start_time
        return 0.0
