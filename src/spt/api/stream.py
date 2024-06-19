
from fastapi import WebSocket, WebSocketDisconnect
from spt.models.workers import WorkerStreamManageRequest, WorkerStreamManageResponse, WorkerStreamType
from config import POLLING_TIMEOUT, SERVICE_KEEP_ALIVE
import asyncio
import zmq
import traceback
from zmq.asyncio import Context, Poller

from spt.api.app import app, logger

async def stream(websocket: WebSocket, request: WorkerStreamManageRequest, response: WorkerStreamManageResponse):
    await websocket.accept()

    logger.info(f"Websocket {response}")

    context = Context()
    sender = context.socket(zmq.PUSH)
    sender.bind(f"tcp://*:{request.port}")

    receiver = context.socket(zmq.PULL)
    receiver.connect(f"tcp://{response.ip_address}:{response.port}")

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

    output_ws_funcs = {
        WorkerStreamType.text: websocket.send_text,
        WorkerStreamType.bytes: websocket.send_bytes,
        WorkerStreamType.json: websocket.send_json
    }
    input_ws_funcs = {
        WorkerStreamType.text: websocket.receive_text,
        WorkerStreamType.bytes: websocket.receive_bytes,
        WorkerStreamType.json: websocket.receive_json
    }

    input_func = input_funcs[request.outtype]
    output_func = output_funcs[request.intype]

    input_ws_func = input_ws_funcs[request.intype]
    output_ws_func = output_ws_funcs[request.outtype]

    stop_event: asyncio.Event = asyncio.Event()

    async def receive_from_ws():
        try:
            while not stop_event.is_set():
                try:
                    data = await asyncio.wait_for(input_ws_func(), timeout=request.timeout)
                    logger.debug(f"receive_from_ws Received data from WebSocket: {data}")
                    await output_func(data)
                except asyncio.TimeoutError:
                    logger.info("receive_from_ws WebSocket timed out due to inactivity")
                    break
        except WebSocketDisconnect as e:
            logger.info(f"receive_from_ws WebSocket disconnected: {e}")
        except asyncio.CancelledError:
            logger.info("Task receive_from_ws cancelled")
        except Exception as e:
            logger.error(f"receive_from_ws Error in receive_from_ws: {e} {traceback.format_exc()}")
        finally:
            try:
                logger.info("receive_from_ws Closing ZeroMQ sockets, and terminating context")
                stop_event.set()
                receiver.close()
                sender.close()
                context.term()
                if websocket.client_state != 3:  # 3 corresponds to WebSocketState.DISCONNECTED
                    logger.info(f"receive_from_ws Closing WebSocket {websocket.client_state}")
                    await websocket.close()
            except Exception as e:
                logger.error(f"receive_from_ws Error during cleanup in receive_from_ws: {e} {traceback.format_exc()}")

    async def send_to_ws():
        try:
            while not stop_event.is_set():
                socks = dict(await poller.poll(request.timeout * 1000))
                if receiver in socks and socks[receiver] == zmq.POLLIN:
                    message = await input_func()
                    logger.debug(f"send_to_ws Received message from ZeroMQ: {message}")
                    await output_ws_func(message)
                else:
                    logger.info(
                        "send_to_ws No activity detected, stopping the stream.")
                    break
        except WebSocketDisconnect as e:
            logger.info(f"send_to_ws WebSocket disconnected: {e}")
        except asyncio.CancelledError:
            logger.info("Task send_to_ws cancelled")
        except Exception as e:
            logger.error(f"send_to_ws Error in send_to_ws: {e} {traceback.format_exc()}")
        finally:
            try:
                logger.info("send_to_ws Closing ZeroMQ sockets, and terminating context")
                stop_event.set()
                receiver.close()
                sender.close()
                context.term()
                if websocket.client_state != 3:  # 3 corresponds to WebSocketState.DISCONNECTED
                    logger.info(f"send_to_ws Closing WebSocket {websocket.client_state}")
                    #await websocket.close()
            except Exception as e:
                logger.error(f"send_to_ws Error during cleanup in send_to_ws: {e} {traceback.format_exc()}")

    receive_task = asyncio.create_task(receive_from_ws(), name="receive_from_ws")
    send_task = asyncio.create_task(send_to_ws(), name="send_to_ws")

    try:
        await asyncio.gather(send_task, receive_task)
    except asyncio.CancelledError:
            logger.info("Main gather task cancelled")
    finally:
        receive_task.cancel()
        send_task.cancel()
        try:
            context.term()
        except Exception as e:
            logger.error(f"Error during final cleanup: {e} {traceback.format_exc()}")