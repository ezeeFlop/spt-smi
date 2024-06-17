import torch
from pynvml import nvmlInit, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex, nvmlDeviceGetName, nvmlDeviceGetMemoryInfo, nvmlDeviceGetUtilizationRates, NVMLError, nvmlShutdown, NVMLError
from spt.models.remotecalls import GPUsInfo, GPUInfo
import os
import json
import tempfile
import logging
from config import TEMP_PATH, STREAMING_PORTS_RANGE
import socket
logger = logging.getLogger(__name__)


def find_free_port() -> int:
    """
    Finds a free port within a specified range.

    Args:
        start_port (int): The starting port of the range.
        end_port (int): The ending port of the range.

    Returns:
        int: A free port within the specified range.

    Raises:
        RuntimeError: If no free port is found within the specified range.
    """
    (start_port, end_port) = STREAMING_PORTS_RANGE.split('-')
    start_port = int(start_port)
    end_port = int(end_port)
    for port in range(start_port, end_port + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('', port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No free port found in range {start_port}-{end_port}")


def load_json(file, dir="./"):
    jsonFile = os.path.join(dir, f"{file}.json")
    if os.path.exists(jsonFile):
        with open(jsonFile, "r") as f:
            return json.load(f)
    else:
        logger.error(f"{jsonFile} does not exist")
        return None
    
def create_temp_file(content: bytes) -> str:
    temp_file = tempfile.NamedTemporaryFile(delete=False, dir=TEMP_PATH)
    temp_file.write(content)
    temp_file.close()
    return temp_file.name

def remove_temp_file(file_path: str) -> None:
    if os.path.exists(file_path):
        os.remove(file_path)

def get_available_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')

    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def gpu_infos(display: bool = False) -> GPUsInfo:
    try:
        nvmlInit()
        devices = []
        for i in range(nvmlDeviceGetCount()):
            handle = nvmlDeviceGetHandleByIndex(i)
            info = GPUInfo(
                name=nvmlDeviceGetName(handle),
                memory_total_gb=nvmlDeviceGetMemoryInfo(
                    handle).total / (1024 ** 3),
                memory_used_gb=nvmlDeviceGetMemoryInfo(
                    handle).used / (1024 ** 3),
                memory_free_gb=nvmlDeviceGetMemoryInfo(
                    handle).free / (1024 ** 3),
                utilization_gpu_percent=nvmlDeviceGetUtilizationRates(
                    handle).gpu,
                utilization_memory_percent=nvmlDeviceGetUtilizationRates(
                    handle).memory
            )
            devices.append(info)
        if display:
            for device in devices:
                logger.info(device.__dict__)
        nvmlShutdown()

        return GPUsInfo(gpus=devices)
    except NVMLError as e:
        if display:
            logger.error(e)
        return GPUsInfo(error=str(e), gpus=[])
