import logging
import json
from spt.models.remotecalls import GPUsInfo, GPUInfo
from pynvml import nvmlInit, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex, nvmlDeviceGetName, nvmlDeviceGetMemoryInfo, nvmlDeviceGetUtilizationRates, NVMLError, nvmlShutdown, NVMLError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('grpc-server')

def gpu_infos(display: bool = False) -> GPUsInfo:
    try:
        nvmlInit()
        devices = []
        for i in range(nvmlDeviceGetCount()):
            handle = nvmlDeviceGetHandleByIndex(i)
            info = GPUInfo(
                name=nvmlDeviceGetName(handle).decode('utf-8'),
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
