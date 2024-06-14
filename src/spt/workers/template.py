from spt.services.service import Worker, Service
from spt.utils import create_temp_file, remove_temp_file, get_available_device
from spt.models.workers import WorkerBaseRequest
from pydantic import BaseModel
from typing import Union, Dict, Any

class Template(Worker):
    def __init__(self, name: str, service: Service, model: str, logger):
        super().__init__(name=name, service=service, model=model, logger=logger)
        self.my_model = None

    async def work(self, request: WorkerBaseRequest) -> BaseModel:
        await super().work(request)

        # load self.model ...

        return None

    async def stream(self, data: Union[bytes | str | Dict[str, Any]]) -> Union[bytes | str | Dict[str, Any]]:
        # do something with data

        return data

    def cleanup(self):
        super().cleanup()
        if self.my_model is not None:
            self.logger.info(f"Closing model {self.my_model}")
            del self.my_model
