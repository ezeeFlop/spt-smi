from spt.services.service import Worker, Service
from spt.utils import create_temp_file, remove_temp_file, get_available_device

class Template(Worker):
    def __init__(self, name: str, service: Service, model: str, logger):
        super().__init__(name=name, service=service, model=model, logger=logger)
        self.my_model = None

    async def work(self, request: BaseModel) -> BaseModel:
        await super().work(request)

        # load self.model ...

        return None

    def cleanup(self):
        super().cleanup()
        if self.my_model is not None:
            self.logger.info(f"Closing model {self.my_model}")
            del self.my_model
