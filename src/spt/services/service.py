import logging
from spt.services.generic.service import GenericServiceServicer
from spt.storage import Storage
from spt.models.jobs import JobStorage
logger = logging.getLogger(__name__)

class Service:
    def __init__(self, servicer: GenericServiceServicer) -> None:
        self.servicer = servicer
        self.storage_type = None
        self.keep_alive = 15
        self.storage:Storage = None

    def cleanup(self):
        logger.info(f"Service Cleanup {self.keep_alive}")

    def should_cleanup(self):
        self.keep_alive = self.keep_alive - 1
        logger.info(f"Service Should Cleanup {self.keep_alive}")

        if self.keep_alive <= 0:
            return True
        return False

    def chunked_request(self, request):
        logger.info(f"Chunked request: {request}")

    def set_storage(self, storage:str):
        self.storage_type = storage
        if self.storage_type == JobStorage.s3 and self.storage is None:
            self.storage = Storage()
    
    def should_store(self)->bool:
        if self.storage_type == JobStorage.s3:
            return True
        return False

    def store_bytes(self, bytes: bytes, name: str, extension: str):
        filename = self.storage.sanitize_filename(name, extension)
        self.storage.upload_from_bytes(self.servicer.type, filename, bytes)
        return self.storage.create_signed_url(self.servicer.type, filename)

    def set_keep_alive(self, keep_alive:int):
        self.keep_alive = keep_alive
    
    def get_keep_alive(self):
        return self.keep_alive