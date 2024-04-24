import logging
from spt.services.generic.service import GenericServiceServicer

logger = logging.getLogger(__name__)

class Service:
    def __init__(self, servicer: GenericServiceServicer) -> None:
        self.servicer = servicer

    def chunked_request(self, request):
        logger.info(f"Chunked request: {request}")