from concurrent import futures
import grpc
from google.protobuf.wrappers_pb2 import FloatValue
import imagegeneration_pb2
import imagegeneration_pb2_grpc
from config import IMAGEGENERATION_SERVICE_PORT, IMAGEGENERATION_SERVICE_HOST

from rich.logging import RichHandler
from rich.console import Console
from spt.models import ServiceStatus

from spt.services.image_generation.models import DiffusionModels
from spt.models import EngineResult
import logging

console = Console()

logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(
        console=console, rich_tracebacks=True, show_time=False)]
)

logger = logging.getLogger(__name__)

class TextToImageServicer(imagegeneration_pb2_grpc.ImageGenerationServicer) :
    def GenerateImage(self, request: imagegeneration_pb2.ImageGenerationRequest, context) -> imagegeneration_pb2.ImageGenerationResponse:
        logger.info(f"Received request {request}")

        images = DiffusionModels.generate_images(request)

        return imagegeneration_pb2.ImageGenerationResponse(images=images, finishReason=EngineResult.success)

    def Status(self, request, context) -> imagegeneration_pb2.StatusResponse:
        logger.info(f"Received status request {request}")
        logger.info(f"Memory usage: {DiffusionModels.memory_usage()}")
        memory_usage = FloatValue(value=DiffusionModels.memory_usage())
        return imagegeneration_pb2.StatusResponse(status=ServiceStatus.idle, memory_usage=memory_usage, message="...")

        #return imagegeneration_pb2.StatusResponse(status=ServiceStatus.idle, memory_usage=DiffusionModels.memory_usage(), message="...")

def serve(max_workers=10, host="localhost", port=50051):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    imagegeneration_pb2_grpc.add_ImageGenerationServicer_to_server(
        TextToImageServicer(), server)
    server.add_insecure_port(f"{host}:{port}")
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    logger.info("Starting ImageGeneration service on host %s port %s", IMAGEGENERATION_SERVICE_HOST, IMAGEGENERATION_SERVICE_PORT)    
    serve(max_workers=10, host=IMAGEGENERATION_SERVICE_HOST,
          port=IMAGEGENERATION_SERVICE_PORT)
