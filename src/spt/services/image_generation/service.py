from concurrent import futures
import grpc
import imagegeneration_pb2
import imagegeneration_pb2_grpc
from config import IMAGEGENERATION_SERVICE_PORT

from rich.logging import RichHandler
from rich.console import Console
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

class TextToImageServicer(imagegeneration_pb2_grpc.ImageGenerationServicer):
    def GenerateImage(self, request, context):
        logger.info(f"Received request {request}")


        # Ici, tu implémenteras la logique pour générer l'image basée sur la requête
        return imagegeneration_pb2.ImageGenerationResponse(base64="iVBORw0KGgoAAAANSUh...", finishReason="SUCCESS", seed=1050625087)

    def Status(self, request, context):
        logger.info(f"Received status request {request}")
        return imagegeneration_pb2.StatusResponse(status="SERVING")

def serve(max_workers=10, host="localhost", port=50051):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    imagegeneration_pb2_grpc.add_ImageGenerationServicer_to_server(
        TextToImageServicer(), server)
    server.add_insecure_port(f"{host}:{port}")
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    logger.info("Starting ImageGeneration service on port %s", IMAGEGENERATION_SERVICE_PORT)    
    serve(max_workers=10, host='localhost', port=IMAGEGENERATION_SERVICE_PORT)
