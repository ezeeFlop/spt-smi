import grpc
import imagegeneration_pb2
import imagegeneration_pb2_grpc
from spt.jobs import Job
from spt.models import TextToImageRequest
import logging

logger = logging.getLogger(__name__)


class TextToImageClient:
    def __init__(self, host, port) -> None:
        self.channel = grpc.insecure_channel(f'{host}:{port}')
        self.stub = imagegeneration_pb2_grpc.ImageGenerationStub(self.channel)

    def status(self) -> imagegeneration_pb2.StatusResponse:
        logger.info("Getting status")
        return self.stub.Status(imagegeneration_pb2.EmptyRequest())

    def generate_image(self, job: Job) -> imagegeneration_pb2.ImageGenerationResponse:
        logger.info(f"Generate Image with {job.payload}")

        return self.stub.GenerateImage(imagegeneration_pb2.ImageGenerationRequest(
            text_prompts=job.payload["text_prompts"], 
            height=job.payload["height"], 
            width=job.payload["width"], 
            steps=job.payload["steps"], 
            samples=job.payload["samples"], 
            cfg_scale=job.payload["cfg_scale"], 
            clip_guidance_preset=job.payload["clip_guidance_preset"], 
            sampler=job.payload["sampler"], 
            seed=job.payload["seed"], 
            style_preset=job.payload["style_preset"],
            model_id = job.model_id))
