from pydantic import BaseModel, Field, validator
from enum import Enum

class JobsTypes(str, Enum):
    image_generation = "IMAGE_GENERATION",
    llm_generation = "LLM_GENERATION",
    audio_generation = "AUDIO_GENERATION",
    video_generation = "VIDEO_GENERATION",
    unknown = "UNKNOWN"

class JobStatuses(str, Enum):
    pending = "PENDING",
    queued = "QUEUED",
    in_progress = "IN_PROGRESS",
    completed = "COMPLETED",
    failed = "FAILED",
    unknown = "UNKNOWN"

class JobPriority(str, Enum):
    low = "LOW",
    normal = "NORMAL",
    high = "HIGH"

class JobStorage(str, Enum):
    local = "LOCAL",
    s3 = "S3"

class JobResponse(BaseModel):
    id: str = Field(..., example="b7b7c5a5-98b0-4a07-af27-93bfcfa38246",
                    description="Unique identifier for the job")
    type: JobsTypes = Field(..., example="image_generation",
                            description="Type of job")
    status: JobStatuses = Field(..., example="completed",
                                description="Status of the job")
    message: str = Field(..., example="Job completed successfully",
                         description="Message of the job")
