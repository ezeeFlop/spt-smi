# SMI Client

Python client for the Sponge Theory SMI (Scalable Models Inferences) API.

## Installation

```bash
pip install smi-client
```

## Features

- Text to Image Generation
- Text to Video Generation
- Text to Text (Chat) Generation
- Image to Text Generation
- Text to Embeddings
- Text to Speech
- Speech to Text
- Worker Management
- GPU Information

## Quick Start

```python
import asyncio
from smi_client import SMIClient
from smi_client.models.image import TextToImageRequest

async def main():
    # Initialize the client
    client = SMIClient("your-api-key", base_url="http://your-api-url")
    
    try:
        # List available workers
        workers = await client.list_workers()
        print("Available workers:", workers)

        # Example: Text to Image
        request = TextToImageRequest(
            worker_id="realisticVision",
            prompt="A beautiful sunset over mountains",
            height=512,
            width=768,
            samples=1
        )
        response = await client.text_to_image(request, accept_format="image/png")
        print("Generated image:", response)

    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(main())
```

## Advanced Usage

### Async Mode

For long-running operations, you can use async mode:

```python
response = await client.text_to_image(request, async_mode=True)
job_id = response.id

# Later, check the job status
result = await client.get_job_status(job_id, "text-to-image")
```

### Storage Options

Choose between local and S3 storage:

```python
from smi_client.models.jobs import JobStorage

response = await client.text_to_image(request, storage=JobStorage.s3)
```

### Priority Control

Set job priority:

```python
from smi_client.models.jobs import JobPriority

response = await client.text_to_image(request, priority=JobPriority.high)
```

## License

MIT License 