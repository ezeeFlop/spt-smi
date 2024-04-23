import asyncio
import logging
from typing import List
from your_module import JobsTypes, Job  # Assure-toi d'importer tes modules correctement

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AsyncJobReceiver:
    def __init__(self, job_type: JobsTypes):
        self.job_type = job_type
        self.jobs_queue = asyncio.Queue()
        self.running = False

    async def receive_job(self):
        """Simule la réception d'un job. À remplacer par la logique de réception réelle."""
        while self.running:
            await asyncio.sleep(1)  # Simule un délai de réception
            simulated_job = Job(type=self.job_type)  # Crée un job simulé
            await self.jobs_queue.put(simulated_job)
            logger.info(f"Job reçu et ajouté à la queue: {simulated_job.id}")

    async def process_job(self):
        """Traite les jobs de la queue."""
        while self.running:
            job = await self.jobs_queue.get()
            # Traite le job ici. Par exemple:
            logger.info(f"Traitement du job: {job.id}")
            await asyncio.sleep(1)  # Simule un temps de traitement
            self.jobs_queue.task_done()

    async def start(self):
        self.running = True
        receiver_task = asyncio.create_task(self.receive_job())
        processor_task = asyncio.create_task(self.process_job())
        await asyncio.gather(receiver_task, processor_task)

    def stop(self):
        self.running = False

async def main():
    jobs_types = [JobsTypes.image_generation, JobsTypes.llm_generation]
    receivers = [AsyncJobReceiver(job_type) for job_type in jobs_types]

    await asyncio.gather(*(receiver.start() for receiver in receivers))

if __name__ == "__main__":
    asyncio.run(main())
