import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio
from spt.models.workers import WorkerStreamManageRequest, WorkerStreamManageResponse, WorkerStatus
from spt.services.service import Service, GenericServiceServicer


class TestService(unittest.TestCase):
    def setUp(self):
        self.servicer = GenericServiceServicer(type=MagicMock())
        self.service = Service(self.servicer)

    @patch('spt.services.service.Worker', new_callable=AsyncMock)
    async def test_get_worker(self, MockWorker):
        worker_instance = MockWorker.return_value
        worker_instance.get_status.return_value = WorkerStatus.idle
        worker = await self.service.get_worker('test_model')
        self.assertEqual(worker, worker_instance)

    @patch('spt.services.service.Worker', new_callable=AsyncMock)
    async def test_work(self, MockWorker):
        worker_instance = MockWorker.return_value
        worker_instance.work = AsyncMock(return_value=MagicMock())
        request = MagicMock(model='test_model')
        result = await self.service.work(request)
        worker_instance.work.assert_awaited_once_with(request)
        worker_instance.stop.assert_called_once()
        self.assertEqual(result, worker_instance.work.return_value)

    def test_find_free_port(self):
        port = self.service.find_free_port()
        self.assertIsInstance(port, int)
        self.assertGreater(port, 0)

    @patch('asyncio.create_task', new_callable=MagicMock)
    @patch('spt.services.service.Worker', new_callable=AsyncMock)
    async def test_stream(self, MockWorker, mock_create_task):
        worker_instance = MockWorker.return_value
        worker_instance.start_stream = AsyncMock()
        request = WorkerStreamManageRequest(model='test_model')
        response = await self.service.stream(request)
        worker_instance.start_stream.assert_awaited_once()
        mock_create_task.assert_called_once()
        self.assertIsInstance(response, WorkerStreamManageResponse)

    def run_async(self, coro):
        asyncio.run(coro)

    def test_async_methods(self):
        self.run_async(self.test_get_worker())
        self.run_async(self.test_work())
        self.run_async(self.test_stream())


if __name__ == '__main__':
    unittest.main()
