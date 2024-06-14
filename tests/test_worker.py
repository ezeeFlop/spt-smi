import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import time
import asyncio
import zmq.asyncio
from spt.models.workers import WorkerStatus
# Remplacez `src.your_module` par le chemin correct vers votre module
from spt.services.service import Worker


class TestWorker(unittest.TestCase):
    def setUp(self):
        self.context = zmq.asyncio.Context.instance()
        self.worker = Worker(name="TestWorker", logger=MagicMock())
        self.worker.context = self.context

    def tearDown(self):
        self.context.term()

    def test_set_service(self):
        service = MagicMock()
        self.worker.set_service(service)
        self.assertEqual(self.worker.service, service)

    def test_set_status(self):
        self.worker.set_status(WorkerStatus.working)
        self.assertEqual(self.worker.get_status(), WorkerStatus.working)

    def test_get_duration(self):
        self.worker.start_time = time.time() - 10
        duration = self.worker.get_duration()
        self.assertAlmostEqual(duration, 10, delta=1)

    @patch('zmq.asyncio.Poller')
    @patch('zmq.asyncio.Socket.recv', new_callable=AsyncMock)
    @patch('zmq.asyncio.Socket.send', new_callable=AsyncMock)
    async def test_start_stream(self, mock_send, mock_recv, MockPoller):
        mock_poller = MockPoller.return_value
        mock_poller.poll = AsyncMock(
            return_value={self.worker.context.socket(zmq.PULL): zmq.POLLIN})

        await self.worker.start_stream('127.0.0.1', 5555, 5556, timeout=1)

        mock_recv.assert_awaited()
        mock_send.assert_awaited()
        self.assertEqual(self.worker.status, WorkerStatus.streaming)

    def test_stop(self):
        async def async_test():
            self.worker.stream_task = asyncio.create_task(asyncio.sleep(1))
            self.worker.stop()
            self.assertEqual(self.worker.get_status(), WorkerStatus.idle)
            self.assertTrue(self.worker.stop_event.is_set())

        asyncio.run(async_test())


if __name__ == '__main__':
    unittest.main()
