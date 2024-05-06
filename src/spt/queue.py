from typing import Callable, Any, Optional
from pika.exceptions import AMQPConnectionError
from pydantic import BaseModel, validator
import msgpack
from typing import Optional, Dict
from enum import Enum
import time
import ssl
import functools
import asyncio
import pika
from config import RABBITMQ_HOST, RABBITMQ_PASSWORD, RABBITMQ_USER, QUEUE_RETRY_DELAY
import logging

logger = logging.getLogger(__name__)

def sync(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return asyncio.get_event_loop().run_until_complete(f(*args, **kwargs))

    return wrapper

class Priority(Enum):
    LOW = 1
    NORMAL = 5
    HIGH = 10

class Headers(BaseModel):
    job_id: str
    job_type: str
    job_model_id: str
    job_remote_class: str
    job_remote_method: str
    job_request_model_class: str
    job_response_model_class: str
    job_storage: str
    job_keep_alive: int

class RabbitMQConfig(BaseModel):
    host: str
    port: int
    username: str
    password: str
    protocol: str

class QueueClient:
    def __init__(self):
        self.username = RABBITMQ_USER
        self.password = RABBITMQ_PASSWORD
        self.host = RABBITMQ_HOST
        self.port = 5672
        self.protocol = ""

        self._init_connection_parameters()
        self._connect()

    def _connect(self):
        tries = 0
        while True:
            try:
                self.connection = pika.BlockingConnection(self.parameters)
                self.channel = self.connection.channel()
                if self.connection.is_open:
                    break
            except (AMQPConnectionError, Exception) as e:
                time.sleep(5)
                tries += 1
                if tries == 20:
                    raise AMQPConnectionError(e)

    def _init_connection_parameters(self):
        self.credentials = pika.PlainCredentials(self.username, self.password)
        self.parameters = pika.ConnectionParameters(
            host=self.host,
            port=int(self.port),
            #virtual_host="/",
            credentials=self.credentials,
        )
        if self.protocol == "amqps":
            # SSL Context for TLS configuration of Amazon MQ for RabbitMQ
            ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
            ssl_context.set_ciphers("ECDHE+AESGCM:!ECDSA")
            self.parameters.ssl_options = pika.SSLOptions(context=ssl_context)

    def check_connection(self):
        """
        Checks if the connection is open and reconnects if it is closed.

        This function checks if the connection object is None or if it is closed. If either condition is true, it calls the `_connect` method to reestablish the connection.

        Parameters:
            self (QueueClient): The instance of the QueueClient class.

        Returns:
            None
        """
        if not self.connection or self.connection.is_closed:
            self._connect()

    def close(self):
        """
        Closes the channel and the connection.
        """
        self.channel.close()
        self.connection.close()

    def declare_queue(
        self, queue_name, exclusive: bool = False, max_priority: int = 10
    ):
        """
        Declares a queue with the specified name.

        Args:
            queue_name (str): The name of the queue.
            exclusive (bool, optional): Whether the queue should be exclusive. Defaults to False.
            max_priority (int, optional): The maximum priority for the queue. Defaults to 10.

        Returns:
            None
        """
        self.check_connection()
        logger.debug(f"Trying to declare queue({queue_name})...")
        self.channel.queue_declare(
            queue=queue_name,
            exclusive=exclusive,
            durable=True,
            arguments={"x-max-priority": max_priority},
        )

    def declare_exchange(self, exchange_name: str, exchange_type: str = "direct"):
        """
        Declares an exchange with the specified name and type.

        Args:
            exchange_name (str): The name of the exchange.
            exchange_type (str, optional): The type of the exchange. Defaults to "direct".

        Returns:
            None
        """
        self.check_connection()
        self.channel.exchange_declare(
            exchange=exchange_name, exchange_type=exchange_type
        )

    def bind_queue(self, exchange_name: str, queue_name: str, routing_key: str):
        """
        Binds a queue to an exchange with the specified routing key.

        Args:
            exchange_name (str): The name of the exchange to bind the queue to.
            queue_name (str): The name of the queue to bind.
            routing_key (str): The routing key to use for the binding.

        Returns:
            None
        """
        self.check_connection()
        self.channel.queue_bind(
            exchange=exchange_name, queue=queue_name, routing_key=routing_key
        )

    def unbind_queue(self, exchange_name: str, queue_name: str, routing_key: str):
        """
        Unbinds a queue from an exchange with the specified routing key.

        Args:
            exchange_name (str): The name of the exchange to unbind the queue from.
            queue_name (str): The name of the queue to unbind.
            routing_key (str): The routing key used for the binding.

        Returns:
            None
        """
        self.channel.queue_unbind(
            queue=queue_name, exchange=exchange_name, routing_key=routing_key
        )

class QueueMessageSender(QueueClient):
    def encode_message(self, body: Dict, encoding_type: str = "bytes"):
        """
        Encodes a dictionary into a message using the specified encoding type.

        Args:
            body (Dict): The dictionary to be encoded.
            encoding_type (str, optional): The encoding type to use. Defaults to "bytes".

        Returns:
            bytes: The encoded message if the encoding type is "bytes".

        Raises:
            NotImplementedError: If the encoding type is not "bytes".
        """
        if encoding_type == "bytes":
            return msgpack.packb(body)
        else:
            raise NotImplementedError

    def send_message(
        self,
        exchange_name: str,
        routing_key: str,
        body: Dict,
        priority: Priority,
        headers: Optional[Headers],
    ):
        """
        Sends a message to the specified exchange with the given routing key, body, priority, and headers.

        Args:
            exchange_name (str): The name of the exchange to send the message to.
            routing_key (str): The routing key for the message.
            body (Dict): The body of the message as a dictionary.
            priority (Priority): The priority of the message.
            headers (Optional[Headers]): The headers for the message.

        Returns:
            None

        Raises:
            None
        """
        body = self.encode_message(body=body)
        self.channel.basic_publish(
            exchange=exchange_name,
            routing_key=routing_key,
            body=body,
            properties=pika.BasicProperties(
                delivery_mode=pika.spec.PERSISTENT_DELIVERY_MODE,
                priority=priority.value,
                headers=headers.model_dump(),
            ),
        )
        logger.debug(
            f"Sent message. Exchange: {exchange_name}, Routing Key: {routing_key}, Body: {body[:128]}"
        )

class QueueMessageReceiver(QueueClient):
    def __init__(self):
        super().__init__()
        self.channel_tag = None

    def decode_message(self, body):
        """
        Decode the given message body.

        Args:
            body (bytes): The message body to decode.

        Returns:
            The decoded message.

        Raises:
            NotImplementedError: If the message body is not of type bytes.
        """
        if type(body) == bytes:
            return msgpack.unpackb(body)
        else:
            raise NotImplementedError

    def get_message(self, queue_name: str, auto_ack: bool = False):
        """
        Retrieves a message from the specified queue.

        Args:
            queue_name (str): The name of the queue to retrieve the message from.
            auto_ack (bool, optional): Whether to automatically acknowledge the message. Defaults to False.

        Returns:
            tuple or None: A tuple containing the method frame, header frame, and body of the message if a message is found. 
                           None if no message is found.
        """
        method_frame, header_frame, body = self.channel.basic_get(
            queue=queue_name, auto_ack=auto_ack
        )
        if method_frame:
            logger.debug(f"[MESSAGE] -------> {method_frame}, {header_frame}, {body}")
            return method_frame, header_frame, body
        else:
            logger.debug("No message returned")
            return None

    def consume_and_check_messages(self, queue: str, process_callback: Callable[[pika.channel.Channel, pika.spec.Basic.Deliver, pika.spec.BasicProperties, bytes], None], condition_callback: Optional[Callable[[dict], bool]] = None, auto_ack: bool = True) -> None:
        """
        Consumes messages from a queue by establishing a connection, setting up a channel tag using basic_consume,
        and starting message consumption. 

        Parameters:
            queue (str): The name of the queue to consume messages from.
            process_callback (Callable): The function to be called when a message is received.
            condition_callback (Optional[Callable]): The function to check if the message can be processed.
            auto_ack (bool): Whether to automatically acknowledge messages.

        Returns:
            None
        """
        self.check_connection()

        def callback(ch: pika.channel.Channel, method: pika.spec.Basic.Deliver, properties: pika.spec.BasicProperties, body: bytes) -> None:
                # Decode the message first
                #message: dict = self.decode_message(body)
                
                # Check the condition if condition_callback is provided
                if condition_callback and not condition_callback(ch, method, properties, body):
                    logger.info(f"Message does not meet the condition and will not be acknowledged, trying again in {QUEUE_RETRY_DELAY} seconds")
                    if not auto_ack:
                        # Delay before rejecting and requeuing
                        time.sleep(QUEUE_RETRY_DELAY)
                        ch.basic_reject(delivery_tag=method.delivery_tag, requeue=True)
                    return

                # Process the message if condition passes or no condition is provided
                process_callback(ch, method, properties, body)
                if not auto_ack:
                    ch.basic_ack(delivery_tag=method.delivery_tag)

        # Set up the consumer
        self.channel_tag = self.channel.basic_consume(
            queue=queue, on_message_callback=callback, auto_ack=auto_ack
        )
        logger.debug(" [*] Waiting for messages. To exit press CTRL+C")
        self.channel.start_consuming()


    def consume_messages(self, queue, callback):
        """
        Consumes messages from a queue by establishing a connection, setting up a channel tag using basic_consume,
        and starting message consumption. 

        Parameters:
            queue (str): The name of the queue to consume messages from.
            callback (function): The function to be called when a message is received.

        Returns:
            None
        """
        self.check_connection()
        self.channel_tag = self.channel.basic_consume(
            queue=queue, on_message_callback=callback, auto_ack=True
        )
        logger.debug(" [*] Waiting for messages. To exit press CTRL+C")
        self.channel.start_consuming()

    def cancel_consumer(self):
        """
        Cancels the consumer by canceling the current channel tag if it exists.

        This function checks if the channel tag is not None. If it is not None, it cancels the current channel tag using the `basic_cancel` method of the `channel` object. It then sets the channel tag to None. If the channel tag is None, it logs an error message saying "Do not cancel a non-existing job".

        Parameters:
            self (object): The instance of the class.

        Returns:
            None
        """
        if self.channel_tag is not None:
            self.channel.basic_cancel(self.channel_tag)
            self.channel_tag = None
        else:
            logger.error("Do not cancel a non-existing job")
