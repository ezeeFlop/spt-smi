
import os


ROOT_DOMAIN = os.environ['ROOT_DOMAIN'] if os.environ.get(
    'ROOT_DOMAIN') else "http://localhost=8999"

CONFIG_PATH = os.environ['CONFIG_PATH'] if os.environ.get('CONFIR_PATH') else "./configs"

RABBIT_HOST = os.environ['RABBIT_HOST'] if os.environ.get('RABBIT_HOST') else "localhost"
RABBIT_USER = os.environ['RABBIT_USER'] if os.environ.get(
    'RABBIT_USER') else "root"
RABBIT_PASSWORD = os.environ['RABBIT_PASSWORD'] if os.environ.get(
    'RABBIT_PASSWORD') else "jskdljflskdjflkjsqkjflkjqsldf564654"

QUEUE_NAME = os.environ['QUEUE_NAME'] if os.environ.get('QUEUE_NAME') else "ia_jobs_queue"

REDIS_HOST = os.environ['REDIS_HOST'] if os.environ.get('REDIS_HOST') else "localhost"

# Services ports

IMAGEGENERATION_SERVICE_PORT = 55001

POLLING_TIMEOUT = 30