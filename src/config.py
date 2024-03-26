
import os


csf = os.path.abspath(__file__)
csd = os.path.dirname(csf)

ROOT_DOMAIN = os.environ['ROOT_DOMAIN'] if os.environ.get(
    'ROOT_DOMAIN') else "http://localhost=9999"

CONFIG_PATH = os.environ['CONFIG_PATH'] if os.environ.get('CONFIR_PATH') else f"{csd}/../configs"

RABBITMQ_HOST = os.environ['RABBITMQ_HOST'] if os.environ.get(
    'RABBITMQ_HOST') else "localhost"
RABBITMQ_USER = os.environ['RABBITMQ_DEFAULT_USER'] if os.environ.get(
    'RABBITMQ_DEFAULT_USER') else "root"
RABBITMQ_PASSWORD = os.environ['RABBITMQ_DEFAULT_PASS'] if os.environ.get(
    'RABBITMQ_DEFAULT_PASS') else "jskdljflskdjflkjsqkjflkjqsldf564654"

REDIS_HOST = os.environ['REDIS_HOST'] if os.environ.get('REDIS_HOST') else "localhost"

# Services ports

IMAGEGENERATION_SERVICE_PORT = os.environ['IMAGEGENERATION_SERVICE_PORT'] if os.environ.get(
    'IMAGEGENERATION_SERVICE_PORT') else 55001
IMAGEGENERATION_SERVICE_HOST = os.environ['IMAGEGENERATION_SERVICE_HOST'] if os.environ.get(
    'IMAGEGENERATION_SERVICE_HOST') else "localhost"


POLLING_TIMEOUT = 500