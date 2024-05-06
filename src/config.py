
import os


csf = os.path.abspath(__file__)
csd = os.path.dirname(csf)

ROOT_DOMAIN = os.environ['ROOT_DOMAIN'] if os.environ.get(
    'ROOT_DOMAIN') else "http://localhost=9999"

CONFIG_PATH = os.environ['CONFIG_PATH'] if os.environ.get('CONFIG_PATH') else f"{csd}/../configs"

RABBITMQ_HOST = os.environ['RABBITMQ_HOST'] if os.environ.get(
    'RABBITMQ_HOST') else "localhost"
RABBITMQ_USER = os.environ['RABBITMQ_DEFAULT_USER'] if os.environ.get(
    'RABBITMQ_DEFAULT_USER') else "root"
RABBITMQ_PASSWORD = os.environ['RABBITMQ_DEFAULT_PASS'] if os.environ.get(
    'RABBITMQ_DEFAULT_PASS') else "jskdljflskdjflkjsqkjflkjqsldf564654"

REDIS_HOST = os.environ['REDIS_HOST'] if os.environ.get('REDIS_HOST') else "localhost"
REDIS_PORT = os.environ['REDIS_PORT'] if os.environ.get(
    'REDIS_PORT') else 6379
QUEUE_RETRY_DELAY = 5 # in seconds
# Services ports

IMAGE_GENERATION = os.environ['IMAGE_GENERATION'] if os.environ.get(
    'IMAGE_GENERATION') else "localhost:55001"
LLM_GENERATION = os.environ['LLM_GENERATION'] if os.environ.get(
    'LLM_GENERATION') else "localhost:55002"
AUDIO_GENERATION = os.environ['AUDIO_GENERATION'] if os.environ.get(
    'AUDIO_GENERATION') else "localhost:55003"
IMAGE_PROCESSING = os.environ['IMAGE_PROCESSING'] if os.environ.get(
    'IMAGE_PROCESSING') else "localhost:55004"
VIDEO_GENERATION = os.environ['VIDEO_GENERATION'] if os.environ.get(
    'VIDEO_GENERATION') else "localhost:55005"

# OLLAMA Url

OLLAMA_URL = os.environ['OLLAMA_URL'] if os.environ.get(
    'OLLAMA_URL') else "http://localhost:11434"

# Pooling delay in seconds for sync API requests
POLLING_TIMEOUT = 500
SERVICE_KEEP_ALIVE = 5 # in minutes

# Storage Location
TEMP_PATH = os.environ['TEMP_PATH'] if os.environ.get(
    'TEMP_PATH') else f"{csd}/../temp"

# Minio configuration
MINIO_ROOT_USER = os.environ['MINIO_ROOT_USER'] if os.environ.get(
    'MINIO_ROOT_USER') else 'minioadmin'
MINIO_ROOT_PASSWORD = os.environ['MINIO_ROOT_PASSWORD'] if os.environ.get(
    'MINIO_ROOT_PASSWORD') else 'minioadmin'
MINIO_SERVER_ENDPOINT = os.environ['MINIO_SERVER_ENDPOINT'] if os.environ.get(
    'MINIO_SERVER_ENDPOINT') else "localhost:9000"
MINIO_SERVER_URL = os.environ['MINIO_SERVER_URL'] if os.environ.get(
    'MINIO_SERVER_URL') else "http://localhost:9000"
MINIO_SECURE_URL = os.environ['MINIO_SECURE_URL'] if os.environ.get(
    'MINIO_SECURE_URL') else False
MINIO_FILE_DURATION = os.environ['MINIO_FILE_DURATION'] if os.environ.get(
    'MINIO_FILE_DURATION') else 5
