
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