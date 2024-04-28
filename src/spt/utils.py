import os
import json
import tempfile
import logging
from config import TEMP_PATH

logger = logging.getLogger(__name__)


def load_json(file, dir="./"):
    jsonFile = os.path.join(dir, f"{file}.json")
    if os.path.exists(jsonFile):
        with open(jsonFile, "r") as f:
            return json.load(f)
    else:
        logger.error(f"{jsonFile} does not exist")
        return None
    
def create_temp_file(content: bytes) -> str:
    temp_file = tempfile.NamedTemporaryFile(delete=False, dir=TEMP_PATH)
    temp_file.write(content)
    temp_file.close()
    return temp_file.name

def remove_temp_file(file_path: str) -> None:
    if os.path.exists(file_path):
        os.remove(file_path)
