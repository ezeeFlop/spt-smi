import os
import json


import logging

logger = logging.getLogger(__name__)


def load_json(file, dir="./"):
    jsonFile = os.path.join(dir, f"{file}.json")
    if os.path.exists(jsonFile):
        with open(jsonFile, "r") as f:
            return json.load(f)
    else:
        logger.error(f"{jsonFile} does not exist")
        return None