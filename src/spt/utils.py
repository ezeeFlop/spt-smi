import os
import json


def load_json(file, dir="./"):
    jsonFile = os.path.join(dir, f"{file}.json")
    if os.path.exists(jsonFile):
        with open(jsonFile, "r") as f:
            return json.load(f)
    else:
        return None