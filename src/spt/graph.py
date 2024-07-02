import json
from typing import Dict, Type, Any
from spt.models.graph import SequentialGraph, RequestResponseLink
from pydantic import BaseModel
import spt.models as mdls

# Fonction pour exécuter une étape du graphe séquentiel
def execute_step(step: mdls.graph.RequestResponseLink, models: Dict[str, Type[BaseModel]], data: Dict[str, Any]) -> Any:
    source_data = data[step.source_model]
    target_model = models[step.target_model]
    target_data = {}

    for link in step.links:
        source_value = eval(f"source_data.{link.source}")
        exec(f"target_data['{link.target}'] = source_value")

    return target_model(**target_data)

# Fonction principale pour lire la configuration et exécuter le graphe
def execute_graph(config_path: str, models: Dict[str, Type[BaseModel]], initial_data: Dict[str, Any]):
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    graph = SequentialGraph(**config)

    data = initial_data
    for step in graph.steps:
        result = execute_step(step, models, data)
        data[step.target_model] = result

    return data

# Exemple d'utilisation
models = {
    "GenerateRequest": GenerateRequest,
    "GenerateResponse": GenerateResponse,
    "ChatRequest": ChatRequest,
    "ChatResponse": ChatResponse,
    "EmbeddingsRequest": EmbeddingsRequest,
    "EmbeddingsResponse": EmbeddingsResponse
}

initial_data = {
    "GenerateResponse": GenerateResponse(model="text-davinci-003", created_at="2022-08-01T00:00:00Z", response="Hello, World!", done=True)
}

result_data = execute_graph('config.json', models, initial_data)
print(result_data)