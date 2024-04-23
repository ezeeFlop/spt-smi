import grpc
import logging
import generic_pb2
import generic_pb2_grpc
from spt.jobs import Job
import json
# Configuration du logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GenericClient:
    def __init__(self, host, port) -> None:
        """
        Initialise le client gRPC.
        :param host: L'hôte du serveur gRPC.
        :param port: Le port du serveur gRPC.
        """
        # Créer un canal gRPC en utilisant l'hôte et le port fournis.
        self.channel = grpc.insecure_channel(f'{host}:{port}')
        # Créer un stub (proxy) pour communiquer avec le serveur gRPC.
        self.stub = generic_pb2_grpc.GenericServiceStub(self.channel)

    def process_data(self, job: Job) -> generic_pb2.GenericResponse:
        """
        Envoie des données JSON au service gRPC et reçoit une réponse.
        :param json_payload: Le payload JSON sérialisé en bytes.
        :return: Une réponse du serveur gRPC contenant un payload JSON.
        """

        string_payload = json.dumps(job.payload)
        # Encodage du payload JSON en bytes
        json_payload = string_payload.encode('utf-8')
        # Log l'action d'envoi de la demande de traitement.
        logger.info(
            f"Envoi d'une demande de traitement avec payload: {json_payload}")
        # Créer une requête gRPC avec le payload JSON et envoyer la requête.
        request = generic_pb2.GenericRequest(
            json_payload=json_payload, 
            remote_class=job.remote_class, 
            remote_method=job.remote_method, 
            request_model_class=job.request_model_class, 
            response_model_class=job.response_model_class)
        
        response = self.stub.ProcessData(request)
        # Log la réception de la réponse.
        logger.info(f"Réponse reçue avec payload: {response.json_payload}")
        return response
