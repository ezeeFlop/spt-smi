import re
from config import MINIO_ROOT_PASSWORD, MINIO_ROOT_USER, MINIO_SERVER_ENDPOINT, MINIO_SERVER_URL, MINIO_FILE_DURATION
from minio import Minio
from minio.error import S3Error
from rich.logging import RichHandler
from rich.console import Console
import logging
import base64
import io
from typing import Optional
from datetime import datetime, timedelta

console = Console()

logging.basicConfig(
    level="INFO",
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(
        console=console, rich_tracebacks=True, show_time=False)]
)

logger = logging.getLogger('Storage')

class Storage:
    def __init__(self) -> None:
        """
        Initializes a new instance of the Storage class.

        Returns:
            None
        """
        self.endpoint = MINIO_SERVER_ENDPOINT
        self.access_key = MINIO_ROOT_USER
        self.secret_key = MINIO_ROOT_PASSWORD
        self.secure = True if MINIO_SERVER_URL.startswith(
            "https") else False
        self.client = self.create_client()

    def create_client(self) -> Minio:
        return Minio(
            self.endpoint,
            access_key=self.access_key,
            secret_key=self.secret_key,
            secure=self.secure
        )

    def check_connection(self) -> bool:
        try:
            self.client.list_buckets()
            return True
        except S3Error as exc:
            logging.error(f"Connection failed: {exc}")
            return False

    def reset_connection(self) -> None:
        self.client = self.create_client()
        logging.info("MinIO client connection has been reset.")

    def upload_file(self, bucket_name: str, object_name: str, file_path: str) -> Optional[str]:
        """
        Uploads a file to the specified bucket with the given object name.

        Args:
            bucket_name (str): The name of the bucket to upload the file to.
            object_name (str): The name of the object in the bucket.
            file_path (str): The path to the file to be uploaded.

        Returns:
            Optional[str]: The ETag of the uploaded object if successful, None otherwise.

        Raises:
            S3Error: If there is an error uploading the file.
        """
        bucket_name = self.sanitize_bucket_name(bucket_name)

        if not self.check_connection():
            self.reset_connection()
        if not self.create_public_bucket(bucket_name):
            return None
        try:
            result = self.client.fput_object(
                bucket_name, object_name, file_path)
            logging.info(f"Uploaded {result.object_name} to {bucket_name}")
            return result.etag
        except S3Error as exc:
            logging.error(f"Error uploading file: {str(exc)}")
            return None

    def upload_from_base64(self, bucket_name: str, object_name: str, base64_data: str) -> Optional[str]:
        """
        Uploads data from a base64 string to a specified bucket with the given object name.

        Args:
            bucket_name (str): The name of the bucket to upload the data to.
            object_name (str): The name of the object in the bucket.
            base64_data (str): The base64 encoded data to be uploaded.

        Returns:
            Optional[str]: The ETag of the uploaded object if successful, None otherwise.

        Raises:
            S3Error: If there is an error uploading the data.
        """
        bucket_name = self.sanitize_bucket_name(bucket_name)

        if not self.check_connection():
            self.reset_connection()
        if not self.create_public_bucket(bucket_name):
            return None
        try:
            data = base64.b64decode(base64_data)
            data_stream = io.BytesIO(data)
            result = self.client.put_object(
                bucket_name, object_name, data_stream, len(data))
            logging.info(
                f"Uploaded {object_name} from base64 to {bucket_name}")
            return result.etag
        except S3Error as exc:
            logging.error(f"Error uploading from base64: {str(exc)}")
            return None

    def upload_from_bytes(self, bucket_name: str, object_name: str, byte_array: bytes) -> Optional[str]:
        """
        Uploads bytes to a specified bucket and object name.

        Args:
            bucket_name (str): The name of the bucket to upload the bytes to.
            object_name (str): The name of the object in the bucket.
            byte_array (bytes): The bytes data to upload.

        Returns:
            Optional[str]: The ETag of the uploaded object if successful, None otherwise.

        Raises:
            S3Error: If there is an error uploading the bytes.
        """
        bucket_name = self.sanitize_bucket_name(bucket_name)

        if not self.check_connection():
            self.reset_connection()
        if not self.create_public_bucket(bucket_name):
            return None
        try:
            data_stream = io.BytesIO(byte_array)
            result = self.client.put_object(
                bucket_name, object_name, data_stream, len(byte_array))
            logging.info(f"Uploaded {object_name} from bytes to {bucket_name}")
            return result.etag
        except S3Error as exc:
            logging.error(f"Error uploading bytes: {str(exc)}")
            return None

    def create_signed_url(self, bucket_name: str, object_name: str, duration: int = MINIO_FILE_DURATION) -> Optional[str]:
        """
        Creates a signed URL for accessing an object in an S3 bucket.

        Args:
            bucket_name (str): The name of the bucket.
            object_name (str): The name of the object.
            duration (int): The duration in days for which the signed URL is valid.

        Returns:
            Optional[str]: The signed URL if successful, None otherwise.

        Raises:
            S3Error: If there is an error generating the signed URL.

        Logs:
            - Generated signed URL if successful.
            - Error message if there is an error generating the signed URL.
        """
        bucket_name = self.sanitize_bucket_name(bucket_name)

        if not self.check_connection():
            self.reset_connection()
        try:
            url = self.client.presigned_get_object(
                bucket_name, object_name, expires=timedelta(days=duration))
            logging.info(f"Generated signed URL: {url}")
            return url
        except S3Error as exc:
            logging.error(f"Error generating signed URL: {str(exc)}")
            return None
        
    def prune_bucket(self, bucket_name: str, days_to_keep: int) -> None:
        """
        Deletes objects in a specified bucket that are older than a given number of days.

        Args:
            bucket_name (str): The name of the bucket to prune.
            days_to_keep (int): The number of days to keep objects in the bucket.

        Returns:
            None

        Raises:
            None

        Logs:
            - Information about deleted objects.

        Example:
            >>> prune_bucket('my_bucket', 7)
            Deleted: old_file.txt on my_bucket
            Deleted: old_folder/old_file.txt on my_bucket
        """
        bucket_name = self.sanitize_bucket_name(bucket_name)

        cut_off_date = datetime.now(datetime.UTC) - timedelta(days=days_to_keep)

        objects = self.client.list_objects(bucket_name)

        for obj in objects:
            object_date = datetime.strptime(obj.last_modified, '%Y-%m-%dT%H:%M:%S.%fZ')
            if object_date < cut_off_date:
                # Supprimer les fichiers plus vieux que la date limite
                self.client.remove_object(bucket_name, obj.object_name)
                logger.info(f'Deleted: {obj.object_name} on {bucket_name}')


    def sanitize_bucket_name(self, input_name: str) -> str:
        # Convertir en minuscules
        sanitized = input_name.lower()

        # Remplacer les caractères non désirés par des tirets
        sanitized = re.sub(r'[^a-z0-9-]', '-', sanitized)

        # Remplacer les tirets multiples par un seul tiret
        sanitized = re.sub(r'-+', '-', sanitized)

        # Supprimer les tirets en début et en fin de chaîne
        sanitized = sanitized.strip('-')

        # Assurer que la longueur est entre 3 et 63 caractères
        if len(sanitized) < 3:
            # Ajoute un suffixe pour atteindre la longueur minimale
            sanitized = sanitized + 'abc'
        elif len(sanitized) > 63:
            sanitized = sanitized[:63]  # Tronquer à la longueur maximale

        return sanitized

    def sanitize_filename(self, input_name: str, file_extension: str = None) -> str:
        # Liste des noms de fichiers réservés sous Windows
        reserved_names = {"CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4", "COM5", "COM6",
                        "COM7", "COM8", "COM9", "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9"}

        # Supprimer les caractères interdits
        sanitized = re.sub(r'[\\/*?:"<>|]', '', input_name)

        # Remplacer les espaces multiples par un seul espace
        sanitized = re.sub(r'\s+', ' ', sanitized).strip()

        # Tronquer à 255 caractères
        sanitized = sanitized[:255]

        # Vérifier si le nom est un nom réservé
        if sanitized.upper() in reserved_names:
            sanitized = '_' + sanitized

        # Assurer que le nom de fichier n'est pas vide
        if sanitized == '':
            sanitized = 'default_filename'

        return f"{sanitized}.{file_extension}" if file_extension else sanitized

    def create_public_bucket(self, bucket_name:str):
        """
        Creates a public bucket with the given name.

        Args:
            bucket_name (str): The name of the bucket to create.

        Returns:
            bool: True if the bucket is created successfully, False otherwise.

        Raises:
            S3Error: If there is an error creating the bucket or setting the policy.

        This function checks the connection to the storage client and resets it if necessary. 
        It then checks if the bucket already exists and creates it if it doesn't. 
        After creating the bucket, it sets the policy to make it public by allowing any principal to perform the "s3:GetObject" action on the bucket's objects.
        If there is an error creating the bucket or setting the policy, it logs the error and returns None.
        """
        bucket_name = self.sanitize_bucket_name(bucket_name)
        if not self.check_connection():
            self.reset_connection()
        try:
            if not self.client.bucket_exists(bucket_name):
                self.client.make_bucket(bucket_name)
                logger.info(f"Bucket '{bucket_name}' created.")
            else:
                logger.info(f"Bucket '{bucket_name}' already exists.")
        except S3Error as exc:
            logger.info(f"Erreur lors de la création du bucket {bucket_name}: {str(exc)}")
            return None

        # Définir la politique du bucket pour le rendre public
        policy_read_only = """
        {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": "*",
                    "Action": "s3:GetObject",
                    "Resource": "arn:aws:s3:::%s/*"
                }
            ]
        }
        """ % bucket_name

        try:
            self.client.set_bucket_policy(bucket_name, policy_read_only)
            logger.info(f"Politique publique appliquée au bucket '{bucket_name}'.")
        except S3Error as exc:
            logger.info(f"Error setting bucket policy : {str(exc)}")
            return None
        return True

# Usage example
if __name__ == "__main__":
    storage = Storage()
    # Example usage of the different upload methods
    #storage.upload_file('mybucket', 'example.jpg', '/path/to/example.jpg')
    #storage.upload_from_base64(
    #    'mybucket', 'example_from_base64.jpg', 'base64_string_here')
    storage.upload_from_bytes(
        'mybucket', 'example_from_bytes.jpg', b'byte_array_here')
    signed_url = storage.create_signed_url(
        'mybucket', 'example_from_bytes.jpg', 1)
    if signed_url:
        logging.info(f"Signed URL for secure access: {signed_url}")
