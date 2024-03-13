import logging
from rich.logging import RichHandler

# Configuration de base du logger
logging.basicConfig(
    level="INFO", 
    format="%(message)s", 
    datefmt="[%X]", 
    handlers=[RichHandler()]
)

logger = logging.getLogger("nom_de_ton_logger")

# Exemple d'utilisation
logger.info("Ceci est un message d'info.")
logger.warning("Ceci est un avertissement.")

