from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env file

class Settings(BaseSettings):
    azure_openai_endpoint: str = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_openai_api_key: str = os.getenv("AZURE_OPENAI_API_KEY")
    azure_openai_deployment: str = os.getenv("AZURE_OPENAI_DEPLOYMENT")

    azure_communication_connection_string: str = ""
    email_sender_address: str = ""

    azure_storage_connection_string: str = ""
    artifacts_container: str = "lineage-artifacts"

    cosmos_endpoint: str = ""
    cosmos_key: str = ""
    cosmos_database: str = "lineage-agent"
    cosmos_container: str = "checkpoints"

    max_sme_iterations: int = 3

    # Alation
    alation_base_url: str = ""           # e.g. https://your-org.alationcloud.com
    alation_api_token: str = ""
    alation_lineage_parent_path: str = "/Lineage Agent"

    # Collibra
    collibra_base_url: str = ""          # e.g. https://your-org.collibra.com
    collibra_username: str = ""
    collibra_password: str = ""
    # UUID of the "Technical lineage" relation type in your Collibra instance
    collibra_lineage_type_id: str = "00000000-0000-0000-0000-000000003115"
    # Domain UUID where assets will be created
    collibra_domain_id: str = ""

    model_config = {"env_file": ".env"}


settings = Settings()
