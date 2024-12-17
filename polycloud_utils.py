import os
import yaml
from polycloud.client import PolycloudClient

class PolyCloudStorageSupport:
    def __init__(self):
        self.service_provider = os.environ.get("SERVICE_PROVIDER")
        self.region = os.environ.get("REGION")
        self.mlops_bucket = os.environ.get("MLOPS_BUCKET_NAME")

        self.create_polycloud_config()

        polycloud_client = PolycloudClient(config_file_path="config.yaml")
        api = polycloud_client.api(self.service_provider)

        if self.service_provider == "aws":
            storage_key = "s3"
        elif self.service_provider == "gcp":
            storage_key = "gcp-storage"
        elif self.service_provider == "azure":
            storage_key = "azure-storage"
        else:
            raise Exception("The configuration for the given service provider doesn't exist!")

        self.storage = api.storage(storage_key)

    def create_polycloud_config(self):
        config = None
        if self.service_provider == "aws":
            config = {
                "aws": {
                    "use_key_vault": "True",
                    "key_vault": {"region": self.region},
                    "blob": {"s3": {"container_name": self.mlops_bucket} if self.mlops_bucket else {}},
                }
            }
        elif self.service_provider == "gcp":
            project_id = os.environ.get("GCP_PROJECT_ID")
            config = {
                "gcp": {
                    "use_key_vault": "True",
                    "key_vault": {"region": self.region, "project_id": project_id},
                    "blob": {"gcp-storage": {"project_id": project_id, "container_name": self.mlops_bucket} if self.mlops_bucket else {}},
                }
            }
        elif self.service_provider == "azure":
            client_id = os.environ.get("AZURE_CLIENT_ID")
            client_secret = os.environ.get("AZURE_CLIENT_SECRET")
            tenant = os.environ.get("AZURE_TENANT_ID")
            vault_url = os.environ.get("AZURE_VAULT_URL")
            connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
            config = {
                "azure": {
                    "use_key_vault": "True",
                    "key_vault": {
                        "client_id": client_id,
                        "client_secret": client_secret,
                        "tenant": tenant,
                        "vault_url": vault_url,
                    },
                    "blob": {
                        "azure-storage": {
                            "container_name": self.mlops_bucket,
                            "connection_string": connection_string
                        }
                        if self.mlops_bucket
                        else {}
                    },
                }
            }

        with open('config.yaml', 'w') as config_file:
            yaml.dump(config, config_file, default_flow_style=False)

    def upload_file_to_cloud(self, file, key):
        self.storage.upload_file_object(file, key)

        if self.service_provider == "aws":
            dataset_path = f"s3://{self.mlops_bucket}/{key}"
        elif self.service_provider == "gcp":
            dataset_path = f"gs://{self.mlops_bucket}/{key}"
        elif self.service_provider == "azure":
            dataset_path = f"abfs://{self.mlops_bucket}/{key}"
        else:
            dataset_path = None

        return dataset_path

    def read_file_from_cloud(self, key):
        try:
            content = self.storage.download_file_object(key)
            return content
        except Exception:
            return None
        
    def delete_from_cloud(self, key):
        try:
            content = self.storage.delete_file(key)
            return content
        except Exception:
            return None
    
    def list_blobs(self, prefix):
        try:
            content = self.storage.list_blobs(prefix = prefix)
            return content
        except Exception:
            return None
        
    def delete_folder_from_cloud(self, folder_path):
        try:
            print('hello')
            files_list = self.list_blobs(folder_path)
            print(files_list)
            for file in files_list:
                self.delete_from_cloud(file)

            return True
        
        except Exception:
            return False