import requests

class MlopsClient(): 
    def __init__(self, url): 
        self.headers = {
            "Content-Type": "application/json",
            "x-client-name": "AutoML-Users",
            "x-request-id": "r1",
            "x-correlation-id": "c1"
        }
        self.url = url
    
    def get_config(self, type, entity_id):
        URL = f"{self.url}/api/v1/mlops/{type}/{entity_id}"
        response = requests.get(URL, headers=self.headers)
        if response.status_code==201 or response.status_code==200: 
            return response.json()
        else: 
            return {}
        
    def list_config(self, type, params={}): 
        URL = f"{self.url}/api/v1/mlops/{type}"
        response = requests.get(URL, headers=self.headers, params=params)
        if response.status_code==201 or response.status_code==200: 
            return response.json()
        else: 
            return {}

    def set_config(self, type, config): 
        URL = f"{self.url}/api/v1/mlops/{type}"
        response = requests.post(URL, headers=self.headers, json=config)

        if response.status_code==201 or response.status_code==200: 
            return response.json()
        else: 
            return {}

    def delete_config(self, type, entity_id): 
        URL = f"{self.url}/api/v1/mlops/{type}/{entity_id}"
        response = requests.delete(URL, headers=self.headers)

        if response.status_code==201 or response.status_code==200: 
            return response.json()
        else: 
            return {}

    def update_config(self, type, entity_id, config): 
        URL = f"{self.url}/api/v1/mlops/{type}/{entity_id}"
        response = requests.patch(URL, headers=self.headers, json = config)

        if response.status_code==201 or response.status_code==200: 
            return response.json()
        else: 
            return {}