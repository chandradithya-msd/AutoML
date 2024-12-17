# AutoML

AutoML is a Automated Machine Learning library that is used to solve classical Machine Learning problems without the need of much interaction with the user. It automates different steps in the ML lifecycle such as Data Preprocessing, Model Training, and Model Evaluation.

## Installation and Testing

### Create a virtual env 
```bash
$ python3 -m venv venv
$ source venv/bin/activate
```

### Clone the repository
```bash
$ git clone https://github.com/mad-street-den/vue-pre-sdk.git
$ cd AutoML
```

### Install required dependencies
```bash
$ pip3 install -r requirements.txt
```

### Export Environment variables
```bash
$ export MLOPS_HOST=<MLOPS ENDPOINT>
$ export MLFLOW_HOST=<MLFLOW ENDPOINT>       
$ export SERVICE_PROVIDER=<SERVICE_PROVIDER>
$ export MLOPS_BUCKET_NAME=<MLOPS_BUCKET_NAME>
```
For polycloud installation you need to specify the following environment variables
```bash
$ export PYPI_USERNAME=<PYPI_USERNAME>
$ export PYPI_PASSWORD=<PYPI_PASSWORD>       
$ export PYPI_IP=<PYPI_IP>
```

Based on the service provider that you need to use you need to specify the following environmental variables

#### AWS
```bash
$ export REGION=<REGION>
$ export AWS_ACCESS_KEY_ID=<AWS_ACCESS_KEY_ID>
$ export AWS_SECRET_ACCESS_KEY=<AWS_SECRET_ACCESS_KEY>
$ export AWS_SESSION_TOKEN=<AWS_SESSION_TOKEN>
```
#### AZURE
```bash
$ export AZURE_CLIENT_ID=<AZURE_CLIENT_ID>
$ export AZURE_CLIENT_SECRET=<AZURE_CLIENT_SECRET>       
$ export AZURE_TENANT_ID=<AZURE_TENANT_ID>
$ export AZURE_VAULT_URL=<AZURE_VAULT_URL>
$ export AZURE_STORAGE_CONNECTION_STRING=<AZURE_STORAGE_CONNECTION_STRING>
```

### Install the Polycloud library
```bash
$ pip install --index-url http://${PYPI_USERNAME}:${PYPI_PASSWORD}@${PYPI_IP}/simple/ --trusted-host ${PYPI_IP} polycloud==0.2.22
```

The environmental setup is done. Refer the following files to know about the concepts and how to use AutoML.

    1. notebooks/AutoML-Concepts.ipynb
    2. notebooks/AutoML-Getting-started.ipynb
