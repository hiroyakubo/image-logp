import time
import json

import boto3
import sagemaker
from sagemaker.pytorch.model import PyTorchModel
from sagemaker import get_execution_role


# role = get_execution_role()
role = "arn:aws:iam::552371609228:role/service-role/AmazonSageMaker-ExecutionRole-20221012T102437"
s3_path="s3://model-bucket1/model.tar.gz"
endpoint_name = "logp1"
region = "us-east-1"


def deploy():
    pytorch_model = sagemaker.pytorch.model.PyTorchModel(
        model_data=s3_path,
        role=role,
        framework_version='1.12',
        py_version="py38",
        entry_point="inference.py"
    )
    deploy_params = {
        'endpoint_name'          : endpoint_name,
        'instance_type'          : 'ml.t2.large',
        # 'instance_type'          : 'ml.p2.xlarge',
        'initial_instance_count' : 1, 
    }
    predictor = pytorch_model.deploy(**deploy_params)


def predict():
    client = boto3.client("sagemaker-runtime", region_name=region)

    response = client.invoke_endpoint(
        EndpointName=endpoint_name,
        Body=json.dumps({"smiles": "C1CC2(C(=O)C=CC2=O)C3=CC=CC=C31"}),
        ContentType='application/json'
    )

    _output = response['Body'].read()
    output = json.loads(_output)
    print(output)


def delete():
    sagemaker_client = boto3.client('sagemaker', region_name=region)

    response = sagemaker_client.describe_endpoint_config(EndpointConfigName=endpoint_name)
    endpoint_config_name = response["EndpointConfigName"]

    model_idx = 0
    model_name = response["ProductionVariants"][model_idx]["ModelName"]

    try:
        sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
        print(f"{endpoint_name} deleted.")
    except:
        print(f"{endpoint_name} not found.")

    try:
        sagemaker_client.delete_endpoint_config(EndpointConfigName=endpoint_config_name)
        print(f"{endpoint_config_name} deleted.")
    except:
        print(f"{endpoint_config_name} not found.")

    try:
        sagemaker_client.delete_model(ModelName=model_name)
        print(f"{model_name} deleted.")
    except:
        print(f"{model_name} not found.")


def main():
    # deploy()
    # predict()
    delete()

if __name__ == "__main__":
    main()