import json
import os

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Draw


image_size = (224, 224)
model_name = "googlenet"
device = "cpu"


def input_fn(input_data:str, content_type:str) -> str:
    assert content_type == "application/json"
    request = json.loads(input_data)
    return request['smiles']


def model_fn(model_dir:str):
    model = torch.hub.load('pytorch/vision:v0.10.0', model_name, pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 1)
    model.load_state_dict(
        torch.load(os.path.join(model_dir, "model.pth"), 
                   map_location=torch.device(device)))
    model.to(device)
    model.eval()
    return model


def predict_fn(data, model) -> dict:
    mol = Chem.MolFromSmiles(data)
    if mol is not None:
        img = np.array(Draw.MolsToGridImage([mol], molsPerRow=1))
    else:
        raise Exception("Invalid smiles.")

    img = img[np.newaxis, ...].transpose(0, 3, 1, 2)
    img = torch.Tensor(img)
    img = img.to(device)

    output = model(img).to('cpu').detach().numpy().copy()
    return {"prediction": str(output[0, 0])}


def output_fn(prediction:float, accept:str):
    return json.dumps(prediction), accept


if __name__ == "__main__":
    input_data = '{"smiles":"C1CC2(C(=O)C=CC2=O)C3=CC=CC=C31"}'
    ct = "application/json"
    i = input_fn(input_data, ct)
    m = model_fn("weight")
    o = predict_fn(i, m)
    j, a = output_fn(o, ct)
    print(j, a)