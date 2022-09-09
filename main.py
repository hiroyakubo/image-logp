import os
import glob
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from rdkit import Chem
from rdkit.Chem import Crippen, Draw
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.models.feature_extraction import create_feature_extractor


batch_size = 2
lr = 0.001
momentum = 0.9
epochs = 15
model_name = "googlenet"
device = "cuda"
data_dir = "data"
weight_dir = "weight"
model_path = os.path.join(weight_dir, "model.pt")


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir:str, input_file:str) -> None:
        self.data_dir = data_dir
        self.file_stem = os.path.splitext(input_file)[0]
        self.df = pd.read_csv(os.path.join(self.data_dir, input_file))
        self.image_dir = os.path.join(self.data_dir, self.file_stem)
        os.makedirs(self.image_dir, exist_ok=True)
    

    def create_image(self, size:tuple=(224, 224)) -> None:
        """Create image from smiles

        Parameters
        ------
        size : tuple
            image size
        """
        data = {}
        pbar = tqdm(enumerate(self.df["SMILES"].to_numpy()), total=len(self.df))
        for i, smiles in pbar:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                idx = str(i).zfill(5)
                Draw.MolToFile(mol, os.path.join(self.image_dir, f"{idx}.png"), size=size)

                logp = Crippen.MolLogP(mol)
                data[idx] = [logp, smiles]

        with open(os.path.join(self.data_dir, f"{self.file_stem}_label.json"), "w") as f:
            json.dump(data, f, indent=4)
    

    def create_dataset(self, prepare:bool=False) -> None:
        """Create logp dataset

        Parameters
        ------
        prepare : bool
            if True, save image.
        """
        if prepare:
            self.create_image()

        with open(os.path.join(self.data_dir, f"{self.file_stem}_label.json"), "r") as f:
            label_data = json.load(f)

        image_files = [f for f in glob.glob(os.path.join(self.image_dir, "*.png"))]
        images, labels, self.smiles = [], [], []
        for image_file in image_files:
            idx = os.path.splitext(os.path.basename(image_file))[0]
            label = label_data[idx][0]
            image = cv2.imread(image_file)

            images.append(image)
            labels.append(label)
            self.smiles.append(label_data[idx][1])

        self.images = np.array(images).transpose(0, 3, 1, 2)
        self.labels = np.array(labels).reshape(-1, 1)


    def __len__(self):
        return len(self.images)


    def __getitem__(self, idx):        
        return self.images[idx], self.labels[idx]


def create_model(network:str, load_weight:bool=False):
    """Create pytorch model.
    
    Parameters
    ------
    network : str
        network name to use
    load_weight : bool
        load model weight
    
    Returns
    ------
    torchvision.models : pytorch model
    """
    model = torch.hub.load('pytorch/vision:v0.10.0', network, pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 1)
    model.to(device)

    if load_weight:
        model.load_state_dict(torch.load(model_path))
    
    return model


def plot_distribution(
        pred:np.ndarray, dataset:torch.utils.data.Dataset
    ) -> None:
    """Plot carbon vs logp
    Parameters
    ------
    pred : np.ndarray
        predicted logp
    dataset : torch.utils.data.Dataset
        used dataset
    """
    with open(os.path.join(data_dir, f"{dataset.file_stem}_label.json"), "r") as f:
        _label = json.load(f)
    label = {}
    for v in _label.values():
        label[v[1]] = v[0]

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 6))

    df = dataset.df.copy()
    df["logp"] = df["SMILES"].apply(lambda x: label[x])

    not_carbon = ["Ca", "Cr", "Co", "Cu", "Cs", "Cn", "Ce", "Cf"]
    df["c_count"] = df["SMILES"].apply(lambda x: x.count("C") - sum([x.count(nc) for nc in not_carbon]))
    df.plot.scatter(x="c_count", y="logp", ax=axes[0])
    axes[0].set_title("C vs logp")

    df_pred = pd.DataFrame(pred.flatten(), columns=["pred"])
    df_pred["SMILES"] = dataset.smiles
    df_pred["c_count"] = df_pred["SMILES"].apply(lambda x: x.count("C"))
    df_pred.plot.scatter(x="c_count", y="pred", ax=axes[1])
    axes[1].set_title("C vs prediction")
    
    y_max = max(df["logp"].max(), df_pred["pred"].max()) + 5
    y_min = min(df["logp"].min(), df_pred["pred"].min()) - 5
    axes[0].set_ylim(y_min, y_max)
    axes[1].set_ylim(y_min, y_max)
    
    # plt.show()
    plt.savefig("carbon_logp.png")


def plot_layer(model, testset, ds, plot_num:int=5):
    feature_extractor = create_feature_extractor(model, {"inception3b": "feature"})
    testloader = torch.utils.data.DataLoader(testset, batch_size=1)
    fig, axes = plt.subplots(nrows=2, ncols=plot_num, figsize=(9, 6))
    step = 0
    for i, (inputs, _) in enumerate(testloader):
        inputs = inputs.to(device).float()
        inputs_aug = ds(inputs)
        feature = feature_extractor(inputs_aug)
        fm = feature["feature"]
        
        inp = inputs[0].to("cpu").detach().numpy().transpose(1, 2, 0).astype(int)
        fm = fm[0, 0, :, :].to("cpu").detach().numpy()

        axes[0, i].imshow(inp)
        axes[1, i].imshow(fm)

        step += 1
        if i >= plot_num - 1:
            break

    # plt.show()
    plt.savefig("layer.png")


def train():
    os.makedirs(weight_dir, exist_ok=True)

    data_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    trainset = Dataset(data_dir, "train_input.csv")
    trainset.create_dataset(prepare=False)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size)
    valset = Dataset(data_dir, "test_input.csv")
    valset.create_dataset(prepare=False)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size)

    model = create_model(model_name)
    model = model.to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    writer = SummaryWriter()

    print("start training.")
    for epoch in range(epochs):
        step = 0
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs = inputs.to(device).float()
            labels = labels.to(device).float()

            inputs = data_transform(inputs)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            step += 1
        
        epoch_loss = running_loss / len(trainloader)
        mse = val(valloader, model, silent=True)

        writer.add_scalar("loss/train", epoch_loss, epoch)
        writer.add_scalar("loss/validation", mse, epoch)
        print(f"epoch: {epoch}, train loss: {epoch_loss:.2f}, validation loss: {mse:.2f}")

        torch.save(model.state_dict(), model_path)


def val(valloader=None, model=None, silent=False, plot=False):
    if valloader is None:
        valset = Dataset(data_dir, "test_input.csv")
        valset.create_dataset(prepare=False)
        valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size)

    data_transform = transforms.Compose([
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if model is None:
        model = create_model(model_name, load_weight=True)
        model.eval()

    criterion = torch.nn.MSELoss()

    loss_value = 0
    if silent:
        pbar = valloader
    else:
        pbar = tqdm(valloader)

    output_list = []
    for inputs, labels in pbar:
        inputs = inputs.to(device).float()
        labels = labels.to(device).float()

        inputs = data_transform(inputs)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss_value += loss.item()
        output_list.append(outputs.to("cpu").detach().numpy().copy())
    
    mse = loss_value / len(valloader)
    if not silent:
        print(f"validation mse: {mse}")
    
    if plot:
        output_array = np.array(output_list)
        plot_distribution(output_array, valset)
        plot_layer(model, valset, data_transform)

    return mse


if __name__ == "__main__":
    train()
    # val(plot=True)
