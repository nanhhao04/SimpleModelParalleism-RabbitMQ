# data.py
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

def load_data(path = "config.yaml"):
    with open(path, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        return cfg["data"]

def get_dataloader(dataset_name="MNIST", batch_size=32, train=True):
    if dataset_name.upper() == "MNIST":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        dataset = datasets.MNIST(
            root="./data",
            train=train,
            download=True,
            transform=transform
        )

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return loader

    else:
        raise ValueError(f"Dataset {dataset_name} chưa được hỗ trợ.")

def load_custom_data(X, y, batch_size=32, shuffle=True):
    if not torch.is_tensor(X):
        X = torch.tensor(X, dtype=torch.float32)
    if not torch.is_tensor(y):
        y = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader


if __name__ == "__main__":
    loader = get_dataloader("MNIST", batch_size=16)
    for imgs, labels in loader:
        print(imgs.shape, labels.shape)
        break
