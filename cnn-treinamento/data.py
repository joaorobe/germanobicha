import os
import shutil
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def remove_hidden_folders(path):
    for root, dirs, files in os.walk(path):
        for dir_name in dirs:
            if dir_name.startswith("."):
                shutil.rmtree(os.path.join(root, dir_name), ignore_errors=True)

def load_data(dataset_path, batch_size):
    remove_hidden_folders(dataset_path)

    # Transformações de Treino: Aqui a mágica acontece para poucos dados
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15), # Gira a imagem em até 15 graus
        transforms.ToTensor()
    ])

    # Transformações de Teste: Apenas redimensionar, sem bagunçar a imagem
    test_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    train_dataset = datasets.ImageFolder(
        root=f"{dataset_path}/train",
        transform=train_transform
    )

    test_dataset = datasets.ImageFolder(
        root=f"{dataset_path}/test",
        transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size
    )

    return train_loader, test_loader