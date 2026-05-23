import os
import torch
import torch.nn as nn
import torch.optim as optim
from config import CONFIG
from data import load_data
from model import CNN

def train():
    print(f"Iniciando treinamento no device: {CONFIG['device']}")
    
    # Carrega os dados (buffalo, cat, dog, panda)
    train_loader, test_loader = load_data(CONFIG["dataset_path"], CONFIG["batch_size"])
    classes = train_loader.dataset.classes 
    print(f"Classes encontradas: {classes}")
    
    model = CNN().to(CONFIG["device"])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
        
    # Loop de Treinamento
    epochs = CONFIG["epochs"]
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(CONFIG["device"]), labels.to(CONFIG["device"])
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f"Época [{epoch+1}/{epochs}] - Loss: {running_loss/len(train_loader):.4f}")
        
    os.makedirs("models_saved", exist_ok=True)
    
    # Salva no formato que o server.js do professor exige
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "classes": classes,
        "config": {"input_size": 128}
    }
    
    torch.save(checkpoint, "models_saved/model.pth")
    print("\n Treinamento concluído! Modelo salvo em models_saved/model.pth")

if __name__ == "__main__":
    train()