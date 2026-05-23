import torch

CONFIG = {
    "dataset_path": "datasets/animals", # Ajuste para o caminho exato do seu Colab
    "batch_size": 16, # Reduzido porque você tem poucas imagens
    "learning_rate": 0.001,
    "epochs": 20, # Aumentado por causa do Data Augmentation

    "optimizer": "adam",

    # Detecta automaticamente se há GPU disponível
    "device": "cuda" if torch.cuda.is_available() else "cpu" 
}