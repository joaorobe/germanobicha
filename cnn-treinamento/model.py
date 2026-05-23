import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            # Bloco 1
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Bloco 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Bloco 3 (Ajuda a extrair padrões mais complexos)
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            
            # Desliga 50% dos neurônios aleatoriamente no treino para evitar overfitting
            nn.Dropout(0.5), 
            nn.Linear(128 * 16 * 16, 128),
            nn.ReLU(),
            
            # Mais um dropout mais leve
            nn.Dropout(0.3),
            nn.Linear(128, 4)
        )

    def forward(self, x):
        return self.network(x)