import os
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule

class SimpleCNN(LightningModule):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = torch.nn.Linear(32 * 16 * 16, 128)  # 32 canales de 16x16 tras max pooling
        self.fc2 = torch.nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)  # Reduce la dimensión a 32x32
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)  # Reduce la dimensión a 16x16
        x = x.view(x.size(0), -1)  # Aplana el tensor para la capa totalmente conectada
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        images, labels = batch  # Desempaqueta el lote en imágenes y etiquetas
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)  # Calcula la pérdida
        self.log('train_loss', loss)  # Registro de la pérdida
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)
        self.log('val_loss', loss)  # Registro de la pérdida de validación
        return loss