import pytorch_lightning as pl
from pytorch_lightning import Trainer
from model import SimpleCNN
from datamodule import EuroSATDataModule
import torch
from PIL import Image
import torchvision.transforms as transforms

# Configuración
data_dir = 'D:/GitHub/eurosat_project/data'
batch_size = 32
num_epochs = 10

# Inicializar el DataModule
data_module = EuroSATDataModule(data_dir=data_dir, batch_size=batch_size)

# Inicializar el modelo
model = SimpleCNN(num_classes=10)

# Entrenador de PyTorch Lightning
trainer = Trainer(max_epochs=num_epochs)

# Entrenamiento
trainer.fit(model, data_module)

# Función para predecir la clase de una imagen
def predict_image(model, image_path, class_to_idx):
    # Cargar la imagen
    image = Image.open(image_path)

    # Transformar la imagen (debe ser el mismo pipeline de transformaciones que usaste para entrenar)
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    
    image = transform(image).unsqueeze(0)  # Añadir dimensión de batch

    # Colocar el modelo en modo evaluación
    model.eval()

    # Deshabilitar el cálculo de gradientes
    with torch.no_grad():
        outputs = model(image)

    # Obtener la clase con la mayor probabilidad
    _, predicted = torch.max(outputs, 1)

    # Mapeo inverso del índice al nombre de la clase
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    predicted_class = idx_to_class[predicted.item()]

    return predicted_class

# Prueba del modelo con una imagen específica
if __name__ == "__main__":
    # Ruta de la imagen que quieres clasificar
    image_path = "D:/GitHub/eurosat_project/data/EuroSAT/2750/Forest/Forest_1234.jpg"  # Cambia esto a la imagen que quieras probar

    # Realizar la predicción
    predicted_class = predict_image(model, image_path, data_module.train_dataset.class_to_idx)
    
    print(f"La imagen fue clasificada como: {predicted_class}")
