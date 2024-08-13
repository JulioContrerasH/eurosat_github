import os
from torch.utils.data import Dataset
from PIL import Image
import torch
import torchvision.transforms as transforms

class EuroSATDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir (string): Directorio raíz de los datos.
            split (string): 'train', 'val' o 'test'.
            transform (callable, optional): Transformaciones aplicadas a las imágenes.
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        # Leer los archivos del split correspondiente
        split_file = os.path.join(root_dir, f'eurosat-{split}.txt')
        with open(split_file, 'r') as f:
            self.image_files = f.read().splitlines()

        # Crear un diccionario para mapear nombres de clase a índices
        self.class_to_idx = {
            'AnnualCrop': 0,
            'Forest': 1,
            'HerbaceousVegetation': 2,
            'Highway': 3,
            'Industrial': 4,
            'Pasture': 5,
            'PermanentCrop': 6,
            'Residential': 7,
            'River': 8,
            'SeaLake': 9
        }

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Obtener el nombre de la imagen
        img_name = self.image_files[idx]

        # Extraer el nombre de la clase desde el nombre de la imagen
        class_name = img_name.split('_')[0]
        
        # Construir la ruta completa a la imagen
        img_path = os.path.join(self.root_dir, 'EuroSAT', '2750', class_name, img_name)
        
        # Leer la imagen con PIL
        image = Image.open(img_path)

        # Aplicar las transformaciones a la imagen
        if self.transform:
            image = self.transform(image)

        # Obtener la etiqueta como un índice basado en el nombre de la clase
        label = self.class_to_idx[class_name]
        label = torch.tensor(label)  # Convertir la etiqueta a Tensor

        return image, label