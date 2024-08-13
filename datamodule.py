import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataset import EuroSATDataset
import torchvision.transforms as transforms

class EuroSATDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=32, num_workers=0):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])

        self.train_dataset = EuroSATDataset(root_dir=self.data_dir, split='train', transform=transform)
        self.val_dataset = EuroSATDataset(root_dir=self.data_dir, split='val', transform=transform)
        self.test_dataset = EuroSATDataset(root_dir=self.data_dir, split='test', transform=transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
